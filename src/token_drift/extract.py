"""Manufacture a per-layer "vocab embedding" by running every token through the model.

v0 is context-free: input is [BOS, tok], we take the residual at position 1. With only
BOS to attend to this is mostly "what the MLPs do to a token in isolation". See
docs/EXPERIMENT.md for the v1 corpus-averaged version. `extract_corpus_means` is the v1
corpus-averaged version.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from token_drift.labels import merge_ranks


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(name: str, *, random_init: bool, seed: int, device: str, revision: str | None = None):
    """Load the trained model, or the same architecture with fresh random weights.

    `revision` = a Hub branch, i.e. a training checkpoint (Pythia: step0 .. step143000).
    None = main = the finished model.
    """
    if revision is not None and random_init:
        raise ValueError(f"revision={revision!r} with random_init: true: random init never loads any weights")
    torch.manual_seed(seed)
    # always main's tokenizer: Pythia's vocab is the same at every checkpoint
    tokenizer = AutoTokenizer.from_pretrained(name)
    if random_init:
        # from_config runs HF's init (normal, std=initializer_range) instead of loading
        # weights. Seeded above, so the control is reproducible.
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name))
    elif revision is not None:
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision)
    else:
        model = AutoModelForCausalLM.from_pretrained(name)
    return model.to(device).eval(), tokenizer


def missing_revisions(name: str, revisions: list[str]) -> list[str]:
    """The revisions (Hub branches) `name` doesn't have. One API call, no downloads, so a typo
    in the config fails in a second instead of after three checkpoints' worth of work."""
    from huggingface_hub import list_repo_refs  # only checkpoint runs need the network for this

    have = {b.name for b in list_repo_refs(name).branches}
    return [r for r in revisions if r not in have]


def vocab_tokens(tokenizer) -> list[str]:
    """Decoded string for every id the tokenizer actually knows about.

    Pythia's embedding matrix is padded to 50304 rows but the tokenizer has 50277
    entries; we only care about the ones that are real tokens.
    """
    return [tokenizer.decode([i]) for i in range(len(tokenizer))]


def vocab_freq_ranks(tokenizer) -> np.ndarray:
    """BPE merge rank per vocab id (-1 for base tokens). A corpus-free frequency proxy.

    Reads the merge list out of the fast tokenizer's JSON; token strings are matched in
    the tokenizer's internal alphabet ('Ġthe'), not the decoded form.
    """
    model = json.loads(tokenizer.backend_tokenizer.to_str())["model"]
    merges = model.get("merges", [])  # "a b" strings (older) or [a, b] pairs (newer)
    internal = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    return merge_ranks(internal, merges)


def final_norm(model) -> torch.nn.Module:
    """The final LayerNorm, applied after the last block and before the unembed.

    Named differently per architecture; these two are the ones we run.
    """
    base = model.base_model
    for attr in ("final_layer_norm", "ln_f"):  # GPT-NeoX, GPT-2
        if hasattr(base, attr):
            return getattr(base, attr)
    raise ValueError(f"don't know where the final LayerNorm lives in {type(base).__name__}")


def get_final_ln(model) -> tuple[np.ndarray, np.ndarray]:
    """(gain, bias) of the final LayerNorm, float32. For naming the massive dimension."""
    ln = final_norm(model)
    return ln.weight.detach().cpu().float().numpy(), ln.bias.detach().cpu().float().numpy()


@torch.no_grad()
def extract_activations(
    model, token_ids: np.ndarray, *, bos_id: int, batch_size: int, device: str
) -> np.ndarray:
    """-> (n_layers+2, len(token_ids), d_model) float16. Residual stream at position 1.

    Frames: [embed, block 1 .. block L-1, block L pre-LN, block L post-LN].
    HF's hidden_states[-1] is *after* the final LayerNorm, so without the extra frame
    the last step mixes "what block L did" with "what the LN did". A forward pre-hook
    on the LN grabs its input, which is the raw residual after block L.
    """
    token_ids = np.asarray(token_ids)
    n = len(token_ids)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    out = np.empty((n_layers + 2, n, d), dtype=np.float16)
    grabbed: dict[str, torch.Tensor] = {}
    hook = final_norm(model).register_forward_pre_hook(
        lambda mod, args: grabbed.__setitem__("pre_ln", args[0])
    )
    try:
        for start in tqdm(range(0, n, batch_size), desc="extract", unit="batch"):
            ids = torch.as_tensor(token_ids[start : start + batch_size], device=device)
            inp = torch.stack([torch.full_like(ids, bos_id), ids], dim=1)  # (b, 2)
            hs = model(input_ids=inp, output_hidden_states=True).hidden_states
            frames = [*hs[:-1], grabbed.pop("pre_ln"), hs[-1]]
            for layer, h in enumerate(frames):
                # position 1 = the token. Position 0 is BOS and identical for every row.
                out[layer, start : start + batch_size] = h[:, 1, :].to(torch.float16).cpu().numpy()
    finally:
        hook.remove()  # a leftover hook would silently fire on every later forward
    return out


def get_embed_unembed(model) -> tuple[np.ndarray, np.ndarray]:
    """(embed_in, embed_out), both (vocab, d_model) float16. Pythia keeps them untied."""
    embed = model.get_input_embeddings().weight.detach().cpu().to(torch.float16).numpy()
    unembed = model.get_output_embeddings().weight.detach().cpu().to(torch.float16).numpy()
    return embed, unembed


def self_similarity(sq_norm: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Mean pairwise cosine of a token's occurrences, from the norm of their unit-vector sum.

    ||u_1 + ... + u_n||^2 = n (each vector with itself) + sum over ordered pairs i != j of
    cos_ij, so the mean over the n(n-1) pairs falls out without storing any occurrence.
    Ethayarajh (2019) self-similarity. NaN for n < 2 (no pairs). Broadcasts (F, V) with (V,).
    """
    sq_norm = np.asarray(sq_norm, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (sq_norm - n) / (n * (n - 1))
    return np.where(n >= 2, s, np.nan)


@torch.no_grad()
def extract_corpus_means(
    model, windows: np.ndarray, *, eos_id: int, min_context: int, batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Per-token mean residual over real text, same frames as `extract_activations`.

    -> (unit_mean, raw_mean, counts, self_sim, self_sim_baseline). Means are
    (n_layers+2, model vocab rows, d) float16; zero-count rows are zero.

    Two running sums per frame live on the device in float32 (2 x 8 x 50,304 x 512 x 4 B =
    1.65 GB for Pythia). unit_mean averages unit-normed occurrences: an occasional ordinary
    token turns into an attention sink (norm ~170 vs ~12) and would dominate a plain mean.
    float32 is plenty: " the" gets ~430k additions, relative error ~sqrt(n) x 1e-7 = 1e-4.
    """
    windows = np.asarray(windows)
    if windows.shape[1] <= min_context:
        raise ValueError(
            f"window ({windows.shape[1]}) must be longer than min_context ({min_context}), "
            "or no position ever gets counted"
        )
    n_frames = model.config.num_hidden_layers + 2
    vocab = model.get_input_embeddings().weight.shape[0]
    d = model.config.hidden_size
    sum_unit = torch.zeros(n_frames, vocab, d, device=device)
    sum_raw = torch.zeros(n_frames, vocab, d, device=device)
    counts = torch.zeros(vocab, dtype=torch.int64, device=device)
    pos = torch.arange(windows.shape[1], device=device)
    grabbed: dict[str, torch.Tensor] = {}
    hook = final_norm(model).register_forward_pre_hook(
        lambda mod, args: grabbed.__setitem__("pre_ln", args[0])
    )
    try:
        for start in tqdm(range(0, len(windows), batch_size), desc="extract", unit="batch"):
            ids = torch.as_tensor(windows[start : start + batch_size], dtype=torch.long, device=device)
            # base_model skips the unembed (50,304-wide logits we never look at here) - same
            # hidden_states, way less compute and memory per batch.
            hs = model.base_model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            frames = [*hs[:-1], grabbed.pop("pre_ln"), hs[-1]]
            # early positions and every EOS are attention sinks (L3 norm ~120 vs ~12); skip them
            keep = (pos >= min_context)[None, :] & (ids != eos_id)
            kept = ids[keep]
            counts += torch.bincount(kept, minlength=vocab)
            for f, h in enumerate(frames):
                h = h[keep].float()
                sum_raw[f].index_add_(0, kept, h)
                sum_unit[f].index_add_(0, kept, h / h.norm(dim=1, keepdim=True).clamp_min(1e-8))
    finally:
        hook.remove()

    counts_np = counts.cpu().numpy()
    denom = counts.clamp_min(1).float()[:, None]  # zero-count rows: 0 / 1 = 0, not NaN
    unit_mean = np.empty((n_frames, vocab, d), dtype=np.float16)
    raw_mean = np.empty((n_frames, vocab, d), dtype=np.float16)
    sq = np.empty((n_frames, vocab))
    total_sq = np.empty(n_frames)
    for f in range(n_frames):
        unit_mean[f] = (sum_unit[f] / denom).half().cpu().numpy()
        raw_mean[f] = (sum_raw[f] / denom).half().cpu().numpy()
        s = sum_unit[f].cpu().double()  # float64 on CPU: MPS has none, and ||S||^2 - n cancels a lot
        sq[f] = (s * s).sum(1).numpy()
        tot = s.sum(0)
        total_sq[f] = float(tot @ tot)
    self_sim = self_similarity(sq, counts_np).astype(np.float32)
    # anisotropy baseline: same formula on the grand total, i.e. mean cosine between any two
    # counted occurrences of any tokens. Uncentered, like Ethayarajh's.
    n_all = np.full(n_frames, counts_np.sum())
    baseline = [float(b) for b in self_similarity(total_sq, n_all)]
    return unit_mean, raw_mean, counts_np, self_sim, baseline


@torch.no_grad()
def extract_probe(
    model, tokens: np.ndarray, offsets: np.ndarray, *, pad_id: int, batch_size: int, device: str,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Residual at each window's last token, same frames as `extract_activations`.

    -> (n_layers+2, n_windows, d) float16, in the original window order. Windows are ragged
    (flat `tokens`, window i = tokens[offsets[i]:offsets[i+1]]); each batch is right-padded with
    `pad_id`. That's safe because attention is causal: padding after the last token can't reach
    it and positions before it don't shift. `out` lets the caller hand in a disk-backed array
    (occ.npy is ~2 GB for the real run).
    """
    tokens = np.asarray(tokens)
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    n = len(lengths)
    if n and lengths.min() < 1:
        raise ValueError("empty window in the probe corpus")
    n_frames = model.config.num_hidden_layers + 2
    if out is None:
        out = np.empty((n_frames, n, model.config.hidden_size), dtype=np.float16)
    # longest first: batches hold similar lengths (little padding), and an out-of-memory shows
    # up on batch 1 instead of 15 minutes in
    order = np.argsort(-lengths, kind="stable")
    grabbed: dict[str, torch.Tensor] = {}
    hook = final_norm(model).register_forward_pre_hook(
        lambda mod, args: grabbed.__setitem__("pre_ln", args[0])
    )
    try:
        for s in tqdm(range(0, n, batch_size), desc="extract", unit="batch"):
            idx = order[s : s + batch_size]
            batch = np.full((len(idx), int(lengths[idx].max())), pad_id, dtype=np.int64)
            for r, i in enumerate(idx):
                batch[r, : lengths[i]] = tokens[offsets[i] : offsets[i + 1]]
            ids = torch.as_tensor(batch, device=device)
            hs = model.base_model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            frames = [*hs[:-1], grabbed.pop("pre_ln"), hs[-1]]
            rows = torch.arange(len(idx), device=device)
            last = torch.as_tensor(lengths[idx] - 1, device=device)
            for f, h in enumerate(frames):
                out[f, idx] = h[rows, last].to(torch.float16).cpu().numpy()
    finally:
        hook.remove()
    return out
