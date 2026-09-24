"""Manufacture a per-layer "vocab embedding" by running every token through the model.

v0 is context-free: input is [BOS, tok], we take the residual at position 1. With only
BOS to attend to this is mostly "what the MLPs do to a token in isolation". See
docs/EXPERIMENT.md for the v1 corpus-averaged version.
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


def build_model(name: str, *, random_init: bool, seed: int, device: str):
    """Load the trained model, or the same architecture with fresh random weights."""
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if random_init:
        # from_config runs HF's init (normal, std=initializer_range) instead of loading
        # weights. Seeded above, so the control is reproducible.
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name))
    else:
        model = AutoModelForCausalLM.from_pretrained(name)
    return model.to(device).eval(), tokenizer


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
