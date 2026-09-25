"""Vocab -> per-layer activations. Uses a tiny in-memory GPT-NeoX; no downloads."""
import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from token_drift.extract import extract_activations, final_norm, get_embed_unembed, get_final_ln

VOCAB, D, LAYERS = 200, 32, 2


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=VOCAB, hidden_size=D, num_hidden_layers=LAYERS, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=16,
    )
    return GPTNeoXForCausalLM(cfg).eval()


def test_shape_and_dtype(model):
    ids = np.arange(VOCAB)
    acts = extract_activations(model, ids, bos_id=0, batch_size=64, device="cpu")
    # embed, blocks 1..L-1, block L pre-LN, block L post-LN
    assert acts.shape == (LAYERS + 2, VOCAB, D)
    assert acts.dtype == np.float16


@torch.no_grad()
def test_last_frame_is_hf_final_hidden_state(model):
    ids = np.array([3, 17, 150])
    acts = extract_activations(model, ids, bos_id=0, batch_size=64, device="cpu")
    inp = torch.stack([torch.zeros(3, dtype=torch.long), torch.as_tensor(ids)], dim=1)
    hf = model(input_ids=inp, output_hidden_states=True).hidden_states[-1][:, 1].numpy()
    np.testing.assert_allclose(acts[-1].astype(np.float32), hf, atol=1e-2)


@torch.no_grad()
def test_pre_ln_frame_is_the_input_to_the_final_layernorm(model):
    # The strong check: the model's own final LN applied to our pre-LN frame must give
    # back the post-LN frame. Proves the hook grabbed the right tensor, not just *a*
    # different one.
    ids = np.arange(VOCAB)
    acts = extract_activations(model, ids, bos_id=0, batch_size=64, device="cpu")
    pre = torch.as_tensor(acts[-2].astype(np.float32))
    np.testing.assert_allclose(final_norm(model)(pre).numpy(), acts[-1].astype(np.float32), atol=2e-2)
    assert not np.allclose(acts[-2], acts[-1], atol=1e-2)


def test_final_ln_params(model):
    gain, bias = get_final_ln(model)
    assert gain.shape == (D,) and bias.shape == (D,)
    np.testing.assert_array_equal(gain, final_norm(model).weight.detach().numpy())


def test_final_norm_finds_gpt2_ln_f():
    from transformers import GPT2Config, GPT2LMHeadModel

    m = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_embd=16, n_layer=1, n_head=2, n_positions=8))
    assert final_norm(m) is m.transformer.ln_f


def test_layer0_is_raw_input_embedding_of_the_token_not_bos(model):
    # For GPT-NeoX (rotary), hidden_states[0] == embed_in[tok]. This also proves we
    # index position 1: position 0 would be embed_in[bos] for every row.
    ids = np.array([3, 17, 150])
    acts = extract_activations(model, ids, bos_id=0, batch_size=64, device="cpu")
    embed, _ = get_embed_unembed(model)
    np.testing.assert_allclose(acts[0].astype(np.float32), embed[ids].astype(np.float32), atol=1e-3)
    assert not np.allclose(acts[0][0], acts[0][1])


def test_batching_does_not_change_results(model):
    ids = np.arange(VOCAB)
    a = extract_activations(model, ids, bos_id=0, batch_size=7, device="cpu")
    b = extract_activations(model, ids, bos_id=0, batch_size=VOCAB, device="cpu")
    np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32), atol=2e-3)


def test_deeper_layers_actually_change_something(model):
    ids = np.arange(VOCAB)
    acts = extract_activations(model, ids, bos_id=0, batch_size=64, device="cpu")
    assert not np.allclose(acts[0], acts[-1], atol=1e-2)


def test_embed_unembed_shapes_and_source(model):
    embed, unembed = get_embed_unembed(model)
    assert embed.shape == (VOCAB, D) and unembed.shape == (VOCAB, D)
    assert embed.dtype == np.float16 and unembed.dtype == np.float16
    # transformers 5 calls this lm_head (was embed_out); go through the accessor
    ref = model.get_output_embeddings().weight.detach().numpy()
    np.testing.assert_allclose(unembed.astype(np.float32), ref, atol=1e-3)
    # the whole experiment leans on these being *untied*
    assert not np.array_equal(embed, unembed)


def test_vocab_freq_ranks_from_a_tiny_bpe_tokenizer():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    from token_drift.extract import vocab_freq_ranks

    bpe = models.BPE(vocab={"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}, merges=[("a", "b"), ("ab", "c")])
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(bpe))
    ranks = vocab_freq_ranks(tok)
    assert ranks.tolist() == [-1, -1, -1, 0, 1]


# ---------- corpus mode (v1): running sums over real windows ----------
from collections import defaultdict

from token_drift.extract import extract_corpus_means, self_similarity

EOS, MIN_CTX = 0, 4


@pytest.fixture(scope="module")
def windows():
    rng = np.random.default_rng(0)
    w = rng.integers(1, 20, size=(6, 16))  # few distinct ids, so every token repeats
    w[:, 5] = EOS
    w[2, 10] = EOS
    return w


@torch.no_grad()
def _occurrences(model, windows):
    """Every counted occurrence, the slow obvious way: {(frame, tok): [vectors]}."""
    grabbed = {}
    hook = final_norm(model).register_forward_pre_hook(lambda m, a: grabbed.__setitem__("pre", a[0]))
    occ = defaultdict(list)
    try:
        for w in windows:
            hs = model(input_ids=torch.as_tensor(w[None], dtype=torch.long), output_hidden_states=True).hidden_states
            frames = [*hs[:-1], grabbed.pop("pre"), hs[-1]]
            for p, t in enumerate(w):
                if p < MIN_CTX or t == EOS:
                    continue
                for f, h in enumerate(frames):
                    occ[f, int(t)].append(h[0, p].numpy().astype(np.float64))
    finally:
        hook.remove()
    return occ


def _means(model, windows, batch_size=4):
    return extract_corpus_means(model, windows, eos_id=EOS, min_context=MIN_CTX,
                                batch_size=batch_size, device="cpu")


def test_corpus_means_match_a_plain_loop(model, windows):
    unit, raw, counts, self_sim, _ = _means(model, windows)
    assert unit.shape == raw.shape == (LAYERS + 2, VOCAB, D) and unit.dtype == np.float16
    occ = _occurrences(model, windows)
    for (f, t), vs in occ.items():
        vs = np.stack(vs)
        assert counts[t] == len(vs)
        np.testing.assert_allclose(raw[f, t], vs.mean(0), atol=2e-3)
        units = vs / np.linalg.norm(vs, axis=1, keepdims=True)
        np.testing.assert_allclose(unit[f, t], units.mean(0), atol=2e-3)
        if len(vs) >= 2:  # self-sim = mean cosine over ordered pairs i != j
            cos = units @ units.T
            brute = (cos.sum() - len(vs)) / (len(vs) * (len(vs) - 1))
            assert self_sim[f, t] == pytest.approx(brute, abs=1e-4)


def test_zero_count_rows_are_zero_and_self_sim_nan(model, windows):
    unit, raw, counts, self_sim, _ = _means(model, windows)
    unseen = counts == 0
    assert unseen.any()  # ids >= 20 never appear
    assert not unit[:, unseen].any() and not raw[:, unseen].any()
    assert np.isnan(self_sim[:, unseen]).all()


def test_frame0_is_the_embedding_so_unit_mean_is_its_direction_and_self_sim_is_one(model, windows):
    unit, raw, counts, self_sim, _ = _means(model, windows)
    seen = counts > 0
    embed, _ = get_embed_unembed(model)
    e = embed[seen].astype(np.float32)
    np.testing.assert_allclose(raw[0, seen].astype(np.float32), e, atol=1e-3)
    np.testing.assert_allclose(unit[0, seen].astype(np.float32), e / np.linalg.norm(e, axis=1, keepdims=True), atol=2e-3)
    np.testing.assert_allclose(self_sim[0, counts >= 2], 1.0, atol=1e-4)


def test_early_positions_and_eos_are_never_counted(model, windows):
    _, _, counts, _, _ = _means(model, windows)
    pos = np.arange(windows.shape[1])
    keep = (pos >= MIN_CTX)[None, :] & (windows != EOS)
    assert counts.sum() == keep.sum()
    assert counts[EOS] == 0
    only_early = np.setdiff1d(np.unique(windows[:, :MIN_CTX]), np.unique(windows[keep]))
    assert (counts[only_early] == 0).all()


def test_corpus_batch_size_does_not_matter(model, windows):
    a = _means(model, windows, batch_size=1)
    b = _means(model, windows, batch_size=6)
    np.testing.assert_allclose(a[0].astype(np.float32), b[0].astype(np.float32), atol=2e-3)
    np.testing.assert_array_equal(a[2], b[2])


def test_baseline_is_mean_cosine_over_all_counted_occurrence_pairs(model, windows):
    *_, baseline = _means(model, windows)
    occ = _occurrences(model, windows)
    assert len(baseline) == LAYERS + 2
    for f in (0, LAYERS + 1):
        vs = np.concatenate([np.stack(v) for (ff, _), v in occ.items() if ff == f])
        u = vs / np.linalg.norm(vs, axis=1, keepdims=True)
        brute = ((u @ u.T).sum() - len(u)) / (len(u) * (len(u) - 1))
        assert baseline[f] == pytest.approx(brute, abs=1e-4)


def test_self_similarity_formula_on_known_vectors():
    u = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # cos pairs: 0, 1, 0 -> mean 1/3
    s = u.sum(0)
    assert self_similarity(np.array([s @ s]), np.array([3]))[0] == pytest.approx(1 / 3)
    assert np.isnan(self_similarity(np.array([1.0]), np.array([1]))[0])


def test_window_no_longer_than_min_context_fails_fast(model):
    w = np.ones((2, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="min_context"):
        extract_corpus_means(model, w, eos_id=EOS, min_context=4, batch_size=2, device="cpu")
