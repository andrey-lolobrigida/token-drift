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
