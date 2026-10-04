"""A real-but-tiny BPE tokenizer and GPT-NeoX, so corpus-mode extract runs for real offline."""
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

LETTERS = "abcdefghijklmnopqrstuvwxyz"
MERGES = [("t", "h"), ("th", "e"), (" ", "the"), ("i", "n"), (" ", "in"), ("a", "n"), ("o", "n"), ("e", "r")]
WORDS = ["the", "then", "in", "inner", "an", "on", "under", "a", "cat", "sat", "mat", "dog", "ran",
         "far", "quiz", "jump", "box", "vex", "kept", "why", "go", "lol"]


def tiny_tokenizer():
    base = ["<|endoftext|>", " ", "."] + list(LETTERS)
    vocab = {t: i for i, t in enumerate(base)}
    for a, b in MERGES:
        vocab.setdefault(a + b, len(vocab))
    bpe = Tokenizer(models.BPE(vocab=vocab, merges=MERGES))
    return PreTrainedTokenizerFast(
        tokenizer_object=bpe, eos_token="<|endoftext|>", bos_token="<|endoftext|>"
    )


def tiny_model(tok, window: int = 16):
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=len(tok), hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=32, max_position_embeddings=window,
    )
    return GPTNeoXForCausalLM(cfg).eval()


def tiny_docs(n_docs: int = 12, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, size=60)) + "." for _ in range(n_docs)]


@pytest.fixture(scope="session")
def pythia_tok():
    """The real Pythia tokenizer, from the HF cache only (tests never hit the network)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", local_files_only=True)
    except OSError:
        pytest.skip("EleutherAI/pythia-70m tokenizer is not in the HF cache")
