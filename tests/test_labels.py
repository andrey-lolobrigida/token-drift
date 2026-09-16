"""Token category heuristics. Rules are in docs/EXPERIMENT.md; first match wins."""
import numpy as np
import pytest

from token_drift.labels import CATEGORIES, categorize, categorize_all


@pytest.mark.parametrize(
    "tok, expected",
    [
        ("<|endoftext|>", "special"),
        ("<|padding|>", "special"),
        ("�", "byte"),  # replacement char = undecodable byte fallback
        ("é", "nonascii"),
        (" café", "nonascii"),
        ("123", "digit"),
        (" 42", "digit"),
        (".", "punct"),
        (" ", "punct"),
        ("\n", "punct"),
        (" ...", "punct"),
        (" The", "space_cap"),
        (" bank", "space_lower"),
        ("The", "cap"),
        ("ing", "lower"),
        (" 3D", "mixed"),
        ("a1", "mixed"),
        (" Hello!", "mixed"),
        ("", "mixed"),
    ],
)
def test_categorize(tok, expected):
    assert categorize(tok) == expected


def test_special_tokens_can_be_passed_in():
    # a tokenizer's special-token strings win over every other rule
    assert categorize("<bos>", special={"<bos>"}) == "special"


def test_categorize_all_returns_ints_indexing_categories():
    toks = [" The", "ing", "7"]
    ids = categorize_all(toks)
    assert [CATEGORIES[i] for i in ids] == ["space_cap", "lower", "digit"]


def test_categories_are_stable_and_complete():
    assert CATEGORIES == [
        "special", "byte", "nonascii", "digit", "punct",
        "space_cap", "space_lower", "cap", "lower", "mixed",
    ]


# ---- frequency bins from BPE merge rank ----

def test_merge_ranks_from_merges_list():
    from token_drift.labels import merge_ranks

    # merges in the order the tokenizer learned them; earlier merge ~ more frequent
    merges = ["t h", "th e", "i n", "in g"]
    tokens = ["t", "th", "the", "ing", "in", "zzz"]
    ranks = merge_ranks(tokens, merges)
    # base tokens (never produced by a merge) get rank -1; unknown strings too
    assert ranks.tolist() == [-1, 0, 1, 3, 2, -1]


def test_freq_bins_quantile_over_merged_tokens_base_in_bin0():
    from token_drift.labels import freq_bins

    ranks = np.array([-1, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    bins = freq_bins(ranks, n_bins=4)
    assert bins.dtype == np.int8
    assert bins[:2].tolist() == [0, 0]  # base/byte tokens: bin 0, "most frequent"
    assert bins[2:].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]  # quantiles of the rest -> bins 1..n
    assert bins.max() == 4
