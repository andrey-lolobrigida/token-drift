"""Token category heuristics. Rules are in docs/EXPERIMENT.md; first match wins."""
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
