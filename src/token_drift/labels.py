"""Cheap surface-form categories for tokens. First matching rule wins.

These exist for two jobs: coloring the UMAP plots and a rough silhouette score.
They are deliberately dumb. See docs/EXPERIMENT.md "Token categories".
"""
from __future__ import annotations

import unicodedata
from collections.abc import Iterable

import numpy as np

CATEGORIES = [
    "special",
    "byte",
    "nonascii",
    "digit",
    "punct",
    "space_cap",
    "space_lower",
    "cap",
    "lower",
    "mixed",
]
_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

# Pythia's tokenizer has a handful of these; other tokenizers should pass their own.
_DEFAULT_SPECIAL = {"<|endoftext|>", "<|padding|>"}


def _is_punct_or_space(ch: str) -> bool:
    # Unicode categories: P* = punctuation, S* = symbols, Z* = separators, C* = control.
    # Symbols count as punct here because "$" and "+" behave like punctuation in a vocab.
    return unicodedata.category(ch)[0] in "PSZC"


def categorize(tok: str, special: Iterable[str] | None = None) -> str:
    special_set = set(special) if special is not None else _DEFAULT_SPECIAL
    if tok in special_set:
        return "special"
    # The tokenizer emits U+FFFD for byte-fallback tokens that don't decode to valid UTF-8.
    if "�" in tok:
        return "byte"
    if any(ord(ch) > 127 for ch in tok):
        return "nonascii"
    if not tok:
        return "mixed"
    body = tok[1:] if tok.startswith(" ") else tok
    if body and body.isdigit():
        return "digit"
    if all(_is_punct_or_space(ch) for ch in tok):
        return "punct"
    leading_space = tok.startswith(" ")
    if body.isalpha():
        # "capitalized" = first letter upper, rest lower ("The" yes, "USA" no -> mixed)
        if body[0].isupper() and body[1:].islower() or len(body) == 1 and body.isupper():
            return "space_cap" if leading_space else "cap"
        if body.islower():
            return "space_lower" if leading_space else "lower"
    return "mixed"


def categorize_all(tokens: Iterable[str], special: Iterable[str] | None = None) -> np.ndarray:
    """Vector of category indices into CATEGORIES, one per token."""
    return np.array([_INDEX[categorize(t, special)] for t in tokens], dtype=np.int8)


# ---- frequency proxy: BPE merge rank ----
# A BPE tokenizer learns merges most-frequent-pair first, so the index of the merge that
# *produces* a token is a decent stand-in for its corpus frequency without needing a
# corpus. Frequency is famously one of the top PCs of any embedding matrix, so we want
# it available for coloring and for Voita-style "do frequent tokens change more" checks.


def merge_ranks(tokens: Iterable[str], merges: Iterable[str | tuple[str, str]]) -> np.ndarray:
    """Rank of the merge that creates each token string; -1 for base tokens (never merged).

    `merges` is the tokenizer's merge list in learned order, as "a b" strings or pairs.
    Token strings must be in the tokenizer's *internal* alphabet (e.g. 'Ġthe'), not decoded.
    """
    rank: dict[str, int] = {}
    for i, m in enumerate(merges):
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        rank.setdefault(a + b, i)  # first merge that produces this string wins
    return np.array([rank.get(t, -1) for t in tokens], dtype=np.int64)


def freq_bins(ranks: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Quantile-bin merge ranks into 1..n_bins (1 = earliest merges ~ most frequent).

    Base tokens (rank -1: single bytes, and the odd unmergeable entry) go to bin 0: they
    aren't rank-comparable with merged tokens, and the byte ones are mostly *very* common.
    """
    ranks = np.asarray(ranks)
    out = np.zeros(len(ranks), dtype=np.int8)
    merged = ranks >= 0
    if merged.any():
        # rank positions within the merged subset -> equal-count bins
        order = np.argsort(ranks[merged], kind="stable")
        pos = np.empty(merged.sum(), dtype=np.int64)
        pos[order] = np.arange(merged.sum())
        out[merged] = 1 + (pos * n_bins) // merged.sum()
    return out


def corpus_freq_bins(counts: np.ndarray, eligible: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Equal-count bins by *actual* corpus count: 0 = most frequent ... n_bins-1 = rarest.

    Only eligible tokens get a bin; the rest are -1 (knn_change_by_bin never asks for -1,
    so they drop out). Unlike merge rank there's no special base/byte bin: counts are
    comparable across every token.
    """
    counts = np.asarray(counts)
    eligible = np.asarray(eligible, dtype=bool)
    out = np.full(len(counts), -1, dtype=np.int8)
    m = int(eligible.sum())
    if m:
        order = np.argsort(-counts[eligible], kind="stable")
        pos = np.empty(m, dtype=np.int64)
        pos[order] = np.arange(m)
        out[eligible] = (pos * n_bins) // m
    return out
