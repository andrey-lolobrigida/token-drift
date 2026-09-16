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
