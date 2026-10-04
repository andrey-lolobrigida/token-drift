"""Milestone B: find a word's occurrences in targeted texts and cut a window ending at each.

Pythia is causal, so a hit's residual only depends on the tokens before it: find the hits
first, then run the model on just those windows instead of on billions of tokens.
Spec: docs/superpowers/specs/2026-10-03-v1-milestone-b-design.md
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

POS = ("noun", "adj")
ROLES = ("deficiency", "mean", "excess")
GROUPS = ("books", "pile")
_BARE = re.compile(r"[a-z][a-z'-]*")


def load_words(path) -> dict:
    """configs/probe_words.yaml -> {"triples": [...], "polysemy": [...], "null_exclude": [...]}.

    Triples come out flat, in file order: {"id", "set", "concept", "pos", "words"} with
    words = (deficiency, mean, excess).
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    triples = []
    for set_name, concepts in (raw.get("sets") or {}).items():
        for concept, by_pos in concepts.items():
            for pos, rows in by_pos.items():
                where = f"{set_name}/{concept}/{pos}"
                if pos not in POS:
                    raise ValueError(f"{where}: part of speech {pos!r} must be one of {POS}")
                for row in rows:
                    if len(row) != 3:
                        raise ValueError(f"{where}: {row} is not deficiency/mean/excess")
                    triples.append({"id": f"{where}/{','.join(row)}", "set": set_name,
                                    "concept": concept, "pos": pos, "words": tuple(row)})
    words = {"triples": triples, "polysemy": list(raw.get("polysemy") or []),
             "null_exclude": list(raw.get("null_exclude") or [])}
    for w in [w for t in triples for w in t["words"]] + words["polysemy"]:
        # capitals and leading spaces are different tokens; the code adds the one space itself
        if not isinstance(w, str) or not _BARE.fullmatch(w):
            raise ValueError(f"probe word {w!r} must be bare lowercase (the space is added by the code)")
    return words


def probe_word_pos(words: dict) -> dict[str, str]:
    """Every triple word -> its part of speech, each word once (courage is in 3 triples)."""
    out: dict[str, str] = {}
    for t in words["triples"]:
        for w in t["words"]:
            out.setdefault(w, t["pos"])
    return out
