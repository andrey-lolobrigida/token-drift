"""Milestone B: find a word's occurrences in targeted texts and cut a window ending at each.

Pythia is causal, so a hit's residual only depends on the tokens before it: find the hits
first, then run the model on just those windows instead of on billions of tokens.
Spec: docs/superpowers/specs/2026-10-03-v1-milestone-b-design.md
"""
from __future__ import annotations

import re
import urllib.request
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


# ---------- books ----------

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def fetch_gutenberg(book_id: int, cache_dir) -> str:
    """Plain text of a Gutenberg book, downloaded once into cache_dir (outside the repo)."""
    path = Path(cache_dir).expanduser() / f"pg{book_id}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(GUTENBERG_URL.format(id=book_id), timeout=60) as r:
            data = r.read()
        tmp = path.with_suffix(".part")  # a killed download never looks like a cached book
        tmp.write_bytes(data)
        tmp.replace(path)
    return path.read_text(encoding="utf-8-sig")  # -sig: drop the BOM many Gutenberg files have


def _first_line(lines: list[str], pred, frm: int, what: str) -> int:
    for i in range(frm, len(lines)):
        if pred(lines[i]):
            return i
    raise ValueError(f"marker {what!r} not found")


def gutenberg_body(text: str, start: str | None = None, end: str | None = None) -> str:
    """Text between the *** START / *** END lines, then from the first line equal to `start`
    (kept) up to the first line equal to `end` after it (cut). Raises naming a missing marker."""
    lines = text.replace("\r\n", "\n").split("\n")
    a = _first_line(lines, lambda s: s.startswith("*** START OF"), 0, "*** START OF") + 1
    b = _first_line(lines, lambda s: s.startswith("*** END OF"), a, "*** END OF")
    body = lines[a:b]
    if start is not None:
        body = body[_first_line(body, lambda s: s.strip() == start, 0, start):]
    if end is not None:
        body = body[:_first_line(body, lambda s: s.strip() == end, 1, end)]
    return "\n".join(body).strip("\n")


def unwrap(text: str) -> str:
    """Hard-wrapped lines -> paragraphs: single newlines become spaces, blank lines stay.

    Gutenberg wraps at ~70 chars, so ~8% of words start a line (Summa I-II, measured), and a
    line-start word tokenizes without its space ("courage", not " courage"): we'd miss it.
    """
    paras = re.split(r"\n\s*\n", text.strip())
    return "\n\n".join(re.sub(r"\s*\n\s*", " ", p).strip() for p in paras)


# ---------- token matching ----------

def to_uint16(ids) -> np.ndarray:
    """Token ids as uint16 (half the RAM of int32 for 360M Pile tokens). Pythia/GPT-2 fit."""
    a = np.asarray(ids, dtype=np.int64)
    if a.size and a.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"token id {a.max()} doesn't fit in uint16; switch the dtype for this tokenizer")
    return a.astype(np.uint16)


def concat_docs(docs) -> tuple[np.ndarray, np.ndarray]:
    """List of token-id docs -> (flat uint16 ids, int64 offsets with offsets[i]:offsets[i+1] = doc i)."""
    parts = [to_uint16(d) for d in docs]
    offsets = np.zeros(len(parts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(p) for p in parts])
    ids = np.concatenate(parts) if parts else np.empty(0, np.uint16)
    return ids, offsets


def word_token_ids(tokenizer, words, *, eos_id: int, prefix: str = " ") -> dict[str, tuple[int, ...]]:
    """Token ids of prefix + word for each word (prefix "" for the capitalized-skip count)."""
    out = {}
    for w in words:
        ids = tuple(tokenizer(prefix + w, add_special_tokens=False)["input_ids"])
        if not ids or eos_id in ids:
            raise ValueError(f"{prefix + w!r} tokenizes to {ids}: empty or contains EOS")
        out[w] = ids
    return out


def boundary_table(tokenizer) -> np.ndarray:
    """ok[i] = a hit followed by token i is a whole word: token i doesn't start with a letter,
    digit, hyphen or apostrophe. Else " courage" matches inside "courageous" and " self"
    inside "self-mastery". Decided once per id on the decoded string."""
    ok = np.ones(len(tokenizer), dtype=bool)
    for i in range(len(tokenizer)):
        s = tokenizer.decode([i])
        if s and (s[0].isalnum() or s[0] in "-'’"):  # hyphen, straight quote, curly quote U+2019
            ok[i] = False
    return ok


def find_hits(ids, offsets, patterns, ok_next) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every whole-word occurrence of each pattern (a tuple of ids), inside one document.

    -> (pattern index, doc index, global index of the pattern's last piece). One vectorized
    scan per pattern; a hit on a doc's last token counts (nothing follows it).
    """
    ids = np.asarray(ids)
    offs = np.asarray(offsets, dtype=np.int64)
    n = len(ids)
    pats, docs, ends = [], [], []
    for p, pat in enumerate(patterns):
        k = len(pat)
        end = np.flatnonzero(ids[k - 1:] == pat[-1]) + (k - 1)
        for j in range(1, k):
            end = end[ids[end - j] == pat[-1 - j]]
        doc = np.searchsorted(offs, end, side="right") - 1  # empty docs never contain `end`
        inside = end - (k - 1) >= offs[doc]
        nxt = end + 1
        whole = (nxt >= offs[doc + 1]) | ok_next[ids[np.minimum(nxt, n - 1)]]
        keep = inside & whole
        pats.append(np.full(int(keep.sum()), p, dtype=np.int64))
        docs.append(doc[keep])
        ends.append(end[keep])
    cat = lambda xs: np.concatenate(xs) if xs else np.empty(0, np.int64)  # noqa: E731
    return cat(pats), cat(docs).astype(np.int64), cat(ends).astype(np.int64)
