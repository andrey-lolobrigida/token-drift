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


# ---------- pass 1: counts and the null pool ----------

def count_single_token_words(ids, offsets, ok_next, vocab_size: int) -> np.ndarray:
    """Whole-word hits of every id at once (pass 1, for the null pool): a position counts if
    the next token doesn't continue the word, or it's the doc's last token."""
    ids = np.asarray(ids)
    offs = np.asarray(offsets, dtype=np.int64)
    whole = np.ones(len(ids), dtype=bool)
    whole[:-1] = ok_next[ids[1:]]
    last = offs[1:][offs[1:] > offs[:-1]] - 1  # skip empty docs
    whole[last] = True
    return np.bincount(ids[whole], minlength=vocab_size).astype(np.int64)


# Part of speech by suffix. Crude on purpose: the pool only has to be "abstract nouns" and
# "adjectives" in bulk; junk that slips through goes in the word file's null_exclude.
SUFFIXES = {
    "noun": ("ness", "ity", "ence", "ance", "ion", "ment", "ism", "ship", "dom"),
    "adj": ("ous", "ful", "ive", "ent", "ant", "less", "able", "ible", "al", "ic", "ish"),
}


def guess_pos(word: str) -> str | None:
    for pos in POS:  # nouns first: "-ment" also ends in the adjective suffix "-ent"
        for suf in SUFFIXES[pos]:
            # root of >= 2 chars: "table" is not "t" + "-able", "ion" is not "-ion"
            if word.endswith(suf) and len(word) - len(suf) >= 2:
                return pos
    return None


def null_candidates(tokens, counts_books, counts_pile, *, min_count: int, exclude: set[str]) -> dict:
    """Single-token lowercase words (" kindness") with >= min_count whole-word hits in both
    groups, not in the word file -> {pos: {word: book count}}."""
    out: dict[str, dict[str, int]] = {p: {} for p in POS}
    for i, s in enumerate(tokens):
        if not s.startswith(" "):
            continue
        w = s[1:]
        # short words are mostly function-ish or ambiguous; the pool wants plain abstract nouns/adjectives
        if len(w) < 5 or not (w.isascii() and w.isalpha() and w.islower()) or w in exclude:
            continue
        if counts_books[i] < min_count or counts_pile[i] < min_count:
            continue
        pos = guess_pos(w)
        if pos is not None:
            out[pos][w] = int(counts_books[i])
    return out


def pick_null_pool(cands: dict, targets: dict, *, n: int, seed: int) -> dict[str, list[str]]:
    """Per pos, n pool words whose book counts look like the probe words' book counts.

    Books are the binding group (most candidates are plentiful in the Pile): for each probe
    word, the unused candidate nearest in log book-count, then a seeded random fill up to n.
    More probe words than n is an error: the ascending walk would stop early and leave the
    most frequent probe words with no matched pool word.
    """
    rng = np.random.default_rng(seed)
    pool = {}
    for pos in POS:
        if len(targets.get(pos, [])) > n:
            raise ValueError(f"{pos}: {len(targets[pos])} probe words but null_pool n={n}: "
                             "raise metrics.null_pool above the probe words per pos")
        words = sorted(cands.get(pos, {}))
        logc = np.log(np.maximum([cands[pos][w] for w in words], 1)) if words else np.empty(0)
        used = np.zeros(len(words), dtype=bool)
        for t in sorted(targets.get(pos, [])):
            if used.sum() >= n or used.all():
                break
            dist = np.abs(logc - np.log(max(t, 1)))
            dist[used] = np.inf
            used[int(np.argmin(dist))] = True
        rest = np.flatnonzero(~used)
        used[rng.permutation(rest)[: max(0, n - int(used.sum()))]] = True
        pool[pos] = [w for w, u in zip(words, used) if u]
    return pool


def shared_last_piece(words: dict, wid: dict[str, tuple[int, ...]]) -> list[dict]:
    """Triples where two words end in the same token id -> [{"id", "pairs": [[w1, w2], ...]}].

    Vectors are read at the last piece, so at frame 0 (the embedding) those two words are the
    same point: "temperance" / "intemperance" both end in "ance". Worth knowing before
    reading any frame-0 verdict (run_q16 gives those frames none)."""
    out = []
    for t in words["triples"]:
        ws = t["words"]
        pairs = [[ws[i], ws[j]] for i in range(3) for j in range(i + 1, 3) if wid[ws[i]][-1] == wid[ws[j]][-1]]
        if pairs:
            out.append({"id": t["id"], "pairs": pairs})
    return out


# ---------- sampling and windows ----------

def reservoir(keys, cap: int, seed: int) -> np.ndarray:
    """At most `cap` items per key, uniform without replacement, seeded -> sorted indices.

    Bottom-k by random priority: the same distribution reservoir sampling gives, in one
    vectorized pass, since all hits fit in memory anyway.
    """
    keys = np.asarray(keys)
    if len(keys) == 0:
        return np.empty(0, dtype=np.int64)
    pri = np.random.default_rng(seed).random(len(keys))
    order = np.lexsort((pri, keys))  # by key, then priority
    sk = keys[order]
    first = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
    rank = np.arange(len(sk)) - np.repeat(first, np.diff(np.r_[first, len(sk)]))
    return np.sort(order[rank < cap]).astype(np.int64)


def cut_windows(ids, offsets, doc, end, *, max_context: int, eos_id: int) -> list[np.ndarray]:
    """One window per hit, ending at the hit's last piece: EOS + the whole doc prefix if that
    fits in max_context (a doc start always follows an EOS in Pythia's training data),
    else the last max_context tokens. Hits are already filtered for min_context."""
    offs = np.asarray(offsets, dtype=np.int64)
    eos = np.array([eos_id], dtype=np.uint16)
    out = []
    for d, e in zip(np.asarray(doc), np.asarray(end)):
        start = offs[d]
        if e - start + 2 <= max_context:
            out.append(np.concatenate([eos, ids[start : e + 1]]))
        else:
            out.append(np.asarray(ids[e + 1 - max_context : e + 1], dtype=np.uint16))
    return out


def snippets(tokenizer, windows, n_tokens: int = 48, n_chars: int = 150) -> list[str]:
    """Decoded tail of each window, for reading the sense of an occurrence later."""
    texts = tokenizer.batch_decode([w[-n_tokens:].tolist() for w in windows], skip_special_tokens=True)
    return [" ".join(t.split())[-n_chars:] for t in texts]


def role_tags(words: dict, pool: dict) -> dict[str, str]:
    """word -> "set/concept/pos/role;..." | "null:<pos>" | "polysemy", for occ_meta.parquet."""
    tags: dict[str, list[str]] = {}
    for t in words["triples"]:
        for role, w in zip(ROLES, t["words"]):
            tags.setdefault(w, []).append(f"{t['set']}/{t['concept']}/{t['pos']}/{role}")
    for w in words.get("polysemy", []):
        tags.setdefault(w, []).append("polysemy")
    for pos, ws in pool.items():
        for w in ws:
            tags.setdefault(w, []).append(f"null:{pos}")
    return {w: ";".join(dict.fromkeys(v)) for w, v in tags.items()}


# ---------- count report ----------

def count_report(words: dict, counts: dict, pool: dict, *, min_count: int) -> str:
    """Markdown for Andrey to review before the GPU pass: kept counts per triple and group
    (✗ = below min_count, so that triple is missing in that group), the null pool, and the
    capitalized / sentence-initial hits we skipped."""
    kept = lambda w, g: counts.get(w, {}).get(g, {}).get("kept", 0)  # noqa: E731
    lines = ["# Probe count report", "",
             f"Kept occurrences per word (after min_context and the cap). ✗ = below "
             f"min_count = {min_count}: the triple is reported missing in that group.", ""]
    for set_name in dict.fromkeys(t["set"] for t in words["triples"]):
        lines += [f"## {set_name}", "", "| concept | pos | deficiency / mean / excess | books | pile | usable in |",
                  "|---|---|---|---|---|---|"]
        for t in (t for t in words["triples"] if t["set"] == set_name):
            cells, usable = [], []
            for g in GROUPS:
                cells.append(" / ".join(f"{kept(w, g)}{'✗' if kept(w, g) < min_count else ''}" for w in t["words"]))
                if all(kept(w, g) >= min_count for w in t["words"]):
                    usable.append(g)
            lines.append(f"| {t['concept']} | {t['pos']} | {' / '.join(t['words'])} | {cells[0]} | {cells[1]} "
                         f"| {' + '.join(usable) or 'neither'} |")
        lines.append("")
    lines += ["## Null pool", "", "Whole-word hits found (before the cap).", "",
              "| pos | word | books | pile |", "|---|---|---|---|"]
    for pos in POS:
        for w in pool.get(pos, []):
            lines.append(f"| {pos} | {w} | {counts[w]['books']['found']} | {counts[w]['pile']['found']} |")
    lines += ["", "## Capitalized / sentence-initial hits skipped", "",
              "| word | books lowercase | books capitalized | pile lowercase | pile capitalized |",
              "|---|---|---|---|---|"]
    for w in dict.fromkeys(w for t in words["triples"] for w in t["words"]):
        c = counts.get(w)
        if c and (c["books"]["capitalized"] or c["pile"]["capitalized"]):
            lines.append(f"| {w} | {c['books']['found']} | {c['books']['capitalized']} "
                         f"| {c['pile']['found']} | {c['pile']['capitalized']} |")
    return "\n".join(lines) + "\n"
