"""Milestone B probe helpers: pure functions, no network."""
import io
from pathlib import Path

import numpy as np
import pytest

from token_drift import probe as pb


def _write(tmp_path, text):
    p = tmp_path / "words.yaml"
    p.write_text(text)
    return p


WORDS = """
sets:
  classical:
    fear:
      noun: [[cowardice, courage, rashness], [fear, courage, fearlessness]]
      adj: [[cowardly, brave, rash]]
  everyday:
    fear:
      noun: [[cowardice, bravery, recklessness]]
polysemy: [bank, bat]
null_exclude: [business]
"""


def test_load_words_flattens_triples_in_file_order(tmp_path):
    w = pb.load_words(_write(tmp_path, WORDS))
    assert [t["words"] for t in w["triples"]] == [
        ("cowardice", "courage", "rashness"), ("fear", "courage", "fearlessness"),
        ("cowardly", "brave", "rash"), ("cowardice", "bravery", "recklessness"),
    ]
    t = w["triples"][2]
    assert (t["set"], t["concept"], t["pos"]) == ("classical", "fear", "adj")
    assert t["id"] == "classical/fear/adj/cowardly,brave,rash"
    assert w["polysemy"] == ["bank", "bat"] and w["null_exclude"] == ["business"]


def test_probe_word_pos_lists_each_word_once():
    w = {"triples": [
        {"words": ("cowardice", "courage", "rashness"), "pos": "noun"},
        {"words": ("fear", "courage", "fearlessness"), "pos": "noun"},
        {"words": ("cowardly", "brave", "rash"), "pos": "adj"},
    ]}
    assert pb.probe_word_pos(w) == {
        "cowardice": "noun", "courage": "noun", "rashness": "noun", "fear": "noun",
        "fearlessness": "noun", "cowardly": "adj", "brave": "adj", "rash": "adj",
    }


@pytest.mark.parametrize("bad, match", [
    ("sets: {s: {c: {verb: [[a, b, c]]}}}", "part of speech"),
    ("sets: {s: {c: {noun: [[a, b]]}}}", "deficiency/mean/excess"),
    ("sets: {s: {c: {noun: [[' a', b, c]]}}}", "bare lowercase"),
    ("sets: {s: {c: {noun: [[Courage, b, c]]}}}", "bare lowercase"),
])
def test_load_words_rejects_malformed_files(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        pb.load_words(_write(tmp_path, bad))


def test_the_real_word_file_loads():
    w = pb.load_words(Path(__file__).parents[1] / "configs" / "probe_words.yaml")
    assert {t["set"] for t in w["triples"]} == {"classical", "everyday"}
    assert len(w["triples"]) >= 40 and "bank" in w["polysemy"]


# ---------- Gutenberg ----------

BOOK = ("The Project Gutenberg eBook\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n"
        "intro\r\nQUESTION 49\r\nhabit is\r\na quality\r\n\r\nnext para\r\nQUESTION 90\r\nlaw\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\nlicense\r\n")


def test_gutenberg_body_cuts_header_and_footer():
    assert pb.gutenberg_body(BOOK) == "intro\nQUESTION 49\nhabit is\na quality\n\nnext para\nQUESTION 90\nlaw"


def test_gutenberg_body_markers_start_inclusive_end_exclusive():
    assert pb.gutenberg_body(BOOK, "QUESTION 49", "QUESTION 90") == "QUESTION 49\nhabit is\na quality\n\nnext para"


def test_gutenberg_body_missing_marker_says_which():
    with pytest.raises(ValueError, match="QUESTION 12"):
        pb.gutenberg_body(BOOK, "QUESTION 12")
    with pytest.raises(ValueError, match="START OF"):
        pb.gutenberg_body("no markers here")


def test_unwrap_joins_hard_wrapped_lines_keeps_paragraphs():
    # Gutenberg wraps at ~70 chars; a line-start word would tokenize without its space
    assert pb.unwrap("the soldier showed\ncourage here\n\n  next\npara  ") == "the soldier showed courage here\n\nnext para"


def test_fetch_gutenberg_downloads_once_then_uses_the_cache(tmp_path, monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO("﻿hello".encode("utf-8"))  # Gutenberg files often start with a BOM

    monkeypatch.setattr(pb.urllib.request, "urlopen", fake_urlopen)
    assert pb.fetch_gutenberg(8438, tmp_path) == "hello"
    assert calls == ["https://www.gutenberg.org/cache/epub/8438/pg8438.txt"]

    def offline(url, timeout):
        raise OSError("no network")

    monkeypatch.setattr(pb.urllib.request, "urlopen", offline)
    assert pb.fetch_gutenberg(8438, tmp_path) == "hello"  # cached copy, no network needed


# ---------- token matching ----------

def test_concat_docs_and_uint16_overflow():
    ids, offs = pb.concat_docs([[1, 2], [], [3]])
    assert ids.dtype == np.uint16 and ids.tolist() == [1, 2, 3] and offs.tolist() == [0, 2, 2, 3]
    with pytest.raises(ValueError, match="uint16"):
        pb.to_uint16([70000])


def test_word_token_ids_space_prefix_and_eos_check(pythia_tok):
    ids = pb.word_token_ids(pythia_tok, ["courage", "temperance"], eos_id=pythia_tok.eos_token_id)
    assert pythia_tok.convert_ids_to_tokens(list(ids["courage"])) == ["Ġcourage"]
    assert len(ids["temperance"]) == 2  # " temper" + "ance": multi-token, read at the last piece
    bare = pb.word_token_ids(pythia_tok, ["Courage"], eos_id=pythia_tok.eos_token_id, prefix="")
    assert bare["Courage"] != ids["courage"]
    with pytest.raises(ValueError, match="EOS"):
        pb.word_token_ids(pythia_tok, ["<|endoftext|>"], eos_id=pythia_tok.eos_token_id, prefix="")


def test_boundary_table(pythia_tok):
    ok = pb.boundary_table(pythia_tok)
    tid = lambda s: pythia_tok.convert_tokens_to_ids(s)  # noqa: E731
    assert ok[tid(",")] and ok[tid("Ġthe")] and ok[pythia_tok.eos_token_id]
    assert not ok[tid("ous")] and not ok[tid("-")] and not ok[tid("'s")] and not ok[tid("7")]
    # Token starting with curly apostrophe (U+2019) should also block
    curly_apos_token_id = 434  # "'s" with curly apostrophe
    assert pythia_tok.decode([curly_apos_token_id]).startswith("'"), "token 434 should start with curly apostrophe"
    assert not ok[curly_apos_token_id]


def _docs(tok, *texts):
    return pb.concat_docs([tok(t, add_special_tokens=False)["input_ids"] for t in texts])


def test_find_hits_whole_lowercase_words_only(pythia_tok):
    ids, offs = _docs(pythia_tok, "He showed courage, not courageous zeal; Courage too. self-mastery and self.")
    ok = pb.boundary_table(pythia_tok)
    w = pb.word_token_ids(pythia_tok, ["courage", "self"], eos_id=0)
    pat, doc, end = pb.find_hits(ids, offs, [w["courage"], w["self"]], ok)
    # courageous, Courage and self-mastery are all skipped
    assert pat.tolist() == [0, 1] and doc.tolist() == [0, 0]
    assert ids[end[0]] == w["courage"][-1]


def test_find_hits_multi_token_word_ends_at_its_last_piece(pythia_tok):
    ids, offs = _docs(pythia_tok, "true temperance is rare, and a temper is not")
    ok = pb.boundary_table(pythia_tok)
    w = pb.word_token_ids(pythia_tok, ["temperance", "temper"], eos_id=0)
    pat, _, end = pb.find_hits(ids, offs, [w["temperance"], w["temper"]], ok)
    # " temper" inside "temperance" is followed by "ance" (a letter), so it isn't a hit
    assert pat.tolist() == [0, 1]
    assert tuple(ids[end[0] - 1 : end[0] + 1]) == w["temperance"]


def test_find_hits_never_spans_two_documents():
    ids, offs = pb.concat_docs([[5, 7], [8, 9]])
    pat, doc, end = pb.find_hits(ids, offs, [(7, 8)], np.ones(10, bool))
    assert len(end) == 0


def test_find_hits_word_at_the_end_of_a_doc_counts():
    # Review Focus 3: no next token to check -> whole word
    ids, offs = pb.concat_docs([[1, 2, 3], [4, 3, 5]])
    ok = np.zeros(10, bool)  # every next token "continues the word"
    pat, doc, end = pb.find_hits(ids, offs, [(3,)], ok)
    assert end.tolist() == [2] and doc.tolist() == [0]


def test_find_hits_empty_doc_in_the_middle_keeps_doc_indices():
    # Review Focus 2: the Pile has empty texts
    ids, offs = pb.concat_docs([[1, 3], [], [3, 1]])
    _, doc, end = pb.find_hits(ids, offs, [(3,)], np.ones(10, bool))
    assert doc.tolist() == [0, 2] and end.tolist() == [1, 2]


# ---------- pass 1 counts + null pool ----------

def test_count_single_token_words():
    ids, offs = pb.concat_docs([[3, 4, 3], [3, 5]])
    ok = np.array([True, True, True, True, False, True])  # id 4 continues a word
    # doc 0: 3 then 4 -> not whole; 4 then 3 -> whole; 3 at doc end -> whole. doc 1: 3 then 5, 5 at end.
    assert pb.count_single_token_words(ids, offs, ok, vocab_size=6).tolist() == [0, 0, 0, 2, 1, 1]


def test_guess_pos_by_suffix():
    assert pb.guess_pos("kindness") == "noun" and pb.guess_pos("government") == "noun"
    assert pb.guess_pos("famous") == "adj" and pb.guess_pos("possible") == "adj"
    assert pb.guess_pos("table") is None


def test_null_candidates_need_both_groups_and_skip_excluded():
    tokens = [" kindness", " famous", " table", " Goodness", "darkness", " sadness", " courage",
              " ion", " vanity"]
    books = np.array([30, 25, 40, 50, 50, 5, 90, 90, 40])
    pile = np.array([30, 25, 40, 50, 50, 99, 90, 90, 40])
    c = pb.null_candidates(tokens, books, pile, min_count=20, exclude={"vanity"})
    # table: no suffix; Goodness: capital; darkness: no space; sadness: 5 book hits; ion: too short
    assert c == {"noun": {"kindness": 30}, "adj": {"famous": 25}}


def test_pick_null_pool_matches_book_counts_then_fills_seeded():
    nouns = {f"w{i}ness": c for i, c in enumerate([10, 20, 40, 80, 160, 320, 640])}
    cands = {"noun": nouns, "adj": {"aous": 50, "bous": 60}}
    pool = pb.pick_null_pool(cands, {"noun": [39, 700], "adj": []}, n=4, seed=0)
    assert len(pool["noun"]) == 4 and {"w2ness", "w6ness"} <= set(pool["noun"])  # nearest to 39 and 700
    assert pool["adj"] == ["aous", "bous"]  # fewer candidates than n: take them all
    assert pool == pb.pick_null_pool(cands, {"noun": [39, 700], "adj": []}, n=4, seed=0)



def test_pick_null_pool_refuses_more_targets_than_pool_words():
    # F1: walking targets ascending and stopping at n silently left the most frequent words unmatched
    cands = {"noun": {f"w{i}ness": 10 * (i + 1) for i in range(10)}, "adj": {}}
    with pytest.raises(ValueError, match=r"noun.*3.*2"):
        pb.pick_null_pool(cands, {"noun": [10, 20, 30], "adj": []}, n=2, seed=0)


def test_shared_last_piece_on_the_real_tokenizer(pythia_tok):
    words = {"triples": [
        {"id": "a", "words": ("insensibility", "temperance", "intemperance")},
        {"id": "b", "words": ("cowardice", "courage", "rashness")},
    ]}
    flat = [w for t in words["triples"] for w in t["words"]]
    wid = pb.word_token_ids(pythia_tok, flat, eos_id=pythia_tok.eos_token_id)
    out = pb.shared_last_piece(words, wid)
    assert [r["id"] for r in out] == ["a"]
    assert ["temperance", "intemperance"] in out[0]["pairs"]

# ---------- sampling + windows ----------

def test_reservoir_caps_each_key_seeded_and_uniform():
    keys = np.repeat([0, 1, 2], [5, 3000, 10000])
    kept = pb.reservoir(keys, cap=1000, seed=0)
    assert np.bincount(keys[kept]).tolist() == [5, 1000, 1000]
    assert np.array_equal(kept, pb.reservoir(keys, 1000, 0))
    assert not np.array_equal(kept, pb.reservoir(keys, 1000, 1))
    pos = kept[keys[kept] == 2] - 3005  # where in key 2's stream the picks came from
    hist = np.bincount(pos // 1000, minlength=10)  # expect ~100 per tenth
    assert hist.min() > 60 and hist.max() < 140


def test_cut_windows_caps_context_and_adds_eos_only_at_doc_start():
    ids, offs = pb.concat_docs([np.arange(1, 11)])  # one doc, tokens 1..10 at positions 0..9
    w = pb.cut_windows(ids, offs, np.array([0, 0, 0]), np.array([3, 4, 9]), max_context=6, eos_id=0)
    assert w[0].tolist() == [0, 1, 2, 3, 4]        # whole prefix + EOS fits in 6
    assert w[1].tolist() == [0, 1, 2, 3, 4, 5]     # exactly 6 with the EOS
    assert w[2].tolist() == [5, 6, 7, 8, 9, 10]    # capped at 6, starts mid-doc: no EOS
    assert all(x.dtype == np.uint16 for x in w)


def test_snippets_decode_the_tail(pythia_tok):
    win = np.array(pythia_tok(" the soldier showed courage", add_special_tokens=False)["input_ids"], np.uint16)
    s = pb.snippets(pythia_tok, [np.r_[np.uint16(0), win]], n_chars=20)
    assert s == ["soldier showed courage"[-20:]]


def test_role_tags():
    words = {"triples": [{"set": "classical", "concept": "fear", "pos": "noun",
                          "words": ("cowardice", "courage", "rashness")}],
             "polysemy": ["bank", "courage"]}
    tags = pb.role_tags(words, {"noun": ["kindness"], "adj": []})
    assert tags["courage"] == "classical/fear/noun/mean;polysemy"
    assert tags["rashness"] == "classical/fear/noun/excess"
    assert tags["kindness"] == "null:noun" and tags["bank"] == "polysemy"


def test_count_report_marks_short_words_and_lists_the_pool():
    words = {"triples": [
        {"set": "classical", "concept": "fear", "pos": "noun", "words": ("cowardice", "courage", "rashness")},
    ], "polysemy": []}
    c = lambda b, p, cap=0: {"books": {"found": b, "context_ok": b, "kept": b, "capitalized": cap},  # noqa: E731
                             "pile": {"found": p, "context_ok": p, "kept": p, "capitalized": 0}}
    counts = {"cowardice": c(44, 72), "courage": c(118, 944, cap=66), "rashness": c(32, 3),
              "kindness": c(30, 500)}
    md = pb.count_report(words, counts, {"noun": ["kindness"], "adj": []}, min_count=20)
    assert "| fear | noun | cowardice / courage / rashness | 44 / 118 / 32 | 72 / 944 / 3✗ | books |" in md
    assert "| noun | kindness | 30 | 500 |" in md
    assert "| courage | 118 | 66 | 944 | 0 |" in md  # capitalized hits we skipped, per group
