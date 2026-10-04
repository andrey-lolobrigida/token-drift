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
