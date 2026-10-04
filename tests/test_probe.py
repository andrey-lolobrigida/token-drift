"""Milestone B probe helpers: pure functions, no network."""
from pathlib import Path

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
