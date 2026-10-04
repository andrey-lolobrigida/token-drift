"""Probe (milestone B) stages end to end on a tiny fake shelf: the real Pythia tokenizer (HF
cache), a 2-layer random GPT-NeoX, two 'books' and a two-shard local 'Pile'. No network."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from token_drift import cli
from conftest import tiny_model

# The hard wrap before "courage" is on purpose: without unwrap() the book hits would be 0.
S = ("Courage is rare. The soldier showed\ncourage and not cowardice or rashness, so a brave man "
     "is neither cowardly nor rash. Business and community and experience and performance and "
     "information and government are famous, careful, active, different, important, possible.")
BOOK = ("Title page\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nPREFACE\n\n"
        + "\n\n".join([S] * 30) + "\n\nTHE END\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicense\n")
WORDS = """
sets:
  classical:
    fear:
      noun: [[cowardice, courage, rashness]]
      adj: [[cowardly, brave, rash]]
  everyday:
    fear:
      noun: [[timidity, courage, rashness]]
polysemy: [bank]
null_exclude: [government]
"""


@pytest.fixture
def probe_cfg(tmp_path, monkeypatch, pythia_tok):
    monkeypatch.setattr(cli, "_load_tokenizer", lambda name: pythia_tok)
    monkeypatch.setattr(cli.pb, "fetch_gutenberg", lambda book_id, cache_dir: BOOK)
    model = tiny_model(pythia_tok, window=64)
    monkeypatch.setattr(cli.ex, "build_model", lambda name, **kw: (model, pythia_tok))
    wf = tmp_path / "words.yaml"
    wf.write_text(WORDS)
    src = tmp_path / "pile"
    src.mkdir()
    doc = S.replace("\n", " ")
    pq.write_table(pa.table({"text": [doc + " " + doc] * 20 + [None]}), src / "part0.parquet")
    pq.write_table(pa.table({"text": [doc] * 5}), src / "part1.parquet")  # shard 1: not read
    c = {
        "run_name": "tiny_probe", "model": "from-cache", "random_init": False, "seed": 0, "device": "cpu",
        "probe": {
            "words": str(wf), "cache_dir": str(tmp_path / "cache"),
            "books": [{"name": "book_a", "gutenberg": 1},
                      {"name": "book_b", "gutenberg": 2, "start": "PREFACE", "end": "THE END"}],
            "pile": {"source": str(src), "text_field": "text", "shards": 1},
            "cap": 6, "min_context": 8, "max_context": 64,
        },
        "extract": {"mode": "probe", "batch_size": 4, "dtype_on_disk": "float16"},
        "metrics": {"min_count": 3, "null_pool": 4, "null_k": 2, "between_pct": 5},
        "viz": {"occ_frames": [0, 3]},
        "out_dir": str(tmp_path / "runs"),
    }
    p = tmp_path / "probe.yaml"
    p.write_text(yaml.safe_dump(c))
    return p


def _windows(pc):
    t, o = np.load(pc / "windows_tokens.npy"), np.load(pc / "windows_offsets.npy")
    return [t[o[i]:o[i + 1]] for i in range(len(o) - 1)]


def test_probe_corpus_writes_windows_meta_counts_and_report(probe_cfg, pythia_tok):
    rd = cli.stage_probe_corpus(cli.load_config(probe_cfg))
    pc = rd / "probe_corpus"
    counts = json.loads((pc / "counts.json").read_text())
    # unwrap() recovered the line-start " courage" in both books: 30 paragraphs each
    assert counts["courage"]["books"]["found"] == 60
    assert counts["courage"]["pile"]["found"] == 40  # shard 0 only: 20 docs x 2
    assert counts["courage"]["pile"]["capitalized"] > 0
    assert all(counts[w][g]["kept"] <= 6 for w in counts for g in ("books", "pile"))
    assert counts["bank"]["pile"]["found"] == 0
    meta = pq.read_table(pc / "occ_meta.parquet").to_pydict()
    wins = _windows(pc)
    assert len(wins) == len(meta["word"]) and len(wins) > 0
    ids = {w: tuple(pythia_tok(" " + w, add_special_tokens=False)["input_ids"]) for w in set(meta["word"])}
    for win, w, pos in zip(wins, meta["word"], meta["position"]):
        assert len(win) <= 64 and pos >= 8
        assert tuple(win[-len(ids[w]):]) == ids[w]  # every window ends on its word's last piece
    assert {s for s, g in zip(meta["source"], meta["group"]) if g == "pile"} == {"pile:0"}
    assert set(meta["source"]) >= {"book_a", "book_b"}
    tag = dict(zip(meta["word"], meta["role_tags"]))
    assert tag["courage"] == "classical/fear/noun/mean;everyday/fear/noun/mean"
    pool = json.loads((pc / "null_pool.json").read_text())
    assert len(pool["noun"]) == 4 and "government" not in [r["word"] for r in pool["noun"]]
    assert (pc / "count_report.md").read_text().startswith("# Probe count report")
    assert (pc / "probe_words.yaml").read_text() == WORDS
    books = json.loads((pc / "books.json").read_text())
    assert set(books) == {"book_a", "book_b"} and len(books["book_a"]["sha256"]) == 64


def test_probe_corpus_warns_when_a_cached_book_changed(probe_cfg, monkeypatch, capsys):
    cli.stage_probe_corpus(cli.load_config(probe_cfg))
    capsys.readouterr()
    monkeypatch.setattr(cli.pb, "fetch_gutenberg", lambda book_id, cache_dir: BOOK.replace("rare", "scarce"))
    cli.stage_probe_corpus(cli.load_config(probe_cfg))
    out = capsys.readouterr().out
    assert "WARNING" in out and "book_a" in out


def test_probe_corpus_too_few_null_candidates_fails_before_the_gpu(probe_cfg):
    c = cli.load_config(probe_cfg)
    c["metrics"]["null_k"] = 10
    with pytest.raises(ValueError, match="null-pool"):
        cli.stage_probe_corpus(c)


def test_probe_corpus_failed_rerun_keeps_the_previous_words_copy_and_books(probe_cfg):
    # a run that dies midway must not pair a new word list with the old run's occurrences
    cli.stage_probe_corpus(cli.load_config(probe_cfg))
    pc = cli.run_dir(cli.load_config(probe_cfg)) / "probe_corpus"
    books_before = (pc / "books.json").read_text()
    c = cli.load_config(probe_cfg)
    open(c["probe"]["words"], "a").write("# edited\n")
    c["metrics"]["null_k"] = 10
    with pytest.raises(ValueError, match="null-pool"):
        cli.stage_probe_corpus(c)
    assert (pc / "probe_words.yaml").read_text() == WORDS
    assert (pc / "books.json").read_text() == books_before


def test_probe_extract_writes_occ_per_window_and_no_acts(probe_cfg):
    c = cli.load_config(probe_cfg)
    cli.stage_probe_corpus(c)
    rd = cli.stage_extract(c)
    occ = np.load(rd / "extract" / "occ.npy")
    n = len(np.load(rd / "probe_corpus" / "windows_offsets.npy")) - 1
    assert occ.shape == (4, n, 16) and occ.dtype == np.float16  # tiny model: 2 layers -> 4 frames
    assert not (rd / "extract" / "acts.npy").exists()
    assert json.loads((rd / "extract" / "layer_names.json").read_text())[-1] == "unembed"


def test_normalize_refuses_a_probe_run(probe_cfg):
    with pytest.raises(ValueError, match="skip normalize"):
        cli.stage_normalize(cli.load_config(probe_cfg))


@pytest.fixture
def extracted(probe_cfg):
    c = cli.load_config(probe_cfg)
    cli.stage_probe_corpus(c)
    cli.stage_extract(c)
    return c, cli.run_dir(c)


def test_probe_metrics_writes_q16(extracted):
    c, rd = extracted
    cli.stage_metrics(c)
    q = json.loads((rd / "metrics" / "q16.json").read_text())
    assert q["groups"] == ["books", "pile"] and len(q["layer_names"]) == 4
    assert [t["id"] for t in q["triples"]] == [
        "classical/fear/noun/cowardice,courage,rashness", "classical/fear/adj/cowardly,brave,rash",
        "everyday/fear/noun/timidity,courage,rashness"]
    present = q["triples"][0]["groups"]["books"]
    assert len(present["null_pct"]) == 4 and len(present["swap_pct"][0]) == 3 and len(present["null_words"]) == 2
    assert q["triples"][2]["groups"]["pile"] == {"missing": ["timidity"]}
    assert q["summary"]["classical"]["books"][0]["n"] == 2 and q["summary"]["everyday"]["pile"][0]["n"] == 0
    assert {(m["id"].split("/")[0], m["group"]) for m in q["missing"]} == {("everyday", "books"), ("everyday", "pile")}
    assert q["self_sim"]["courage"]["books"][0] == pytest.approx(1.0, abs=1e-3)  # frame 0: embedding, same every time


def test_metrics_uses_the_word_list_saved_with_the_occurrences(extracted):
    # Review Focus 5: Andrey edits the yaml after extraction and reruns only metrics
    c, rd = extracted
    with open(c["probe"]["words"], "w") as f:
        f.write("sets: {}\n")
    cli.stage_metrics(c)
    assert len(json.loads((rd / "metrics" / "q16.json").read_text())["triples"]) == 3


def test_metrics_refuses_a_stale_extract(extracted):
    c, rd = extracted
    occ = np.load(rd / "extract" / "occ.npy")
    np.save(rd / "extract" / "occ.npy", occ[:, :-1])
    with pytest.raises(ValueError, match="re-run extract"):
        cli.stage_metrics(c)


def test_all_runs_the_whole_probe_pipeline(probe_cfg):
    cli.all(probe_cfg)
    rd = cli.run_dir(cli.load_config(probe_cfg))
    assert sorted(x.name for x in rd.iterdir()) == ["config.yaml", "extract", "metrics", "probe_corpus", "viz"]
    v = rd / "viz"
    for f in ("q16_summary.png", "q16_classical_fear_noun.png", "q16_classical_fear_adj.png",
              "q16_occ_classical_fear_noun.png"):
        assert (v / f).exists(), f
    assert not (v / "q16_everyday_fear_noun.png").exists()  # every row missing: nothing to draw


def test_viz_rejects_occ_frames_out_of_range(extracted):
    c, rd = extracted
    cli.stage_metrics(c)
    c["viz"]["occ_frames"] = [0, 9]
    with pytest.raises(ValueError, match="occ_frames"):
        cli.stage_viz(c)


def test_occ_report_lists_most_and_least_between_with_snippets(extracted):
    c, _ = extracted
    txt = cli.occ_report(c, ["cowardice", "courage", "rashness"], frame=2, group="pile", n=2)
    assert "most between" in txt and "least between" in txt
    assert txt.count("seg=") == 4 and "soldier" in txt and "[pile:0]" in txt


def test_occ_report_explains_bad_input(extracted):
    c, _ = extracted
    with pytest.raises(ValueError, match="timidity"):
        cli.occ_report(c, ["timidity", "courage", "rashness"], frame=0, group="books", n=2)
    with pytest.raises(ValueError, match="group"):
        cli.occ_report(c, ["cowardice", "courage", "rashness"], frame=0, group="web", n=2)
    with pytest.raises(ValueError, match="three"):
        cli.occ_report(c, ["courage"], frame=0, group="books", n=2)
    with pytest.raises(ValueError, match=r"0\.\.3"):
        cli.occ_report(c, ["cowardice", "courage", "rashness"], frame=4, group="books", n=2)
    with pytest.raises(ValueError, match=r"0\.\.3"):
        cli.occ_report(c, ["cowardice", "courage", "rashness"], frame=-1, group="books", n=2)


def test_probe_corpus_refuses_fewer_pile_shards_than_asked(probe_cfg):
    c = cli.load_config(probe_cfg)
    c["probe"]["pile"]["shards"] = 3  # the fixture has 2 files
    with pytest.raises(ValueError, match=r"2.*3"):
        cli.stage_probe_corpus(c)


def test_probe_corpus_reports_triples_sharing_a_last_piece(probe_cfg):
    rd = cli.stage_probe_corpus(cli.load_config(probe_cfg))
    pc = rd / "probe_corpus"
    assert json.loads((pc / "shared_last_piece.json").read_text()) == []  # fixture words all end differently
    assert "## Triples sharing a last token piece" in (pc / "count_report.md").read_text()


def test_probe_metrics_q16_is_strict_json_with_shared_pieces_and_null_counts(extracted):
    c, rd = extracted
    cli.stage_metrics(c)

    def no_nan(x):
        raise ValueError(f"{x} in q16.json")

    q = json.loads((rd / "metrics" / "q16.json").read_text(), parse_constant=no_nan)
    for t in q["triples"]:
        assert t["shared_last_piece"] == []
    e = q["triples"][0]["groups"]["books"]
    assert len(e["null_counts"]) == len(e["null_words"])


def test_metrics_refuses_an_extract_from_different_windows(extracted):
    # F3: same counts, one token changed -> the occurrence count check alone would pass
    c, rd = extracted
    pc = rd / "probe_corpus"
    wt = np.load(pc / "windows_tokens.npy")
    wt[0] = wt[0] + 1
    np.save(pc / "windows_tokens.npy", wt)
    with pytest.raises(ValueError, match="re-run extract"):
        cli.stage_metrics(c)


def test_metrics_refuses_an_extract_without_probe_source(extracted):
    c, rd = extracted
    (rd / "extract" / "probe_source.json").unlink()
    with pytest.raises(ValueError, match="re-run extract"):
        cli.stage_metrics(c)


def test_interrupted_probe_extract_leaves_no_occ(probe_cfg, monkeypatch):
    c = cli.load_config(probe_cfg)
    cli.stage_probe_corpus(c)

    def boom(*a, **kw):
        raise RuntimeError("killed mid-extract")

    monkeypatch.setattr(cli.ex, "extract_probe", boom)
    with pytest.raises(RuntimeError, match="killed"):
        cli.stage_extract(c)
    assert not (cli.run_dir(c) / "extract" / "occ.npy").exists()
