"""Background corpus: parquet in, packed (n_windows, window) token ids out. No downloads."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from token_drift.corpus import iter_parquet, pack, read_parquet, shuffle_within_windows, source_mix


def _write_parquet(path, texts, sets=None):
    cols = {"text": texts}
    if sets is not None:
        cols["meta"] = [{"pile_set_name": s} for s in sets]
    pq.write_table(pa.table(cols), path)


def test_pack_puts_one_eos_after_each_doc_and_drops_the_tail():
    w = pack([[1, 2, 3], [4], [5, 6]], eos_id=0, window=4)
    # stream is 1 2 3 0 4 0 5 6 0: two full windows, the lone trailing 0 is dropped
    assert w.tolist() == [[1, 2, 3, 0], [4, 0, 5, 6]]
    assert w.dtype == np.int32


def test_pack_respects_max_tokens():
    w = pack([[1] * 10 for _ in range(10)], eos_id=0, window=4, max_tokens=13)
    assert w.shape == (3, 4)  # 13-token budget -> 3 full windows


def test_pack_empty_docs_are_just_an_eos():
    assert pack([[], [7], []], eos_id=0, window=3).tolist() == [[0, 7, 0]]


def test_pack_too_short_gives_zero_windows():
    assert pack([[1, 2]], eos_id=0, window=8).shape == (0, 8)


def test_shuffle_keeps_each_windows_tokens_and_is_seeded():
    w = np.arange(40, dtype=np.int32).reshape(4, 10)
    a = shuffle_within_windows(w, seed=0)
    assert np.array_equal(a, shuffle_within_windows(w, seed=0))
    assert not np.array_equal(a, shuffle_within_windows(w, seed=1))
    assert np.array_equal(np.sort(a, axis=1), w)  # same bag of tokens per window
    assert not np.array_equal(a, w)
    # each row gets its own permutation, not one shared permutation
    assert not all(np.array_equal(np.argsort(a[i]), np.argsort(a[0])) for i in range(1, 4))


def test_read_parquet_texts_and_source_mix(tmp_path):
    p = tmp_path / "a.parquet"
    _write_parquet(p, ["hi", None, "yo"], ["Pile-CC", "Github", "Pile-CC"])
    texts, sets = read_parquet(p, "text")
    assert texts == ["hi", "", "yo"]  # a null text is an empty doc, not a crash
    assert source_mix(sets) == {"Pile-CC": 2, "Github": 1}


def test_read_parquet_without_meta_column(tmp_path):
    p = tmp_path / "a.parquet"
    _write_parquet(p, ["hi"])
    texts, sets = read_parquet(p, "text")
    assert texts == ["hi"] and sets is None
    assert source_mix(None) == {}


def test_iter_parquet_local_dir_is_sorted_and_local_file_is_itself(tmp_path):
    _write_parquet(tmp_path / "b.parquet", ["x"])
    _write_parquet(tmp_path / "a.parquet", ["y"])
    assert [p.name for p in iter_parquet(str(tmp_path))] == ["a.parquet", "b.parquet"]
    assert list(iter_parquet(str(tmp_path / "b.parquet"))) == [tmp_path / "b.parquet"]
