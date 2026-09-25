"""Background corpus for v1: fetch text, pack it into model-sized windows, maybe shuffle.

v1 swaps v0's "[BOS, tok] alone" for "the token as the model usually sees it", so we
need real text in the shape Pythia trained on: docs glued with <|endoftext|> and cut
into 2048-token windows. See docs/superpowers/specs/2026-09-24-v1-milestone-a-design.md.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def iter_parquet(source: str) -> Iterator[Path]:
    """Parquet files of a dataset in a stable order, fetched one at a time.

    `source` is a local file or dir (tests, offline) or an HF dataset repo id. Lazy on
    purpose: with a `max_tokens` budget on a big sharded source (the_pile_deduplicated)
    we stop downloading as soon as the budget is met.
    """
    local = Path(source)
    if local.is_file():
        yield local
        return
    if local.is_dir():
        yield from sorted(local.glob("*.parquet"))
        return
    from huggingface_hub import hf_hub_download, list_repo_files

    names = sorted(f for f in list_repo_files(source, repo_type="dataset") if f.endswith(".parquet"))
    if not names:
        raise ValueError(f"no parquet files in HF dataset {source!r}")
    for name in names:
        yield Path(hf_hub_download(source, name, repo_type="dataset"))


def read_parquet(path, text_field: str) -> tuple[list[str], list[str | None] | None]:
    """(texts, pile_set_name per doc). Set names are None if there's no `meta` column."""
    table = pq.read_table(path)
    texts = [t or "" for t in table.column(text_field).to_pylist()]  # null text = empty doc
    sets = None
    if "meta" in table.column_names:
        sets = [m.get("pile_set_name") if isinstance(m, dict) else None
                for m in table.column("meta").to_pylist()]
    return texts, sets


def source_mix(set_names: list[str | None] | None) -> dict[str, int]:
    """How many docs came from each Pile subset, biggest first. Just for meta.json."""
    if set_names is None:
        return {}
    return dict(Counter(s for s in set_names if s is not None).most_common())


def pack(
    docs: Iterable[Sequence[int]], *, eos_id: int, window: int, max_tokens: int | None = None
) -> np.ndarray:
    """Token-id docs -> (n_windows, window) int32.

    One EOS after every doc (that's how Pythia's training data was packed, so windows
    that span two docs are in-distribution). The partial tail is dropped: it's less than
    one window of data and would be the only window with a different length.
    """
    parts, total = [], 0
    for d in docs:
        a = np.append(np.asarray(d, dtype=np.int32), np.int32(eos_id))
        parts.append(a)
        total += len(a)
        if max_tokens is not None and total >= max_tokens:
            break
    stream = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
    if max_tokens is not None:
        stream = stream[:max_tokens]
    n = len(stream) // window
    return stream[: n * window].reshape(n, window)


def shuffle_within_windows(windows: np.ndarray, seed: int) -> np.ndarray:
    """Independent seeded permutation of each row.

    Keeps each window's bag of tokens (so its topic) and destroys word order: the
    "is the model reading, or just soaking up co-occurrence?" control.
    """
    rng = np.random.default_rng(seed)
    # argsort of iid uniforms = a uniform random permutation, one per row, vectorized
    perm = np.argsort(rng.random(windows.shape), axis=1)
    return np.take_along_axis(windows, perm, axis=1)
