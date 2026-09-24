"""Per-layer normalization. Never skip this stage; see docs/EXPERIMENT.md.

Residual-stream vectors live in a narrow cone (Ethayarajh 2019) and the cone gets
narrower with depth. Uncentered cosine similarity then mostly measures "how far
along the shared direction are you", which is not the geometry we care about.
"""
from __future__ import annotations

import numpy as np


def normalize_layer(
    x: np.ndarray, *, center: bool, unit_norm: bool, drop_top_pcs: int,
    row_norm_first: bool = False,
) -> np.ndarray:
    """(vocab, d) -> (vocab, d) float32. Center, optionally drop top PCs, unit-norm rows.

    Order matters: PCs are computed on centered data, and unit-norming comes last so the
    projection can't un-normalize the rows.

    row_norm_first: unit-norm each row *before* centering too. When one shared direction
    shows up in very different amounts per token (Pythia's pre-LN L6), centering first
    turns that spread into direction differences that swamp everything else. Scaling
    rows first is roughly what the model's own LayerNorm does. See FINDINGS section 8.
    """
    out = np.asarray(x, dtype=np.float32).copy()  # upcast: float16 is for disk only
    if row_norm_first:
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    if center or drop_top_pcs > 0:
        # all-but-the-top (Mu & Viswanath 2018) is defined on centered data, so dropping
        # PCs implies centering even if the flag says otherwise.
        out -= out.mean(axis=0, keepdims=True)
    if drop_top_pcs > 0:
        # SVD of the centered matrix; right singular vectors are the PCs.
        # full_matrices=False keeps this cheap for (50k, 512).
        _, _, vt = np.linalg.svd(out, full_matrices=False)
        top = vt[:drop_top_pcs]  # (k, d)
        out -= (out @ top.T) @ top
    if unit_norm:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out /= np.maximum(norms, 1e-8)  # zero rows stay zero instead of becoming NaN
    return out


def normalize_all(
    acts: np.ndarray, *, center: bool, unit_norm: bool, drop_top_pcs: int,
    row_norm_first: bool = False,
) -> np.ndarray:
    """(layers, vocab, d) -> same shape, float16. Each layer normalized independently."""
    out = np.empty(acts.shape, dtype=np.float16)
    for i in range(acts.shape[0]):
        out[i] = normalize_layer(
            acts[i], center=center, unit_norm=unit_norm, drop_top_pcs=drop_top_pcs,
            row_norm_first=row_norm_first,
        ).astype(np.float16)
    return out
