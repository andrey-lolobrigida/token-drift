"""FINDINGS 11.x / Q13: why the final LN barely moves v1 (unit_mean) neighbourhoods.

One-off check, not part of the pipeline. Needs runs/pythia70m and runs/pythia70m_corpus
(extract + metrics). `uv run python scripts/q13_ln_step.py`, ~3 min on CPU.
"""
import json, numpy as np
from token_drift.normalize import normalize_layer
from token_drift.metrics import knn_indices, knn_overlap, top_pc_share
R='runs/'
idx=np.array(json.load(open(R+'pythia70m_corpus/metrics/metrics.json'))['subsample_idx'])
seen=np.load(R+'pythia70m_corpus/extract/counts.npy')>0
srcs={'v0 (token alone)':(R+'pythia70m/extract/acts.npy',None),
      'v1 unit_mean':(R+'pythia70m_corpus/extract/acts.npy',seen),
      'v1 raw_mean':(R+'pythia70m_corpus/extract/acts_rawmean.npy',seen)}
F={'L5':5,'L6pre':6,'L6post':7}
for name,(p,fit) in srcs.items():
    a=np.load(p,mmap_mode='r'); print(name)
    raw={f:np.asarray(a[i],dtype=np.float32) for f,i in F.items()}
    for f,x in raw.items():
        n=np.linalg.norm(x[idx],axis=1)
        print(f'  {f:7} row-norm spread (std/mean) {n.std()/n.mean():.2f}   top-PC share {top_pc_share(x[idx]):.2f}')
    for rnf in (False,True):
        kn={f:knn_indices(normalize_layer(x,center=True,unit_norm=True,drop_top_pcs=0,row_norm_first=rnf,fit_rows=fit)[idx],10) for f,x in raw.items()}
        print(f'  {"unit-norm first" if rnf else "center first   "}: L5>L6pre {knn_overlap(kn["L5"],kn["L6pre"]):.2f}  L6pre>post {knn_overlap(kn["L6pre"],kn["L6post"]):.2f}  L5>L6post {knn_overlap(kn["L5"],kn["L6post"]):.2f}')
