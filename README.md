# token-drift

Push every token in `pythia-70m`'s vocabulary through the model and watch how the
vocab's cluster structure changes layer by layer. Numeric drift curves plus an
aligned-UMAP flipbook. Random-init control included so we can tell learned structure
from architectural structure.

Design: `docs/EXPERIMENT.md`. How we work: `CLAUDE.md`.

## Quickstart

```
uv sync
uv run token-drift all --config configs/pythia70m.yaml
uv run token-drift all --config configs/random_init.yaml
```

Outputs land in `runs/<run_name>/`.

## Results

_(fill in after the first full run: the two plots and three sentences on what we
saw, including anywhere the predictions in EXPERIMENT.md were wrong)_
