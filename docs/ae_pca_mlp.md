# AE CLS + PCA + MLP fusion

Fuses the depth-profile autoencoder **CLS** embedding with classical
**PCA(`mean_lines`)** scores, then classifies with an **MLP**.

This is feature-level fusion (not a probability blend of `autotransformer` and
`pca_mlp`).

## Pipeline

1. Fit `AutoDepthClassifier` seed ensemble on raw depth sequences (one sequence
   per physical sample; same prep as `scripts/16_autotransformer.py`).
2. Average CLS vectors across seeds (`d_model`, default 64).
3. Build classical `mean_lines` features with **`n_groups=4`** (same augmentation
   as the strong `pca_mlp` recipe).
4. **Broadcast** each sample CLS onto that sample’s classical rows.
5. Fit `StandardScaler` + `PCA(30)` on classical rows; concat CLS ∥ PCA scores →
   `StandardScaler` → `MLPClassifier(256, 128)`.
6. At evaluation / predict time, average row probabilities back to one vote per
   sample.

Outer folds use `StratifiedGroupKFold` on sample IDs (all classical rows of a
sample stay together). PCA and MLP see only the outer training split; AE inner
validation stays inside each autoencoder fit.

## Run

```bash
uv sync --extra cnn
uv run --extra cnn python scripts/17_ae_pca_mlp.py benchmark --device cuda --tag g4_broadcast
```

Smoke (not for reporting):

```bash
uv run --extra cnn python scripts/17_ae_pca_mlp.py benchmark --device cpu \
  --seeds 42 --folds 2 --repeats 1 --epochs 2 --tag smoke_g4
```

Results land under `results/ae_pca_mlp/<tag>/` (`folds.csv`, `oof.csv`,
`summary.csv`, `recipe_ae_pca_mlp.json`). Tags must be new directory names.

## Predict

```bash
uv run --extra cnn python scripts/17_ae_pca_mlp.py predict \
  --recipe results/ae_pca_mlp/g4_broadcast/recipe_ae_pca_mlp.json \
  --device cuda --output submissions/ae_pca_mlp_g4.csv
```

## Tests

```bash
uv run --extra cnn python -m unittest tests.test_ae_pca_mlp -v
uv run ruff check src/libs2026/ae_pca_mlp.py scripts/17_ae_pca_mlp.py tests/test_ae_pca_mlp.py
```
