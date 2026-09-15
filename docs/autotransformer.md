# Transformer autoencoder on raw depth profiles

Each sample uses the same preparation as the spectral depth-transformer
experiment (`prepare_sample`): repaired shots, per-channel L2 spectra, bulk-
relative log1p intensities, surface singles plus late depth bins, wavelength
binning within detector channels.

## Architecture

1. **Sinusoidal wavelength encoding** — fixed sin/cos PE over binned wavelength
   indices, intensity-weighted into each depth token.
2. **Glued CLS** — a learnable CLS token is prepended to the depth sequence.
3. **Transformer encoder** — 2 layers, 4 heads, `d_model=64` by default.
4. **Reconstruction head** — MSE on depth tokens only (CLS excluded).
5. **MLP classifier** — CLS state → 5 logits; labels are **5-dim one-hot**
   targets for soft cross-entropy.

Joint loss: `L = L_cls + λ L_recon` (`λ=1` by default).

## Run

```bash
uv sync --extra cnn
uv run --extra cnn python scripts/16_autotransformer.py benchmark --device cuda --tag initial
```

Smoke (not for reporting):

```bash
uv run --extra cnn python scripts/16_autotransformer.py benchmark --device cpu \
  --seeds 42 --folds 2 --repeats 1 --epochs 2 --tag smoke
```

Results land under `results/autotransformer/<tag>/` (`folds.csv`, `oof.csv`,
`summary.csv`, `recipe_autotransformer.json`). Tags must be new directory names.

## Predict

```bash
uv run --extra cnn python scripts/16_autotransformer.py predict \
  --recipe results/autotransformer/initial/recipe_autotransformer.json \
  --device cuda --output submissions/autotransformer_initial.csv
```

## Tests

```bash
uv run --extra cnn python -m unittest tests.test_autotransformer -v
uv run ruff check src/libs2026/autotransformer.py scripts/16_autotransformer.py tests/test_autotransformer.py
```
