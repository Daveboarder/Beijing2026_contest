# Spectral CNN and depth transformer experiment

Each sample becomes 65 depth tokens: shots 1–20 remain separate, and shots
21–200 are averaged in groups of four. Each token contains shot-normalized
spectra and three log1p L2-intensity ratios relative to the sample's trailing
bulk region. Outliers are repaired before normalization. Baseline removal and
smoothing are disabled for this experiment. Wavelengths are averaged in groups
of four separately within each detector channel (three widths of 1023).

Three small CNN branches (16 then 32 filters) operate within detector channels;
their weights are shared across all depths. Eight pooled wavelength regions per
channel retain spectral location. A projection produces 64-dimensional tokens,
augmented with shot-center and window-width coordinates and intensity features.
One bidirectional transformer block uses four attention heads and a 128-unit
feed-forward layer. Duration-weighted pooling of early (1–20), intermediate
(21–140), and bulk (141–200) tokens feeds a 32-unit MLP and five class logits.
No circular shifts, positional shuffling, or causal mask are used.

The default five-class transformer model has 97,989 trainable parameters.

## Run

Install the existing `cnn` dependency extra and prepare the raw-shot cache with
`scripts/01_prepare_data.py`. The data path must be configured on the training
host; spectra are not distributed in this repository.

```bash
uv sync --extra cnn
uv run --extra cnn python scripts/15_depth_transformer.py benchmark --device cuda --tag initial
```

The default comparison runs pooling-only, depth-CNN, and transformer encoders
with the same spectral encoder, three initialization seeds, and the same
sample-grouped outer folds. CV fold count and repeat count come from the config
(normally 5 x 4). Initialization seeds never change the fold assignments.
This is a substantial training run: each fold performs epoch selection and refit.

For a smoke run, explicitly reduce the training budget (not suitable for reporting
performance):

```bash
uv run --extra cnn python scripts/15_depth_transformer.py benchmark --device cpu --encoders transformer --seeds 42 --folds 2 --repeats 1 --epochs 2 --tag smoke
```

Use `--no-intensity` and a new tag for the intensity-feature ablation. Tags must
be new directory names; existing runs are never overwritten. Results live under
`results/depth_transformer/<tag>/`: `folds.csv`, per-seed and ensemble `oof.csv`,
`summary.csv`, and a recipe for each encoder. Accuracy is the mean of per-repeat
ensemble accuracies; its standard deviation is across repeats, not seeds.
Repeated CV does not create additional independent physical samples.

## Validation and refitting

The inner split contains 20% of each outer training set, stratified by sample
label. Its spectra do not affect the selection-stage scalers, class weights,
or gradients. Unweighted validation cross-entropy chooses the epoch count
(maximum 100, patience 15). A newly initialized model then refits all outer
training samples for that count, using scalers fitted on those samples.
Training uses inverse-frequency weighted cross-entropy, AdamW, gradient clipping,
and dropout. All components, including the spectral encoder, learn end to end.

Prediction requires a benchmark recipe. For each seed, the median selected
epoch count across outer folds/repeats is used to train on all labeled samples.
This epoch-aggregation rule is a final-refit heuristic. The outer CV estimates
the epoch-selection procedure, not the exact final full-data fit.

```bash
uv run --extra cnn python scripts/15_depth_transformer.py predict --recipe results/depth_transformer/initial/recipe_transformer.json --device cuda --output submissions/depth_transformer_initial.csv
```

The recipe preserves preprocessing and model settings and records hashes of the
prepared training data and model source. Prediction rejects changed training
data or model source. `predict --config ...` can override paths for another host
but cannot silently change the evaluated preprocessing. Outputs include the
template-ordered submission and a joblib checkpoint with models, recipe, sample
IDs, and probabilities. Only load trusted joblib checkpoints.

Compare against the established classical baseline before selecting a model.
Choosing the best encoder on these same OOF scores adds selection optimism;
small gains need paired error analysis and independent confirmation. This code
does not choose blend weights or modify the existing classical pipelines.

## Tests

```bash
uv run --extra cnn python -m unittest discover -s tests -v
uv run ruff check src/libs2026/depth_transformer.py scripts/15_depth_transformer.py tests/test_depth_transformer.py
```

The tests need `src` on `PYTHONPATH` if the package is not installed. They use
synthetic data for shape/gradient, preprocessing, validation isolation,
reproducibility, and checkpoint checks; they do not measure contest accuracy.
