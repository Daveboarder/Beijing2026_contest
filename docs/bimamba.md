# Bidirectional Mamba dual-pathway classifier

Each sample is the depth sequence of the spectral depth-transformer experiment
(`prepare_sample`, unchanged): 65 depth tokens — shots 1–20 kept separate,
shots 21–200 averaged in windows of four — each holding three shape-normalised
spectra binned by four within their detector channel (3 × 1023 = 3069 columns)
and three per-channel log intensities relative to the bulk. The classifier
standardises the tensor as its parent does (one scalar mean and standard
deviation per channel block, per column for the intensities) and reads it along
two axes: along wavelength and along depth.

## Architecture

**Region pooling.** A fixed matrix pools the 65 depth tokens into three
physical regions — surface (shots < 20), aged layer (shots < 140) and bulk —
weighting every token by the number of shots it covers. The result is three
pooled spectra `(3, 3069)` and three intensity triplets `(3, 3)` per sample.

**Spectral pathway.** Each pooled channel of 1023 bins is cut into 93
non-overlapping patches of 11 bins (1023 = 11 × 93), so one sample becomes 279
wavelength-ordered tokens. A token is the linear embedding of its 3 × 11 region
values plus a learned channel embedding and a sinusoidal position. One BiMamba
block and a LayerNorm follow. The path summary `h_s` is the average of mean
pooling and a learned attention pooling over the tokens, plus a projection of
the nine region intensities.

**Depth pathway.** Every one of the 65 tokens passes through a copy of the depth
transformer's per-channel convolutional stem (two Conv1d layers with 16 and 32
filters) whose adaptive average pooling is reduced to `stem_bins` = 4 bins per
channel (8 in the depth transformer), is projected to `d_model` and offset by
projections of its (centre, width) coordinate and its intensities. One BiMamba
block along depth and a LayerNorm follow. The summary
`h_d` is a linear map of the three region-weighted token means.

**BiMamba block.** Pre-LayerNorm, a shared input projection into a signal `h`
and a gate `z` (expansion 2, `d_inner = 128`), a forward S6 recurrence over
`h`, a backward S6 over the flipped `h` flipped back again, their sum
multiplied by `SiLU(z)`, an output projection, dropout and the residual. Each
S6 direction is the Mamba recurrence: a depthwise causal `Conv1d(k=4)` with
SiLU, an input-dependent step `Δ = softplus(·)` whose bias is initialised so
that steps start in [1e-3, 0.1], input-dependent `B` and `C`,
`A = -exp(A_log)` with `A_log` initialised to `log(1..d_state)`, and a `D`
skip. The block has 40.8k parameters at `d_model = 64`.

**Chunked selective scan.** The recurrence
`h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t`, `y_t = ⟨h_t, C_t⟩` runs in two levels:
a loop of `chunk` steps for all chunks at once from a zero state, then a loop
over the chunks that carries the entering state, and one vectorised
combination `state = local + (cumulative decay) × carry`. This takes
`chunk + ceil(L / chunk)` sequential steps instead of `L` (34 instead of 279 at
`chunk = 16`); `chunk = 1` is the plain loop. Every exponent is non-positive by
construction — `Δ > 0` from the softplus and `A < 0` from the negated
exponential — so the decays and their cumulative products lie in (0, 1] and
float32 cannot overflow, whatever the sequence length. A log-space parallel
scan would exponentiate sums over hundreds of steps and was rejected for that
reason. The naive sequential loop is kept as `selective_scan_reference`, the
test oracle.

**Adaptive fusion.** Four learned queries attend (multi-head attention, four
heads) over the concatenation of the spectral tokens, the depth tokens and the
two summaries, each tagged with a learned source embedding; the query outputs
are averaged into one cross-path vector. A content gate
`g = softmax(Linear(h_s ∥ h_d))` weights the two summaries, and the fused
embedding is `LayerNorm(g_s h_s + g_d h_d + cross)`. The attention costs
`K × (279 + 65 + 2)` dot products per sample, so it stays linear in the token
counts; token-level self-attention across the two paths would be quadratic and
was not used.

**Heads and loss.** Three identical heads (dropout, `Linear(64, 32)`, GELU,
`Linear(32, 5)`) score `h_s`, `h_d` and the fused embedding. The final logits
are the plain mean of the three — a fixed logit ensemble, since learned head
weights would overfit with 96 training samples per fold. The training loss is
class-weighted cross-entropy on the final logits plus `λ = 0.3` times the
cross-entropies of the spectral and depth heads, so each path must classify on
its own.

**Variants.** `full` is the model above. `spectral` and `depth` keep one
pathway and its head only (the final logits are that head). `concat` keeps both
pathways and the three heads but replaces attention and gate with
`LayerNorm(Linear(h_s ∥ h_d))`. All variants share preparation, folds and
training protocol, so they are the ablations of the fusion module. The fitted
estimator also exposes `transform` (the fused embedding) and `predict_heads`
(per-head probabilities and the gate).

**Augmentation and training.** In training mode only, the standardised batch
receives Gaussian noise (`σ = 0.05`) and depth jitter: each token is replaced
by its shallower neighbour with probability 0.1 (the first token never is).
Dropout 0.2, AdamW with weight decay 1e-3, gradient clipping at 1.0, batch
size 8, learning rate 3e-4. Epoch selection follows the depth transformer: an
inner stratified 20 % split picks the epoch count (maximum 100, patience 15)
on the unweighted validation cross-entropy of the final logits, then a fresh
network refits all outer training samples for that count. Seeds 42, 7 and 123
are ensembled by averaging probabilities.

Approximate parameter counts at `d_model = 64`, `d_state = 16`: `full` ≈ 155k,
`spectral` ≈ 46k, `depth` ≈ 89k, `concat` ≈ 146k. Every component — the two
chunked scans, the stems, the query attention — costs time linear in the
sequence lengths, so longer spectra or finer depth windows raise the cost
proportionally, not quadratically.

## Run

```bash
uv sync --extra cnn
make bimamba-smoke   # 2 folds x 1 repeat x seed 42 x 3 epochs, tag "smoke"
make bimamba         # 4 variants x 2 label sets x 5 folds x 4 repeats x 3 seeds, tag "initial"
```

Run the smoke target first. Every fit prints a line such as
`full labels=corrected repeat=0 seed=42 fold=0 selected_epochs=.. fit_s=.. eta_min=..`;
`fit_s` is the wall time of that fit (epoch selection plus refit) and
`eta_min` extrapolates the running mean fit time over the remaining fits of
the current grid. The smoke run trains three selection epochs plus a refit of
one to three epochs per fit (four to six epochs in all); the full run trains up
to 100 selection epochs (patience 15 may stop it earlier) plus a refit for the
selected count, so scale `fit_s` accordingly before judging the full grid of
480 fits. If the estimate exceeds a few hours, run
`spectral` and `concat` with `--seeds 42` or set `patch: 31` in the config
(99 spectral tokens). `--variants`, `--labels`, `--seeds`, `--folds`,
`--repeats` and `--epochs` narrow the grid from the command line.

Tags are directory names under `results/bimamba/<tag>/` and are never
overwritten; choose a new tag for every run. Each run writes `folds.csv`
(sample_id, repeat, fold), `oof.csv` (per-seed and ensemble out-of-fold
probabilities with the label used for training and the given label),
`summary.csv` and one `recipe_<variant>_<labels>.json` per variant and label
set. `summary.csv` and `oof.csv` are rewritten after every finished variant
and label set, so a partial run is still readable.

The folds are computed once, stratified on the **original** labels, and shared
by every variant and both label sets. `--labels both` trains each variant
twice: on the given labels and on the ten corrections of
`scripts/29_corrected_labels.py`. The summary reports two numbers for each row:

* `all120_*` — all 120 samples scored against the label set used for training.
  With corrected labels this number is optimistic: the corrections were chosen
  because several models predicted them, so scoring a model on them is partly
  circular.
* `unchanged110_*` — the 110 samples whose label nobody touched, the same
  subset for both label sets. This is the fair comparison: does training on
  the corrected labels help predict the samples that were never in question?

`*_accuracy_std` is the standard deviation across repeats of the per-repeat
seed-ensemble accuracy. Compare `unchanged110_*` with
`results/metrics/corrected_labels_g10.csv`, where `pca_mlp` reaches 0.820 on
the original and 0.832 on the corrected labels. Eight rows come from the same
out-of-fold scores, so picking the best one adds selection optimism.

## Predict

```bash
uv run --extra cnn python scripts/33_bimamba.py predict \
  --recipe results/bimamba/initial/recipe_full_corrected.json \
  --device cuda --output submissions/bimamba_full_corrected.csv
```

The recipe stores the configuration, preparation, model parameters, seeds, the
median selected epoch count per seed, training IDs and labels, the applied
corrections and SHA-256 hashes of the prepared training tensor and of
`bimamba.py` plus `depth_transformer.py`. Prediction rebuilds the training and
test tensors, recomputes both hashes and refuses to continue if the source,
the training IDs or the input hash differ. For a corrected-label recipe it
re-applies the corrections from `scripts/29_corrected_labels.py` and requires
the result to match the stored training labels. One network per seed is then
refitted on all 120 samples (no inner split) for its recorded epoch count,
probabilities are averaged over seeds, and the submission is validated against
the sample template (same files, no duplicates, labels 1–5) and written in
template order together with a joblib checkpoint of the models, recipe, sample
IDs and probabilities. Existing output files are never overwritten.
`--config` may override paths for another host but nothing else.

## Tests

```bash
make test-bimamba
uv run ruff check src/libs2026/bimamba.py scripts/33_bimamba.py tests/test_bimamba.py
```

The tests are synthetic, run on CPU and do not measure contest accuracy. They
check the chunked scan against the sequential oracle (values and input
gradients, chunk sizes 1, 4, 5 and 16, a length that is not a multiple of the
chunk), S6 causality, flip-equivariance of a block with tied directions,
finite outputs for inputs scaled by 100, the output keys and gate of every
variant, that gradients reach both pathways, both scan directions, the gate,
the queries, the attention and all heads, that every variant fits and predicts
(probability shape and sum, `classes_`, `best_epoch_`, `predict_heads`,
`transform`), inner-validation isolation, clone determinism and a joblib round
trip, that wavelength and depth order matter, the augmentation semantics, that
invalid inputs raise, the production shape `(2, 65, 3072) → (2, 5)` with fewer
than 200k parameters, learnability of a planted signal, and a CLI round trip
(benchmark on both label sets, predict, and the hash and label checks) on a
temporary synthetic cache.
