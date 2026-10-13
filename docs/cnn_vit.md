# 2-D CNN + vision transformer on depth-spectrum images

Each sample is one single-channel image: rows are depth, columns are
wavelength. `build_images` (bin factor 8, no shot averaging) gives the 200 shots
over 1533 wavelength bins — three detector channels of 511 bins each — and
`depth_images` drops the first 20 shots and averages the remaining 180 in groups
of four, so the network sees a `45 x 1533` image of shots 21–200. The first 20
shots are omitted because they are the least reproducible part of the profile
(the surface layer and the aged-layer dip of `scripts/08_depth_analysis.py`),
which also means that this model does not see the shots in which the classes
differ most; `drop_shots: 0` in the `cnn_vit` section of the config keeps them
(50 rows) and is the first knob to revisit if the model learns anything at all.

## Method and adaptation

The article (ViT-CNN-BS) classifies hyperspectral pixels from very few labelled
examples: a supervised band selection keeps the most discriminative spectral
bands, PCA compresses the remaining band stack, patches around each pixel are
augmented by CenterCrop+Resize, a residual 2-D CNN turns a patch into a grid of
feature vectors, the grid is flattened into tokens, a class token is prepended
and a vision-transformer encoder reads the sequence, the embedding is trained
episodically with a prototype loss whose query samples are mixed under the
guidance of the class-token attention (eq. 10–15), and an RBF SVM on the
embedding makes the final decision. The article pre-trains on a labelled source
scene and adapts to the target scene with a handful of samples; here there is
no source domain, so the embedding is trained episodically on the training fold
alone. The steps map onto the depth-spectrum images as follows.

**Standardisation.** Every wavelength column is standardised with its mean and
standard deviation over all training samples and all depth rows, fitted on the
outer training fold only (and, during episode selection, on the inner training
rows only). This is the lesson of the BiMamba experiment: one scalar mean and
standard deviation per detector channel prevented learning entirely (0.33–0.37,
the majority-class level); per-column standardisation is what makes a deep
network train on these images, even if it then overfits.

**Band selection.** The article's bands are the wavelength columns of the
image. `select_bands` compares the class-mean spectra (images averaged over
depth) for every pair of classes, counts for each pair the `top_per_pair`
columns with the largest absolute difference (`n_bands` of them by default; a
`CnnVitClassifier` argument, not a config key), and keeps the `n_bands` columns
selected most often, ties broken by the total difference, in their original
wavelength order. It uses the labels, so it is fitted on training samples only.
The default `n_bands: 0` keeps all 1533 columns. The article's PCA after band
selection is omitted because it compresses the spectral *depth* of a
hyperspectral cube, which has no counterpart here — wavelength is a spatial
axis of a one-channel image, and a PCA across columns would destroy the local
wavelength neighbourhoods that the convolutions rely on.

**CenterCrop+Resize.** `crop_resize` keeps a random fraction in
`[crop_min, 1]` (default 0.8) of the rows and of the columns around a jittered
centre and resizes the crop back to `45 x 1533` bilinearly, independently for
every image of a batch. Optional Gaussian noise (`noise_std`, default 0) is
added afterwards. Both apply in training only, to support and query samples
alike.

**Residual 2-D CNN.** A `3 x 3` mapping convolution (16 filters) is followed by
two `3 x 7` convolutions with stride 4 along wavelength (32 filters each), so
the long spectral axis shrinks from 1533 to 96 columns while the 45 depth rows
are kept; every convolution carries batch normalisation and ReLU. Then comes the
article's residual bottleneck block — `1 x 1 -> 3 x 3 -> 1 x 1` convolutions
(32 -> 16 -> 16 -> 32 channels) with a skip connection and a ReLU — and a
`1 x 1` projection to `d_model = 64`. Adaptive average pooling reduces the
`45 x 96` feature map to a `9 x 24` grid, so each token summarises about five
depth rows (20 shots) and 64 wavelength bins.

**Tokens and ViT.** The 216 grid cells are flattened row by row into a token
sequence, a learned class token is prepended, learned positions are added and
two pre-norm transformer encoder layers (4 heads, feed-forward width 128,
dropout 0.1) read the sequence. After a final LayerNorm the class token is the
64-dimensional embedding. The last layer's head-averaged attention of the class
token over the grid is returned as a `9 x 24` map (`attention_maps`), which the
query mixing below uses and which can be plotted to see which depth-wavelength
regions the model looks at.

**Episodic prototype loss.** Every episode draws, for each of the five classes,
`n_support = 3` support and `n_query = 4` query samples (with replacement when a
class is short), augments both sets, averages the support embeddings into class
prototypes, scores every query by its negative squared Euclidean distance to the
prototypes and applies the cross-entropy. AdamW (learning rate 1e-3, weight
decay 1e-3) with gradient clipping at 1.0 optimises the network.

**Attention-guided query mixing (eq. 10–15).** In the `proto_mix` variant
(`--variants proto_mix` of `scripts/34_cnn_vit.py`; the `mix_queries` key of
the config is ignored by the benchmark) the queries of an episode are paired by
rolling the batch.
For each pair a mask marks the grid cells in which the first query's class-token
attention is at or above its median; the mask is upsampled to the image and a
synthetic query takes the first query inside the mask and the second one
outside. Its weight `λ` is the first query's attention mass inside the mask
divided by that plus the second query's mass outside, and the synthetic query
is scored against the same prototypes with the loss `λ CE(y_1) + (1 − λ) CE(y_2)`.
The attention is detached, and the term is added with weight `mix_weight`
(default 0.5).

**Cross-entropy ablation.** The `ce` variant (`--variants ce`; the config's
`loss` key is likewise ignored) sends the 35 samples of each episode through a
linear head on the embedding and minimises a class-weighted cross-entropy;
everything else — augmentation, network, heads, episode selection — is
identical, so it tells whether the episodic prototype loss earns its place over
ordinary supervised training.

**Heads.** The training embeddings are standardised and classified by one of
three heads: the article's RBF SVM (`gamma="scale"`, `C = 1`), an MLP
(`128, 64` hidden units, `alpha = 1e-3`) or the nearest prototype (softmax
over the negative squared distance to the class means). `refit_head` returns a
copy of the fitted estimator that shares the network and fits another head on
the same training embedding, so the benchmark pays one network fit per fold for
all three heads.

**Episode selection and refit.** With `val_fraction = 0.2` a stratified inner
split of the training fold is made, band selection and standardisation are
fitted on its training part, and the network trains for up to `episodes` (600)
episodes while every `eval_every` (20) episodes the unweighted cross-entropy of
the inner validation samples is evaluated — against prototypes of the whole
inner training embedding for the prototype loss, through the linear head for
`ce`. Training stops after `patience` (5) evaluations without improvement, and
a fresh network with freshly fitted preprocessing is then trained on all
training samples for the best episode count. Prediction uses `val_fraction = 0`
and the median selected count of each seed.

At the defaults the network has about 120k parameters: 33k in the stem, 3.5k in
the residual block, 2k in the projection, 14k in the class token and positions,
67k in the two transformer layers and a few hundred in the linear head. The
exact count is the `n_params` column of `summary.csv`.

## Run

```bash
uv sync --extra cnn
make cnn-vit-smoke   # proto x 2 label sets x 2 folds x 1 repeat x seed 42 x 20 episodes, tag "smoke"
make cnn-vit         # proto x 2 label sets x 5 folds x 4 repeats x 3 seeds, tag "initial"
```

Run the smoke target first. Every network fit prints a line such as
`proto labels=corrected repeat=0 seed=42 fold=0 selected_episodes=.. fit_s=.. eta_min=..`;
`fit_s` is the wall time of that fit (episode selection plus refit) and
`eta_min` extrapolates the running mean fit time over the remaining fits of the
current grid. The smoke run trains 20 selection episodes plus a refit; the full
run trains up to 600 selection episodes (patience 5 evaluations, i.e. 100
episodes without improvement, may stop it earlier) plus the refit, so scale
`fit_s` accordingly before judging the 120 fits of the default grid.
`--variants proto proto_mix ce` runs all three variants (three times the grid);
`--heads`, `--labels`, `--seeds`, `--folds`, `--repeats` and `--episodes`
narrow or widen the grid from the command line, and `--heads` costs no extra
network fit, since every head is fitted on the same embedding. `--tag` names
the output directory.

Tags are directory names under `results/cnn_vit/<tag>/` and are never
overwritten; choose a new tag for every run. Each run writes `folds.csv`
(sample_id, repeat, fold), `oof.csv` (per-seed and ensemble out-of-fold
probabilities for every `variant+head` model with the label used for training
and the given label), `summary.csv` (one row per variant, head and label set)
and one `recipe_<variant>_<labels>.json` per variant and label set.
`summary.csv` and `oof.csv` are rewritten after every finished variant and
label set, so a partial run is still readable.

The folds are computed once, stratified on the **original** labels, and shared
by every variant, head and label set. `--labels both` trains each variant twice:
on the given labels and on the ten corrections of
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
the original and 0.832 on the corrected labels. The three heads of a variant
and label set are scored on the same out-of-fold embedding, so picking the best
head adds selection optimism.

Expectations are modest. The BiMamba run reached only 0.33–0.37 because its
per-channel scalar standardisation blocked learning; with per-column
standardisation deep networks on these images reached a training accuracy of
1.0 and a test accuracy of 0.42–0.50, far below `pca_mlp`. So treat this
experiment as an honest test of the article's recipe — band selection,
CenterCrop+Resize, episodic prototypes, attention-guided mixing, SVM on the
embedding — on 120 samples, not as a likely winner. The interesting readings
are the gaps: `proto` against `ce` (does the episodic loss help), `proto_mix`
against `proto` (does the mixing regularise), and the SVM against the MLP and
prototype heads on one and the same embedding.

## Predict

```bash
uv run --extra cnn python scripts/34_cnn_vit.py predict \
  --recipe results/cnn_vit/initial/recipe_proto_corrected.json \
  --head svm --device cuda --output submissions/cnn_vit_proto_corrected.csv
```

The recipe stores the configuration, the image preparation (bin factor,
dropped shots, shot bin), the estimator parameters including loss and mixing,
the heads evaluated, the seeds, the median selected episode count per seed,
training IDs and labels, the applied corrections and SHA-256 hashes of the
prepared training images and of `cnn_vit.py` plus `images.py`. Prediction
rebuilds the training and test images, recomputes both hashes and refuses to
continue if the source, the training IDs or the input hash differ. For a
corrected-label recipe it re-applies the corrections from
`scripts/29_corrected_labels.py` and requires the result to match the stored
training labels. `--head` picks one of the heads recorded in the recipe
(default `svm`). One network per seed is then refitted on all 120 samples (no
inner split) for its recorded episode count, probabilities are averaged over
seeds, and the submission is validated against the sample template (same
files, no duplicates, labels 1–5) and written in template order together with
a joblib checkpoint of the models, recipe, sample IDs and probabilities.
Existing output files are never overwritten. `--config` may override paths for
another host but nothing else.

## Tests

```bash
make test-cnn-vit
uv run ruff check src/libs2026/cnn_vit.py scripts/34_cnn_vit.py tests/test_cnn_vit.py
```

The tests are synthetic, run on CPU and do not measure contest accuracy. They
exercise the data helpers (dropped shots and row averaging, band selection,
crop-and-resize), the network's output keys and shapes, the estimator's fit and
prediction with every loss and head (probability shape and sum, `classes_`,
`best_episodes_`, `transform`, `attention_maps`, `refit_head`), that invalid
inputs raise, and the command-line round trip (benchmark on both label sets,
predict, and the hash and label checks) on a temporary synthetic cache.
