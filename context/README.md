# LIBS Foundation Model

A self-supervised foundation model for Laser Induced Breakdown Spectroscopy (LIBS) using a BERT-style transformer. Training uses the **physics-based LIBS pipeline** (Voigt synthesis, ~17k wavelength bins, HDF5 cache) with a choice of **bin (intensity) embedding** or **line-token embedding**.

## Overview

1. **Pre-trains** on unlabeled spectra (Masked Intensity Prediction on bins, or masked line-feature prediction on spectral lines)
2. **Fine-tunes** for classification, regression, **binned quantification** (`quantification_binned` with decoded R²), and **element detection** (`detection` — multi-label presence/absence vs per-element limits of detection)
3. **Data** — `config/libs_data.yaml` (~112k shots from the sample matrix) or `config/libs_data_smoke.yaml` for quick tests

See [ARCHITECTURE.md](ARCHITECTURE.md) for design detail, [PROJECT_SUMMARY.html](PROJECT_SUMMARY.html) / [PROJECT_SUMMARY.json](PROJECT_SUMMARY.json) for full CLI and layout reference.

## Embedding modes


| Mode                  | Config                                | CLI                                                      | Sequence           | Pre-train target                         |
| --------------------- | ------------------------------------- | -------------------------------------------------------- | ------------------ | ---------------------------------------- |
| **Bin (intensity)**   | `embedding_type: intensity` (default) | `--libs_data_config` only                                | ~17,428 bins + CLS | Masked bin intensities (MIP)             |
| **Line-token**        | `embedding_type: line_token`          | `--libs_data_config` + `--line_embedding_config`         | ~2k–8k lines + CLS | Masked Voigt features (max_I, FWHM)      |
| **Line-token-linear** | `embedding_type: line_token_linear`   | same CLI (default when `--line_embedding_config` is set) | ~2k–8k lines + CLS | Same MIP targets; reads pre-baked tokens |


Checkpoints are **not interchangeable** between modes (different sequence length and weights).

### Line-token pipeline (three HDF5 caches)

Offline preprocessing (reusable across training runs):

1. **Line dictionary** (`data/line_dictionary.py`) — theoretical intensities over a Te×Ne grid from `LIBS_data_vacuum.db`; keep top 10% most intense lines per element (or all lines if element has <10) → `line_dict_*.h5`
2. **Line features** (`data/line_features.py`) — per-spectrum Voigt fit at each line centre → `line_features_*.h5` `[n_spectra, n_lines, 6]`
3. **Line tokens** (`data/line_tokenization.py`) — merge dictionary + fits into one tensor per line → `line_tokens_*.h5` `[n_spectra, n_lines, 14]` (raw values + `feature_mean`/`feature_std` in attrs)

At training time:

- `**line_token`** — loads `line_features_*.h5`; `LineTokenEmbedding` concatenates quantum scalars + `nn.Embedding` element/ion + fit features at runtime.
- `**line_token_linear**` — loads only `line_tokens_*.h5`; `LinearLineTokenEmbedding` z-scores the 14 channels and applies `nn.Linear(14, d_model)`.

Build tokens once:

```bash
uv run python scripts/build_line_tokens.py \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml
```

**Token features (14 channels, stored raw):** wavelength, Ei, Ek, log10(gi/gk/Ak/I_theory), atomic number Z, ion_binary (0=I, 1=II+), max_intensity, FWHM, R², Δλ, RMSE. `fit_valid` is a separate uint8 mask for the encoder.

---

## Quick start

```bash
uv sync
```

### Pre-train — bin embedding (RTX 4090)

```bash
uv run python train_pretrain.py \
  --config config/config_libs_4090.yaml \
  --libs_data_config config/libs_data.yaml \
  --experiment_name libs_bin_pretrain \
  --num_workers 4
```

### Pre-train — line-token (runtime embedding)

```bash
uv run python train_pretrain.py \
  --config config/config_libs_token_4090.yaml \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml \
  --experiment_name libs_line_pretrain \
  --num_workers 8
```

First run builds `line_dict_*.h5` and `line_features_*.h5` (Voigt fits can take hours on full data).

### Pre-train — line-token-linear (pre-baked tokens + `nn.Linear`)

```bash
# Optional: materialize line_tokens_*.h5 first (skipped if already cached)
uv run python scripts/build_line_tokens.py \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml

uv run python train_pretrain.py \
  --config config/config_libs_token_linear_4090.yaml \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml \
  --experiment_name libs_token_linear_pretrain \
  --num_workers 4
```

Training only touches `line_tokens_*.h5` (dictionary and Voigt caches remain as intermediate artefacts).

### Fine-tune (binned quantification)

```bash
uv run python train_finetune.py \
  --config config/config_libs_4090.yaml \
  --pretrain_run_dir runs/pretrain_<your_run> \
  --libs_data_config config/libs_data.yaml \
  --task quantification_binned \
  --pool cls_mean \
  --experiment_name libs_binned_ft
```

Use the **same** `--config`, `--libs_data_config`, and (for line-token) `--line_embedding_config` as pre-training.

### Fine-tune (element detection)

Multi-label **presence/absence** per element: concentration ≥ LOD → present (1), else absent (0). LODs are defined in `config/element_lod.yaml` (mass fraction, per element + `default_lod`).

```bash
uv run python train_finetune.py \
  --config config/config_libs_token_linear_4090.yaml \
  --pretrain_run_dir runs/pretrain_<your_run> \
  --libs_data_config config/libs_data.yaml \
  --line_embedding_config config/line_embedding.yaml \
  --task detection \
  --pool cls_mean \
  --element_lod_config config/element_lod.yaml \
  --experiment_name libs_detection
```

Monitored metric: `val/det_f1`. Test results include micro/macro F1 and per-element precision, recall, F1, and support in `run_info.yaml`.

```bash
uv run python list_runs.py --latest
uv run tensorboard --logdir runs/
```

If training fails with **CUDA GPU is busy**, another process is holding the device (exclusive mode). Check `nvidia-smi`, stop the blocking PID, then re-run.

---

## Analysis & publication figures

### Attention importance

CLS-token attention over spectral lines (shared encoder representation; not per-output attribution):

```bash
uv run python analyze_attention_importance.py \
  --run_dir runs/finetune_<your_run> \
  --use_token_cache
```

Writes `evaluation/attention_importance_<timestamp>/` (CSVs + PNG heatmaps).

### Publication figure set

PowerPoint-ready figures (300 dpi PNG + editable SVG, white background). **Requires** an `attention_importance_*` folder from the step above.

```bash
uv run python make_publication_figures.py \
  --run_dir runs/finetune_<your_run>

# selected figures only
uv run python make_publication_figures.py --run_dir runs/finetune_<your_run> --only fig1,fig3

# skip checkpoint inference (attention-only figures)
uv run python make_publication_figures.py --run_dir runs/finetune_<your_run> --skip-inference
```

| Task | Inference figures |
| ---- | ----------------- |
| `quantification_binned` | `fig4_pred_vs_true`, `fig4b_per_element_r2` |
| `detection` | `fig4_presence_detection`, `fig4b_per_element_f1` |

Shared outputs: annotated spectrum, attention heatmaps, training curves, t-SNE embedding map, graphical abstract, Voigt-fit zoom, GIFs. See `FIGURES_README.txt` in the output folder.

### Raw vs tokenized comparison (detection)

Side-by-side figures for two fine-tuned detection runs (intensity vs `line_token_linear`):

```bash
uv run python make_comparison_figures.py \
  --run_dir_raw runs/finetune_2026-06-17_10-44-48_libs_detection_bin_4090 \
  --run_dir_token runs/finetune_2026-06-14_20-14-42_libs_detection
```

Outputs `cmp_fig0`–`cmp_fig7` (aggregate metrics, per-element F1, ΔF1, training curves, embeddings, detection panels, composite) under `runs/comparison_detection_raw_vs_token/publication_<timestamp>/`. The methodology panel documents pretrain differences when runs are not from a matched comparison campaign.

### VASKUT spectrum zoom (element lines)

Plot a 5 nm spectrum slice from `VASKUT K8.json` with color-coded theoretical DB lines (one strongest line per element in the window):

```bash
uv run python scripts/plot_vaskut_spectrum_zoom.py
# manual window: --wl_min 250 --wl_max 255
```

Saves PNG + SVG to `Outputs/`.

---

## Configuration

### Model / training


| Config                                      | Embedding         | GPU      | Notes                          |
| ------------------------------------------- | ----------------- | -------- | ------------------------------ |
| `config/config_libs_smoke.yaml`             | bin               | any      | Smoke wiring                   |
| `config/config_libs_4090.yaml`              | bin               | RTX 4090 | ~17k bins, 5 pretrain epochs   |
| `config/config_libs_a100.yaml`              | bin               | A100     | 60/30 epochs                   |
| `config/config_libs_token_4090.yaml`        | line_token        | RTX 4090 | ~8k lines (default threshold)  |
| `config/config_libs_token.yaml`             | line_token        | any      | Small model for smoke          |
| `config/config_libs_token_linear_4090.yaml` | line_token_linear | RTX 4090 | Pre-baked tokens + `nn.Linear` |
| `config/config_libs_token_linear.yaml`      | line_token_linear | any      | Smoke for token-linear path    |


### Data and line embedding


| Config                             | Role                                                                                    |
| ---------------------------------- | --------------------------------------------------------------------------------------- |
| `config/libs_data.yaml`            | Full dataset (~2256 types × 50 shots)                                                   |
| `config/libs_data_smoke.yaml`      | 3 types × 6 shots                                                                       |
| `config/line_embedding.yaml`       | Te/Ne grid, line-selection mode (`top_percent_per_element` default), Voigt-fit settings |
| `config/line_embedding_smoke.yaml` | `max_lines: 400` for fast tests                                                         |
| `config/element_lod.yaml`          | Per-element limits of detection (mass fraction) for the `detection` task                |


**Line selection:** `line_dictionary.selection` in `line_embedding.yaml` defaults to top 10% per element (`min_keep: 10`); legacy threshold mode remains available via `selection.mode: threshold`.

---

## Training CLI (essentials)


| Flag                      | Description                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--config`                | Model size, epochs, batch size                                                                                |
| `--libs_data_config`      | **Required** for real LIBS data (`libs_data.yaml`)                                                            |
| `--line_embedding_config` | Line-token modes (`line_embedding.yaml`); defaults to `line_token_linear` unless config sets `embedding_type` |
| `--experiment_name`       | Run folder suffix                                                                                             |
| `--num_workers`           | DataLoader workers (default 0)                                                                                |


**Fine-tune:** `--pretrain_run_dir`, `--task` (`quantification_binned`, `detection`, `classification`, …), `--pool` (`cls`, `mean`, `cls_mean`). For `detection`, also `--element_lod_config` (default `config/element_lod.yaml`).

---

## Compare bin vs line-token (same data)

```bash
LIBS=config/libs_data.yaml
SEED=42

# Bin
uv run python train_pretrain.py --config config/config_libs_4090.yaml \
  --libs_data_config $LIBS --seed $SEED --experiment_name compare_bin

# Line-token (runtime embedding)
uv run python train_pretrain.py --config config/config_libs_token_4090.yaml \
  --libs_data_config $LIBS --line_embedding_config config/line_embedding.yaml \
  --seed $SEED --experiment_name compare_line

# Line-token-linear (pre-baked tokens)
uv run python train_pretrain.py --config config/config_libs_token_linear_4090.yaml \
  --libs_data_config $LIBS --line_embedding_config config/line_embedding.yaml \
  --seed $SEED --experiment_name compare_line_linear
```

Match `d_model`, `n_layers`, and epochs in both configs before comparing metrics.

---

## Project layout (main paths)

```
LIBS_foundation/
├── config/
│   ├── config_libs_{4090,a100,smoke}.yaml      # bin embedding
│   ├── config_libs_token_{4090,}.yaml          # line_token (runtime embedding)
│   ├── config_libs_token_linear_{4090,}.yaml   # line_token_linear (pre-baked tokens)
│   ├── libs_data.yaml, libs_data_smoke.yaml
│   ├── line_embedding.yaml, line_embedding_smoke.yaml
│   └── element_lod.yaml
├── data/
│   ├── libs_pipeline.py
│   ├── line_dictionary.py, line_features.py, line_tokenization.py
│   ├── line_embedding_pipeline.py
│   └── dataset.py
├── scripts/build_line_tokens.py                # offline tokenization CLI
├── models/
│   ├── libs_transformer.py, line_token_embedding.py, heads.py
├── training/pretrain.py, training/finetune.py
├── train_pretrain.py, train_finetune.py
├── analyze_attention_importance.py             # CLS attention over lines
├── make_publication_figures.py                 # publication PNG/SVG + GIFs
├── evaluate_model.py, list_runs.py
└── run_pretrain_libs.slurm, run_finetune_libs.slurm
```

---

## SLURM (A100, bin embedding)

```bash
sbatch run_pretrain_libs.slurm
sbatch run_finetune_libs.slurm runs/pretrain_<timestamp>_libs_pretrain_a100
```

Extend SLURM scripts with `--line_embedding_config config/line_embedding.yaml` and `config_libs_token_4090.yaml` (runtime) or `config_libs_token_linear_4090.yaml` (pre-baked tokens) for line-token runs.

---

## License

MIT