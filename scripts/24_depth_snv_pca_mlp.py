"""Depth-trajectory SNV -> shared PCA -> MLP, tuned for precision, recall and F1.

    python scripts/24_depth_snv_pca_mlp.py --n-jobs 8
    python scripts/24_depth_snv_pca_mlp.py --quick          # smoke test

Representation (``features.build_depth_bin_spectra``): shots 1-3 dropped,
log-spaced depth bins of the remaining shots, per-channel SNV on every bin
spectrum. Model: one PCA fitted on all bins of all training samples
(``models.BinnedPCA``), each bin projected onto it, the ``n_bins x n_pc``
score trajectory standardised and fed to an MLP. One row per sample.

A mean-centred shared PCA spends its leading components on the depth trend
common to every sample, so pixel scaling before the PCA is part of the search
(``scaling``: global centring vs. per-depth-bin autoscaling).

Hyper-parameters are searched on sample-level macro precision, macro recall
(= balanced accuracy), macro F1 and weighted F1. PCA and scaling are refitted
in every training fold. Selecting the best of many configurations on the same
folds is optimistic, so a nested CV (inner search inside every outer training
split) gives the honest estimate for each target metric.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from libs2026 import Config, plotting
from libs2026.features import build_depth_bin_spectra
from libs2026.models import RANDOM_STATE, BinnedPCA

CLASSES = np.array([1, 2, 3, 4, 5])
TARGETS = ("precision_macro", "recall_macro", "f1_macro", "f1_weighted")

# ``scaling`` is applied to the bin spectra before the shared PCA, fitted on the
# training fold: "center" leaves centring to the PCA (global mean over all bins),
# so the leading components mostly describe the depth trend shared by every
# sample; "bin_autoscale" centres and scales every pixel within each depth bin,
# so the PCA sees differences *between samples* at the same depth.
FULL_GRID = {
    "scaling": ["center", "bin_autoscale"],
    "n_bins": [4, 8, 12],
    "n_pc": [10, 20, 30, 50],
    "hidden": [(64,), (128, 64)],
    "alpha": [1e-3, 1e-2, 1e-1],
    "balance": [False, True],
}
QUICK_GRID = {"scaling": ["center", "bin_autoscale"], "n_bins": [8], "n_pc": [30], "hidden": [(64,)],
              "alpha": [1e-3], "balance": [False, True]}


def scores(y_true, y_pred) -> dict:
    return {
        "precision_macro": precision_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def make_splits(y, n_splits, n_repeats, seed):
    """``[(repeat, train_idx, test_idx), ...]``; one row per sample, so groups = rows."""
    out = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        out += [(r, tr, te) for tr, te in sgkf.split(np.zeros(len(y)), y, np.arange(len(y)))]
    return out


def _pca_scores(X, train_idx, test_idx, n_max, scaling="center"):
    """Fit one shared PCA on the training bins; return ``(n, n_bins, n_max)`` score cubes."""
    n_bins = X.shape[1]
    x_tr, x_te = X[train_idx].astype(np.float64), X[test_idx].astype(np.float64)
    if scaling == "bin_autoscale":
        mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        x_tr, x_te = (x_tr - mean) / std, (x_te - mean) / std
    elif scaling != "center":
        raise ValueError(f"Unknown scaling '{scaling}'")
    n_max = min(n_max, len(train_idx) * n_bins - 1)
    pca = BinnedPCA(n_bins=n_bins, n_components=n_max).fit(x_tr.reshape(len(x_tr), -1))
    k = pca._pca.n_components_
    tr = pca.transform(x_tr.reshape(len(x_tr), -1)).reshape(len(x_tr), n_bins, k)
    te = pca.transform(x_te.reshape(len(x_te), -1)).reshape(len(x_te), n_bins, k)
    return tr.astype(np.float32), te.astype(np.float32)


def precompute_scores(Xs, splits, n_max, scalings, n_jobs):
    """PCA per (split, n_bins, scaling) once; smaller ``n_pc`` are slices of the same components."""
    keys = [(s, n, sc) for s in range(len(splits)) for n in Xs for sc in scalings]
    out = Parallel(n_jobs=n_jobs)(
        delayed(_pca_scores)(Xs[n], splits[s][1], splits[s][2], n_max, sc) for s, n, sc in keys
    )
    return dict(zip(keys, out))


def oversample(y, rng):
    """Row indices that duplicate minority-class rows up to the largest class size."""
    counts = {c: np.flatnonzero(y == c) for c in np.unique(y)}
    top = max(len(v) for v in counts.values())
    return np.concatenate([np.r_[v, rng.choice(v, top - len(v), replace=True)] for v in counts.values()])


def fit_predict(config, s_tr, y_tr, s_te, seed):
    k = config["n_pc"]
    x_tr = s_tr[:, :, :k].reshape(len(s_tr), -1)
    x_te = s_te[:, :, :k].reshape(len(s_te), -1)
    if config["balance"]:
        idx = oversample(y_tr, np.random.default_rng(seed))
        x_tr, y_tr = x_tr[idx], y_tr[idx]
    scaler = StandardScaler().fit(x_tr)
    clf = MLPClassifier(hidden_layer_sizes=config["hidden"], alpha=config["alpha"], max_iter=2000,
                        random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(scaler.transform(x_tr), y_tr)
    proba = np.zeros((len(x_te), len(CLASSES)))
    proba[:, np.searchsorted(CLASSES, clf.classes_)] = clf.predict_proba(scaler.transform(x_te))
    return proba


def _run_config(config, cache, y, splits):
    out = []
    for s, (r, tr, te) in enumerate(splits):
        s_tr, s_te = cache[(s, config["n_bins"], config["scaling"])]
        out.append(fit_predict(config, s_tr, y[tr], s_te, seed=1000 * r + s))
    return out


def evaluate_grid(Xs, y, splits, grid, n_jobs):
    """Every configuration on every split: per-config OOF probabilities per repeat and scores."""
    n_max = max(grid["n_pc"])
    cache = precompute_scores(Xs, splits, n_max, grid["scaling"], n_jobs)
    configs = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    probas = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(_run_config)(c, cache, y, splits) for c in configs
    )
    n_repeats = max(r for r, _, _ in splits) + 1
    rows, oof = [], []
    for i, (config, per_split) in enumerate(zip(configs, probas)):
        rep = np.zeros((n_repeats, len(y), len(CLASSES)))
        for (r, _, te), p in zip(splits, per_split):
            rep[r, te] = p
        per_repeat = pd.DataFrame([scores(y, CLASSES[rep[r].argmax(axis=1)]) for r in range(n_repeats)])
        row = {"config_id": i, **{k: (str(v) if k == "hidden" else v) for k, v in config.items()}}
        for m in per_repeat:
            row[m] = per_repeat[m].mean()
            row[f"{m}_std"] = per_repeat[m].std(ddof=0)
        rows.append(row)
        oof.append(rep)
    return pd.DataFrame(rows), configs, oof


def pick_best(table, target):
    return int(table.sort_values([target, f"{target}_std"], ascending=[False, True]).iloc[0]["config_id"])


def nested_cv(Xs, y, grid, n_outer_repeats, inner_splits, seed, n_jobs):
    """Outer CV with an inner grid search per outer training split, for all targets at once."""
    outer = make_splits(y, 5, n_outer_repeats, seed)
    preds = {t: np.zeros((n_outer_repeats, len(y)), dtype=int) for t in TARGETS}
    chosen = {t: [] for t in TARGETS}
    for s, (r, tr, te) in enumerate(outer):
        inner = make_splits(y[tr], inner_splits, 1, seed + 100 + s)
        table, configs, _ = evaluate_grid({n: X[tr] for n, X in Xs.items()}, y[tr], inner, grid, n_jobs)
        for t in TARGETS:
            config = configs[pick_best(table, t)]
            chosen[t].append(config)
            s_tr, s_te = _pca_scores(Xs[config["n_bins"]], tr, te, config["n_pc"], config["scaling"])
            proba = fit_predict(config, s_tr, y[tr], s_te, seed=7 * s)
            preds[t][r, te] = CLASSES[proba.argmax(axis=1)]
        print(f"  nested outer split {s + 1}/{len(outer)} done", flush=True)
    rows = []
    for t in TARGETS:
        per_repeat = pd.DataFrame([scores(y, preds[t][r]) for r in range(n_outer_repeats)])
        rows.append({"target": t, **{f"nested_{m}": per_repeat[m].mean() for m in per_repeat},
                     **{f"nested_{m}_std": per_repeat[m].std(ddof=0) for m in per_repeat},
                     "nested_configs": "; ".join(sorted({json.dumps(c, default=str) for c in chosen[t]}))})
    return pd.DataFrame(rows)


def baseline_scores(cfg):
    path = cfg.predictions_dir / "oof_physics_pca_mlp.csv"
    if not path.exists():
        return None
    oof = pd.read_csv(path)
    per_repeat = pd.DataFrame([scores(g["y_true"], g["y_pred"]) for _, g in oof.groupby("repeat")])
    return per_repeat.mean().to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--drop-shots", type=int, default=3)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--nested-repeats", type=int, default=2)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--no-nested", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    grid = QUICK_GRID if args.quick else FULL_GRID
    n_repeats = 1 if args.quick else args.n_repeats
    nested_repeats = 1 if args.quick else args.nested_repeats
    tag = "depth_snv_pca_mlp" + ("_quick" if args.quick else "")

    data = build_depth_bin_spectra(cfg, args.drop_shots, tuple(grid["n_bins"]), n_jobs=args.n_jobs)
    first = data[grid["n_bins"][0]]
    is_train = first["split"] == "train"
    y = first["label"][is_train].astype(int)
    sample_ids = first["sample_ids"][is_train]
    Xs = {n: d["X"][is_train] for n, d in data.items()}
    for n, X in Xs.items():
        print(f"n_bins={n}: training cube {X.shape} (actual bins {X.shape[1]})")
    print(f"class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ---- search ----------------------------------------------------------------------
    splits = make_splits(y, cfg["cv"]["n_splits"], n_repeats, cfg["cv"]["random_state"])
    n_configs = int(np.prod([len(v) for v in grid.values()]))
    print(f"search: {n_configs} configurations x {len(splits)} folds", flush=True)
    table, configs, oof = evaluate_grid(Xs, y, splits, grid, args.n_jobs)
    table.to_csv(cfg.metrics_dir / f"{tag}_search.csv", index=False)

    # Sanity: macro recall is balanced accuracy.
    rep0 = CLASSES[oof[0][0].argmax(axis=1)]
    assert abs(recall_score(y, rep0, average="macro") - balanced_accuracy_score(y, rep0)) < 1e-12

    best_rows, best_json = [], {}
    for t in TARGETS:
        i = pick_best(table, t)
        row = table.loc[table["config_id"] == i].iloc[0]
        best_rows.append({"target": t, **row.to_dict()})
        best_json[t] = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in configs[i].items()},
                        "search_scores": {m: float(row[m]) for m in (*TARGETS, "accuracy")}}
        frames = []
        for r in range(n_repeats):
            f = pd.DataFrame(oof[i][r], columns=[f"p{c}" for c in CLASSES])
            f.insert(0, "repeat", r)
            f.insert(0, "y_true", y)
            f.insert(0, "sample_id", sample_ids)
            f["y_pred"] = CLASSES[oof[i][r].argmax(axis=1)]
            frames.append(f)
        pooled = pd.concat(frames)
        pooled.to_csv(cfg.predictions_dir / f"oof_{tag}_{t}.csv", index=False)
        plotting.plot_confusion(confusion_matrix(pooled["y_true"], pooled["y_pred"], labels=CLASSES), CLASSES,
                                cfg.figures_dir / f"confusion_{tag}_{t}.png",
                                title=f"best for {t}: {configs[i]}")
    best = pd.DataFrame(best_rows)

    # ---- nested CV -------------------------------------------------------------------
    if not args.no_nested:
        print(f"nested CV: 5 outer folds x {nested_repeats} repeats, inner {args.inner_splits}-fold search",
              flush=True)
        nested = nested_cv(Xs, y, grid, nested_repeats, args.inner_splits, cfg["cv"]["random_state"] + 500,
                           args.n_jobs)
        best = best.merge(nested, on="target")
        for _, row in nested.iterrows():
            best_json[row["target"]]["nested_scores"] = {m: float(row[f"nested_{m}"])
                                                          for m in (*TARGETS, "accuracy")}

    base = baseline_scores(cfg)
    if base is not None:
        best = pd.concat([best, pd.DataFrame([{"target": "baseline pca_mlp (mean, 4 blocks)", **base}])],
                         ignore_index=True)
    best.to_csv(cfg.metrics_dir / f"{tag}_best.csv", index=False)
    with open(cfg.models_dir / f"best_params_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump({"drop_shots": args.drop_shots, "n_repeats": n_repeats, "targets": best_json,
                   "baseline": base}, fh, indent=2, default=str)

    cols = ["target", "scaling", "n_bins", "n_pc", "hidden", "alpha", "balance", *TARGETS, "accuracy"]
    cols += [c for c in best.columns if c.startswith("nested_") and not c.endswith("_std")
             and c != "nested_configs"]
    print()
    print(best[[c for c in cols if c in best]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {cfg.metrics_dir / (tag + '_best.csv')}")


if __name__ == "__main__":
    main()
