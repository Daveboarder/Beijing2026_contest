"""Benchmark the bidirectional Mamba dual-pathway classifier, or fit its recipe for submission.

uv run --extra cnn python scripts/33_bimamba.py benchmark --device cuda --tag initial
uv run --extra cnn python scripts/33_bimamba.py predict --recipe PATH --device cuda

``benchmark --smoke`` shrinks the grid to 2 folds, 1 repeat, seed 42 and 3 epochs (unless
the corresponding option is given explicitly) to exercise the whole pipeline and to read
the per-fit time before the full grid. Both label sets, the original labels and the
scripts/29 corrections, are evaluated on identical folds stratified on the original
labels; ``unchanged110_*`` scores only the samples no correction touched, which is the
fair comparison between the two.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from importlib import import_module
from pathlib import Path

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from libs2026.bimamba import VARIANTS, BiMambaClassifier
from libs2026.config import Config
from libs2026.depth_transformer import build_depth_sequences

LABEL_CHOICES = ("original", "corrected", "both")
# The model source and the shared preparation code, hashed in this order.
SOURCE_FILES = ("bimamba.py", "depth_transformer.py")
METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1")
SUMMARY_COLUMNS = [
    "model", "labels",
    "all120_accuracy", "all120_accuracy_std", "all120_balanced_accuracy", "all120_macro_f1",
    "unchanged110_accuracy", "unchanged110_accuracy_std", "unchanged110_balanced_accuracy",
    "unchanged110_macro_f1", "n_params", "fit_seconds_mean",
]


def metrics(y, p, classes):
    pred = classes[p.argmax(1)]
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)))


def _preparation(section, depth_section):
    """Prefer bimamba prep keys; fall back to depth_transformer, then to the defaults."""
    return {
        key: section.get(key, depth_section.get(key, default))
        for key, default in (("bin_factor", 4), ("surface_shots", 20), ("late_bin", 4))
    }


def _source_hash():
    """SHA-256 of the concatenated bytes of ``SOURCE_FILES`` under ``src/libs2026``."""
    return hashlib.sha256(b"".join(
        (Path(_bootstrap.SRC) / "libs2026" / name).read_bytes() for name in SOURCE_FILES
    )).hexdigest()


def _load_corrections():
    """Consensus label corrections of scripts/29; ``scripts/`` is sys.path[0] when run directly."""
    return dict(import_module("29_corrected_labels").CORRECTIONS)


def _apply_corrections(ids, y_orig, corrections):
    """Corrected label vector and the ``[sample_id, given, new]`` triples actually applied.

    Every correction whose sample is present must quote the label the sample
    currently carries; a mismatch means the corrections were written against
    different data. Corrections for absent samples are ignored.
    """
    position = {str(sid): i for i, sid in enumerate(ids)}
    applied = []
    for sid, (given, new) in corrections.items():
        if sid not in position:
            continue
        current = int(y_orig[position[sid]])
        if current != int(given):
            raise ValueError(f"{sid}: correction expects given label {given}, data has {current}")
        applied.append([str(sid), int(given), int(new)])
    y_corr = np.array([corrections.get(str(sid), (None, y))[1] for sid, y in zip(ids, y_orig)],
                      dtype=int)
    return y_corr, applied


def _label_sets(ids, y_orig, labels, corrections):
    """``([(name, y_fit), ...], changed mask, applied corrections)`` for the label choice.

    The corrections are loaded from scripts/29 only when a corrected label set is
    requested and none were passed in; with ``labels="original"`` and no
    corrections the changed mask is all False (unchanged110 then equals all120).
    """
    if labels not in LABEL_CHOICES:
        raise ValueError(f"labels must be one of {LABEL_CHOICES}, got {labels!r}")
    if corrections is None and labels != "original":
        corrections = _load_corrections()
    if corrections is None:
        y_corr, applied = y_orig.copy(), []
        changed = np.zeros(len(ids), dtype=bool)
    else:
        y_corr, applied = _apply_corrections(ids, y_orig, corrections)
        changed = np.isin(ids, list(corrections))
    label_sets = []
    if labels in ("original", "both"):
        label_sets.append(("original", y_orig))
    if labels in ("corrected", "both"):
        label_sets.append(("corrected", y_corr))
    return label_sets, changed, applied


def _setting(value, smoke_default, default, smoke):
    """Explicit argument first, then the smoke default, then the config default."""
    if value is not None:
        return value
    return smoke_default if smoke else default


def _rows(model, labels, repeat, seed, ids, y_fit, y_given, p, classes):
    """Out-of-fold probability rows of one (variant, label set, repeat, seed)."""
    return [dict(model=model, labels=labels, repeat=repeat, seed=seed, sample_id=str(sid),
                 y_true=int(y_fit[i]), y_given=int(y_given[i]),
                 **{f"p{c}": float(p[i, j]) for j, c in enumerate(classes)})
            for i, sid in enumerate(ids)]


def _summary_row(variant, labels, repeat_metrics, n_params, fit_seconds):
    row = dict(model=variant, labels=labels)
    for subset in ("all120", "unchanged110"):
        values = {key: [m[subset][key] for m in repeat_metrics] for key in METRIC_KEYS}
        row[f"{subset}_accuracy"] = float(np.mean(values["accuracy"]))
        row[f"{subset}_accuracy_std"] = float(np.std(values["accuracy"]))
        row[f"{subset}_balanced_accuracy"] = float(np.mean(values["balanced_accuracy"]))
        row[f"{subset}_macro_f1"] = float(np.mean(values["macro_f1"]))
    row["n_params"] = int(n_params)
    row["fit_seconds_mean"] = float(np.mean(fit_seconds))
    return row


def benchmark(cfg, args, corrections=None):
    section = cfg.get("bimamba", {})
    preparation = _preparation(section, cfg.get("depth_transformer", {}))
    smoke = bool(getattr(args, "smoke", False))
    variants = list(args.variants)
    if (not variants or len(set(variants)) != len(variants)
            or any(v not in VARIANTS for v in variants)):
        raise ValueError(f"variants must be distinct entries of {VARIANTS}, got {variants}")
    X, index = build_depth_sequences(cfg, **preparation)
    y_orig = index["label"].to_numpy(dtype=int)
    ids = index["sample_id"].to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Each physical sample must appear exactly once")
    classes = np.unique(y_orig)
    label_sets, changed, applied = _label_sets(ids, y_orig, args.labels, corrections)
    if changed.all():
        raise ValueError("Every sample carries a correction; nothing is left for unchanged110")
    cv = cfg["cv"]
    n_splits = int(_setting(args.folds, 2, cv["n_splits"], smoke))
    repeats = int(_setting(args.repeats, 1, cv["n_repeats"], smoke))
    seeds = [int(s) for s in _setting(args.seeds, [42], section.get("seeds", [42, 7, 123]), smoke)]
    epochs = int(_setting(args.epochs, 3, section.get("epochs", 100), smoke))
    if n_splits < 2 or repeats < 1 or min(np.unique(y_orig, return_counts=True)[1]) < n_splits:
        raise ValueError("Insufficient samples per class for the requested folds")
    if not seeds or len(set(seeds)) != len(seeds) or epochs < 1:
        raise ValueError("seeds must be distinct and epochs >= 1")
    split_seed = int(cv["random_state"])
    bounds = cfg["data"]["channel_bounds"]
    widths = tuple(int((b - a) // preparation["bin_factor"])
                   for a, b in zip(bounds[:-1], bounds[1:]))
    params = dict(
        channel_widths=widths, n_shots=cfg["data"]["n_shots"],
        surface_shots=preparation["surface_shots"], late_bin=preparation["late_bin"],
        bulk_start=section.get("bulk_start", 140),
        d_model=section.get("d_model", 64), d_state=section.get("d_state", 16),
        n_layers=section.get("n_layers", 1), patch=section.get("patch", 11),
        chunk=section.get("chunk", 16), n_queries=section.get("n_queries", 4),
        stem_bins=section.get("stem_bins", 4), dropout=section.get("dropout", 0.2),
        lambda_aux=section.get("lambda_aux", 0.3), noise_std=section.get("noise_std", 0.05),
        token_dropout=section.get("token_dropout", 0.1),
        epochs=epochs, patience=section.get("patience", 15),
        batch_size=section.get("batch_size", 8), lr=section.get("lr", 0.0003),
        weight_decay=section.get("weight_decay", 0.001),
    )
    # Folds are computed once, stratified on the ORIGINAL labels, and shared by every
    # variant and label set; they depend only on the repeat, never on the model seed.
    splits = [list(StratifiedGroupKFold(n_splits, shuffle=True,
                                        random_state=split_seed + repeat)
                   .split(X, y_orig, ids)) for repeat in range(repeats)]
    root = cfg.results_dir / "bimamba" / args.tag
    root.mkdir(parents=True, exist_ok=False)
    manifest = [dict(sample_id=str(ids[i]), repeat=repeat, fold=fold)
                for repeat, folds in enumerate(splits)
                for fold, (_, te) in enumerate(folds) for i in te]
    pd.DataFrame(manifest).to_csv(root / "folds.csv", index=False)
    input_hash = hashlib.sha256(X.tobytes() + y_orig.tobytes()).hexdigest()
    source_hash = _source_hash()
    environment = {name: importlib.metadata.version(name)
                   for name in ["torch", "numpy", "pandas", "scikit-learn"]}

    total_fits = len(variants) * len(label_sets) * repeats * len(seeds) * n_splits
    print(f"grid: {len(variants)} variants x {len(label_sets)} label sets x {repeats} repeats "
          f"x {len(seeds)} seeds x {n_splits} folds = {total_fits} fits "
          f"(changed samples: {int(changed.sum())}, epochs: {epochs}, smoke: {smoke})",
          flush=True)
    done, elapsed = 0, 0.0
    summary, all_predictions = [], []
    for variant in variants:
        for labels, y_fit in label_sets:
            epoch_records, repeat_metrics, fit_seconds = [], [], []
            n_params = None
            for repeat, folds in enumerate(splits):
                probabilities = []
                for seed in seeds:
                    p = np.zeros((len(y_fit), len(classes)))
                    for fold, (tr, te) in enumerate(folds):
                        model = BiMambaClassifier(**params, variant=variant,
                                                  device=args.device, random_state=seed)
                        start = time.perf_counter()
                        model.fit(X[tr], y_fit[tr])
                        fit_s = time.perf_counter() - start
                        if not np.array_equal(model.classes_, classes):
                            raise ValueError("Training fold is missing a class")
                        p[te] = model.predict_proba(X[te])
                        n_params = sum(int(q.numel()) for q in model.net_.parameters())
                        fit_seconds.append(fit_s)
                        done += 1
                        elapsed += fit_s
                        eta_min = elapsed / done * (total_fits - done) / 60
                        epoch_records.append(dict(repeat=repeat, seed=seed, fold=fold,
                                                  epochs=int(model.best_epoch_)))
                        print(f"{variant} labels={labels} repeat={repeat} seed={seed} "
                              f"fold={fold} selected_epochs={int(model.best_epoch_)} "
                              f"fit_s={fit_s:.1f} eta_min={eta_min:.1f}", flush=True)
                    probabilities.append(p)
                    all_predictions.extend(_rows(variant, labels, repeat, str(seed), ids,
                                                 y_fit, y_orig, p, classes))
                ensemble = np.mean(probabilities, axis=0)
                repeat_metrics.append(dict(
                    all120=metrics(y_fit, ensemble, classes),
                    unchanged110=metrics(y_fit[~changed], ensemble[~changed], classes)))
                all_predictions.extend(_rows(variant, labels, repeat, "ensemble", ids,
                                             y_fit, y_orig, ensemble, classes))
            row = _summary_row(variant, labels, repeat_metrics, n_params, fit_seconds)
            summary.append(row)
            recipe = dict(
                config=cfg.raw, preparation=preparation,
                model_params={**params, "variant": variant}, seeds=seeds,
                epochs_by_seed={str(seed): int(np.median(
                    [r["epochs"] for r in epoch_records if r["seed"] == seed]))
                    for seed in seeds},
                epoch_selection=epoch_records, repeat_metrics=repeat_metrics,
                cv=dict(n_splits=n_splits, n_repeats=repeats, random_state=split_seed),
                training_ids=[str(sid) for sid in ids], classes=classes.tolist(),
                labels=labels, training_labels=y_fit.tolist(),
                corrections=applied if labels == "corrected" else [],
                n_params=int(n_params),
                input_sha256=input_hash, source_sha256=source_hash,
                environment=environment, device=args.device,
            )
            (root / f"recipe_{variant}_{labels}.json").write_text(
                json.dumps(recipe, indent=2), encoding="utf-8")
            pd.DataFrame(summary, columns=SUMMARY_COLUMNS).to_csv(root / "summary.csv",
                                                                  index=False)
            pd.DataFrame(all_predictions).to_csv(root / "oof.csv", index=False)
            print(row, flush=True)
    print(f"Results and refit recipes: {root}")


def predict(args, corrections=None):
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    cfg = Config(raw=recipe["config"])
    # A different host can override locations, but not the evaluated preprocessing.
    if args.config:
        cfg.raw["paths"] = Config.load(args.config).raw["paths"]
    if _source_hash() != recipe["source_sha256"]:
        raise ValueError("Model source changed since benchmarking; generate a new recipe")
    X, train = build_depth_sequences(cfg, **recipe["preparation"])
    test_x, test = build_depth_sequences(cfg, split="test", **recipe["preparation"])
    y_orig = train["label"].to_numpy(dtype=int)
    ids = train["sample_id"].to_numpy()
    digest = hashlib.sha256(X.tobytes() + y_orig.tobytes()).hexdigest()
    if ([str(sid) for sid in ids] != list(recipe["training_ids"])
            or digest != recipe["input_sha256"]):
        raise ValueError("Training data/preprocessing changed since benchmarking")
    if recipe["labels"] == "corrected":
        if corrections is None:
            corrections = _load_corrections()
        y_fit, _ = _apply_corrections(ids, y_orig, corrections)
    elif recipe["labels"] == "original":
        y_fit = y_orig
    else:
        raise ValueError(f"Unknown label set {recipe['labels']!r} in the recipe")
    if y_fit.tolist() != list(recipe["training_labels"]):
        raise ValueError("Label corrections changed since benchmarking; generate a new recipe")
    classes = np.asarray(recipe["classes"])
    probabilities, models = [], []
    for seed in recipe["seeds"]:
        params = dict(recipe["model_params"])
        params["epochs"] = int(recipe["epochs_by_seed"][str(seed)])
        model = BiMambaClassifier(**params, val_fraction=0, device=args.device,
                                  random_state=seed)
        model.fit(X, y_fit)
        if not np.array_equal(model.classes_, classes):
            raise ValueError("Fitted classes differ from the recipe classes")
        models.append(model)
        probabilities.append(model.predict_proba(test_x))
    p = np.mean(probabilities, axis=0)
    submission = pd.DataFrame(dict(filename=test["sample_id"] + ".csv",
                                   predicted_label=classes[p.argmax(1)]))
    template = pd.read_csv(cfg.sample_submission)
    if (submission["filename"].duplicated().any() or len(submission) != len(template)
            or set(submission["filename"]) != set(template["filename"])
            or not submission["predicted_label"].isin([1, 2, 3, 4, 5]).all()):
        raise ValueError("Submission does not match the contest template")
    submission = submission.set_index("filename").loc[template["filename"]].reset_index()
    out = Path(args.output)
    checkpoint = out.with_suffix(".joblib")
    if out.exists() or checkpoint.exists():
        raise FileExistsError("Choose a new output path; refusing to overwrite predictions/model")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(models=models, recipe=recipe, probabilities=p,
                     sample_ids=test["sample_id"].tolist()), checkpoint)
    submission.to_csv(out, index=False)
    print(f"Submission: {out}; checkpoint: {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--config")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    bench.add_argument("--labels", choices=list(LABEL_CHOICES), default="both")
    bench.add_argument("--seeds", nargs="+", type=int)
    bench.add_argument("--folds", type=int)
    bench.add_argument("--repeats", type=int)
    bench.add_argument("--epochs", type=int)
    bench.add_argument("--smoke", action="store_true",
                       help="2 folds, 1 repeat, seed 42, 3 epochs unless given explicitly")
    bench.add_argument("--tag", default="initial")
    pred = sub.add_parser("predict")
    pred.add_argument("--config")
    pred.add_argument("--device", default="cpu")
    pred.add_argument("--recipe", required=True)
    pred.add_argument("--output", default="submissions/bimamba.csv")
    args = parser.parse_args()
    if args.command == "benchmark":
        if Path(args.tag).name != args.tag or args.tag in {".", ".."}:
            parser.error("tag must be a directory name, not a path")
        benchmark(Config.load(args.config), args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
