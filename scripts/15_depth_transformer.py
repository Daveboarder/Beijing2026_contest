"""Benchmark the spectral CNN/depth encoder, or fit its saved recipe for submission.

uv run --extra cnn python scripts/15_depth_transformer.py benchmark --device cuda
uv run --extra cnn python scripts/15_depth_transformer.py predict --recipe PATH --device cuda
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from libs2026.config import Config
from libs2026.depth_transformer import DepthTransformerClassifier, build_depth_sequences


def metrics(y, p, classes):
    pred = classes[p.argmax(1)]
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)))


def benchmark(cfg, args):
    section = cfg.get("depth_transformer", {})
    preparation = {key: section.get(key, default) for key, default in
                   (("bin_factor", 4), ("surface_shots", 20), ("late_bin", 4))}
    X, index = build_depth_sequences(cfg, **preparation)
    y = index["label"].to_numpy(dtype=int)
    ids = index["sample_id"].to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Each physical sample must appear exactly once")
    classes = np.unique(y)
    cv = cfg["cv"]
    n_splits = args.folds or cv["n_splits"]
    repeats = args.repeats or cv["n_repeats"]
    if n_splits < 2 or repeats < 1 or min(np.unique(y, return_counts=True)[1]) < n_splits:
        raise ValueError("Insufficient samples per class for the requested folds")
    split_seed = cv["random_state"]
    seeds = args.seeds or section.get("seeds", [42, 7, 123])
    bounds = cfg["data"]["channel_bounds"]
    widths = tuple((b - a) // preparation["bin_factor"]
                   for a, b in zip(bounds[:-1], bounds[1:]))
    params = dict(channel_widths=widths, n_shots=cfg["data"]["n_shots"],
                  surface_shots=preparation["surface_shots"], late_bin=preparation["late_bin"],
                  bulk_start=section.get("bulk_start", 140),
                  epochs=args.epochs or section.get("epochs", 100),
                  batch_size=section.get("batch_size", 8),
                  patience=section.get("patience", 15),
                  lr=section.get("lr", 0.0003), dropout=section.get("dropout", 0.2),
                  use_intensity=not args.no_intensity)
    # Fold partitions depend only on repeat, never on model initialization seed.
    splits = [list(StratifiedGroupKFold(n_splits, shuffle=True,
                                      random_state=split_seed + repeat)
                   .split(X, y, ids)) for repeat in range(repeats)]
    manifest = []
    for repeat, folds in enumerate(splits):
        for fold, (_, te) in enumerate(folds):
            manifest.extend(dict(sample_id=str(ids[i]), repeat=repeat, fold=fold) for i in te)
    root = cfg.results_dir / "depth_transformer" / args.tag
    root.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(manifest).to_csv(root / "folds.csv", index=False)
    summary, all_predictions = [], []
    for encoder in args.encoders:
        epoch_records, repeat_metrics = [], []
        for repeat, folds in enumerate(splits):
            probabilities = []
            for seed in seeds:
                p = np.zeros((len(y), len(classes)))
                for fold, (tr, te) in enumerate(folds):
                    model = DepthTransformerClassifier(
                        **params, depth_encoder=encoder, device=args.device, random_state=seed)
                    model.fit(X[tr], y[tr])
                    if not np.array_equal(model.classes_, classes):
                        raise ValueError("Training fold is missing a class")
                    p[te] = model.predict_proba(X[te])
                    epoch_records.append(dict(repeat=repeat, seed=seed, fold=fold,
                                              epochs=model.best_epoch_))
                    print(f"{encoder} repeat={repeat} seed={seed} fold={fold} "
                          f"selected_epochs={model.best_epoch_}", flush=True)
                probabilities.append(p)
                for i, sid in enumerate(ids):
                    all_predictions.append(dict(
                        model=encoder, repeat=repeat, seed=str(seed), sample_id=sid, y_true=y[i],
                        **{f"p{c}": p[i, j] for j, c in enumerate(classes)}))
            ensemble = np.mean(probabilities, axis=0)
            repeat_metrics.append(metrics(y, ensemble, classes))
            for i, sid in enumerate(ids):
                all_predictions.append(dict(
                    model=encoder, repeat=repeat, seed="ensemble", sample_id=sid, y_true=y[i],
                    **{f"p{c}": ensemble[i, j] for j, c in enumerate(classes)}))
        row = dict(model=encoder, accuracy_std=float(np.std(
            [m["accuracy"] for m in repeat_metrics])))
        row.update({key: float(np.mean([m[key] for m in repeat_metrics]))
                    for key in repeat_metrics[0]})
        summary.append(row)
        recipe = dict(
            config=cfg.raw, preparation=preparation,
            model_params={**params, "depth_encoder": encoder}, seeds=seeds,
            epochs_by_seed={str(seed): int(np.median(
                [r["epochs"] for r in epoch_records if r["seed"] == seed])) for seed in seeds},
            epoch_selection=epoch_records, repeat_metrics=repeat_metrics,
            cv=dict(n_splits=n_splits, n_repeats=repeats, random_state=split_seed),
            training_ids=list(ids), classes=classes.tolist(),
            input_sha256=hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
            source_sha256=hashlib.sha256(
                (Path(_bootstrap.SRC) / "libs2026" / "depth_transformer.py").read_bytes()
            ).hexdigest(),
            environment={name: importlib.metadata.version(name) for name in
                         ["torch", "numpy", "pandas", "scikit-learn"]},
            device=args.device,
        )
        (root / f"recipe_{encoder}.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        pd.DataFrame(summary).to_csv(root / "summary.csv", index=False)
        pd.DataFrame(all_predictions).to_csv(root / "oof.csv", index=False)
        print(row, flush=True)
    print(f"Results and refit recipes: {root}")


def predict(args):
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    cfg = Config(raw=recipe["config"])
    # A different host can override locations, but not the evaluated preprocessing.
    if args.config:
        cfg.raw["paths"] = Config.load(args.config).raw["paths"]
    source_hash = hashlib.sha256(
        (Path(_bootstrap.SRC) / "libs2026" / "depth_transformer.py").read_bytes()).hexdigest()
    if source_hash != recipe["source_sha256"]:
        raise ValueError("Model source changed since benchmarking; generate a new recipe")
    X, train = build_depth_sequences(cfg, **recipe["preparation"])
    test_x, test = build_depth_sequences(cfg, split="test", **recipe["preparation"])
    y = train["label"].to_numpy(dtype=int)
    digest = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    if list(train["sample_id"]) != recipe["training_ids"] or digest != recipe["input_sha256"]:
        raise ValueError("Training data/preprocessing changed since benchmarking")
    probabilities, models = [], []
    params = recipe["model_params"].copy()
    for seed in recipe["seeds"]:
        params["epochs"] = recipe["epochs_by_seed"][str(seed)]
        model = DepthTransformerClassifier(**params, val_fraction=0,
                                           device=args.device, random_state=seed)
        model.fit(X, y)
        models.append(model)
        probabilities.append(model.predict_proba(test_x))
    p = np.mean(probabilities, axis=0)
    classes = np.asarray(recipe["classes"])
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
    bench.add_argument("--encoders", nargs="+", choices=["pool", "conv", "transformer"],
                       default=["pool", "conv", "transformer"])
    bench.add_argument("--seeds", nargs="+", type=int)
    bench.add_argument("--folds", type=int)
    bench.add_argument("--repeats", type=int)
    bench.add_argument("--epochs", type=int)
    bench.add_argument("--no-intensity", action="store_true")
    bench.add_argument("--tag", default="initial")
    pred = sub.add_parser("predict")
    pred.add_argument("--config")
    pred.add_argument("--device", default="cpu")
    pred.add_argument("--recipe", required=True)
    pred.add_argument("--output", default="submissions/depth_transformer.csv")
    args = parser.parse_args()
    if args.command == "benchmark":
        if Path(args.tag).name != args.tag or args.tag in {".", ".."}:
            parser.error("tag must be a directory name, not a path")
        benchmark(Config.load(args.config), args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
