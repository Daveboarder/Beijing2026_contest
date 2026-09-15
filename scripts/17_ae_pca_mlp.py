"""Benchmark or submit AE CLS ∥ PCA(mean_lines, g4) → MLP fusion.

Classical features use ``n_groups=4`` (same as the strong ``pca_mlp`` setup).
Each sample's CLS is **broadcast** onto its classical rows; OOF metrics average
row probabilities back to one vote per physical sample.

uv run --extra cnn python scripts/17_ae_pca_mlp.py benchmark --device cuda --tag g4_broadcast
uv run --extra cnn python scripts/17_ae_pca_mlp.py predict --recipe PATH --device cuda
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

from libs2026.ae_pca_mlp import AEPCAClassifier, remap_row_indices, row_sample_indices
from libs2026.config import Config
from libs2026.depth_transformer import build_depth_sequences
from libs2026.features import build_features
from libs2026.preprocessing import Preprocessor


def metrics(y, p, classes):
    pred = classes[p.argmax(1)]
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)))


def _ae_section(cfg):
    """Merge ae_pca_mlp AE keys with autotransformer / depth_transformer fallbacks."""
    fusion = cfg.get("ae_pca_mlp", {})
    auto = cfg.get("autotransformer", {})
    depth = cfg.get("depth_transformer", {})
    preparation = {
        key: fusion.get(key, auto.get(key, depth.get(key, default)))
        for key, default in (("bin_factor", 4), ("surface_shots", 20), ("late_bin", 4))
    }
    ae_keys = (
        "d_model", "n_layers", "n_heads", "ff_dim", "dropout", "lambda_recon",
        "epochs", "patience", "batch_size", "lr", "weight_decay",
    )
    ae_defaults = dict(
        d_model=64, n_layers=2, n_heads=4, ff_dim=128, dropout=0.1,
        lambda_recon=1.0, epochs=100, patience=15, batch_size=8,
        lr=0.0003, weight_decay=0.001,
    )
    ae_hparams = {
        key: fusion.get(key, auto.get(key, ae_defaults[key])) for key in ae_keys
    }
    return fusion, preparation, ae_hparams


def _classical_matrix(cfg, n_jobs=8):
    fusion = cfg.get("ae_pca_mlp", {})
    feat_cfg = cfg.get("features", {})
    encoding = fusion.get("encoding", feat_cfg.get("encoding", "mean_lines"))
    n_bins = fusion.get("n_bins", feat_cfg.get("n_bins", 8))
    augment = fusion.get("augment", feat_cfg.get("augment", "surface"))
    bin_factor = fusion.get("classical_bin_factor", 1)
    n_groups = int(fusion.get("n_groups", 4))
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    features = build_features(
        cfg, Preprocessor.from_config(cfg), n_groups=n_groups, bin_factor=bin_factor,
        encoding=encoding, n_bins=n_bins, augment=augment, n_jobs=n_jobs,
    )
    return features, dict(encoding=encoding, n_bins=n_bins, augment=augment,
                          bin_factor=bin_factor, n_groups=n_groups)


def _ae_params(cfg, preparation, ae_hparams, n_spectral, n_features, epochs=None):
    params = dict(
        n_spectral=n_spectral, n_features=n_features,
        n_shots=cfg["data"]["n_shots"],
        surface_shots=preparation["surface_shots"], late_bin=preparation["late_bin"],
        **ae_hparams,
    )
    if epochs is not None:
        params["epochs"] = epochs
    return params


def _aggregate_sample_proba(row_proba, row_depth_idx, n_samples, n_classes):
    """Mean classical-row probabilities per depth sample."""
    out = np.zeros((n_samples, n_classes), dtype=np.float64)
    counts = np.zeros(n_samples, dtype=np.int64)
    for p, idx in zip(row_proba, row_depth_idx):
        out[idx] += p
        counts[idx] += 1
    if np.any(counts == 0):
        raise ValueError("Missing classical rows for some depth samples")
    return out / counts[:, None]


def benchmark(cfg, args):
    fusion, preparation, ae_hparams = _ae_section(cfg)
    if args.epochs is not None:
        ae_hparams = {**ae_hparams, "epochs": args.epochs}
    X_depth, index = build_depth_sequences(cfg, **preparation)
    y = index["label"].to_numpy(dtype=int)
    ids = index["sample_id"].to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Each physical sample must appear exactly once in depth data")
    classical_features, classical_meta = _classical_matrix(cfg, n_jobs=args.n_jobs)
    train_c = classical_features.subset("train")
    X_classical = train_c.X
    y_classical = train_c.y.astype(int)
    classical_ids = train_c.sample_ids
    row_to_depth = row_sample_indices(ids, classical_ids)
    if not np.array_equal(y_classical, y[row_to_depth]):
        raise ValueError("Classical and depth labels disagree for shared sample_ids")
    classes = np.unique(y)
    cv = cfg["cv"]
    n_splits = args.folds or cv["n_splits"]
    repeats = args.repeats or cv["n_repeats"]
    if n_splits < 2 or repeats < 1 or min(np.unique(y, return_counts=True)[1]) < n_splits:
        raise ValueError("Insufficient samples per class for the requested folds")
    split_seed = cv["random_state"]
    seeds = args.seeds or fusion.get("seeds", cfg.get("autotransformer", {}).get(
        "seeds", [42, 7, 123]))
    bounds = cfg["data"]["channel_bounds"]
    n_spectral = sum((b - a) // preparation["bin_factor"]
                     for a, b in zip(bounds[:-1], bounds[1:]))
    ae_params = _ae_params(cfg, preparation, ae_hparams, n_spectral, X_depth.shape[-1])
    n_pca = fusion.get("n_pca", 30)
    mlp_hidden = tuple(fusion.get("mlp_hidden", [256, 128]))
    mlp_alpha = fusion.get("mlp_alpha", 1e-4)
    mlp_max_iter = fusion.get("mlp_max_iter", 2000)

    splits = [list(StratifiedGroupKFold(n_splits, shuffle=True,
                                        random_state=split_seed + repeat)
                   .split(X_depth, y, ids)) for repeat in range(repeats)]
    manifest = []
    for repeat, folds in enumerate(splits):
        for fold, (_, te) in enumerate(folds):
            manifest.extend(dict(sample_id=str(ids[i]), repeat=repeat, fold=fold)
                            for i in te)
    root = cfg.results_dir / "ae_pca_mlp" / args.tag
    root.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(manifest).to_csv(root / "folds.csv", index=False)

    epoch_records, repeat_metrics, all_predictions = [], [], []
    for repeat, folds in enumerate(splits):
        p = np.zeros((len(y), len(classes)))
        for fold, (tr, te) in enumerate(folds):
            row_tr, local_tr = remap_row_indices(row_to_depth, tr)
            row_te, local_te = remap_row_indices(row_to_depth, te)
            model = AEPCAClassifier(
                ae_params=ae_params, seeds=seeds, n_pca=n_pca,
                mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha,
                mlp_max_iter=mlp_max_iter, random_state=seeds[0],
                device=args.device,
            )
            model.fit(
                X_depth[tr], X_classical[row_tr], y_classical[row_tr],
                row_sample_idx=local_tr,
            )
            if not np.array_equal(model.classes_, classes):
                raise ValueError("Training fold is missing a class")
            row_p = model.predict_proba(
                X_depth[te], X_classical[row_te], row_sample_idx=local_te,
            )
            p[te] = _aggregate_sample_proba(row_p, local_te, len(te), len(classes))
            for seed, ae in zip(seeds, model.ae_models_):
                epoch_records.append(dict(repeat=repeat, seed=seed, fold=fold,
                                          epochs=ae.best_epoch_))
            print(f"ae_pca_mlp repeat={repeat} fold={fold} "
                  f"n_groups={classical_meta['n_groups']} "
                  f"n_pca={model.n_pca_} d_model={model.d_model_}", flush=True)
        repeat_metrics.append(metrics(y, p, classes))
        for i, sid in enumerate(ids):
            all_predictions.append(dict(
                model="ae_pca_mlp", repeat=repeat, seed="ensemble",
                sample_id=sid, y_true=y[i],
                **{f"p{c}": p[i, j] for j, c in enumerate(classes)}))

    row = dict(model="ae_pca_mlp",
               accuracy_std=float(np.std([m["accuracy"] for m in repeat_metrics])))
    row.update({key: float(np.mean([m[key] for m in repeat_metrics]))
                for key in repeat_metrics[0]})
    recipe = dict(
        config=cfg.raw, preparation=preparation, classical=classical_meta,
        ae_params=ae_params, seeds=seeds, n_pca=n_pca,
        mlp_hidden=list(mlp_hidden), mlp_alpha=mlp_alpha, mlp_max_iter=mlp_max_iter,
        epochs_by_seed={str(seed): int(np.median(
            [r["epochs"] for r in epoch_records if r["seed"] == seed])) for seed in seeds},
        epoch_selection=epoch_records, repeat_metrics=repeat_metrics,
        cv=dict(n_splits=n_splits, n_repeats=repeats, random_state=split_seed),
        training_ids=list(ids), classes=classes.tolist(),
        input_sha256=hashlib.sha256(
            X_depth.tobytes() + X_classical.tobytes() + y.tobytes()
        ).hexdigest(),
        source_sha256=hashlib.sha256(
            (Path(_bootstrap.SRC) / "libs2026" / "ae_pca_mlp.py").read_bytes()
            + (Path(_bootstrap.SRC) / "libs2026" / "autotransformer.py").read_bytes()
        ).hexdigest(),
        device=args.device,
        environment={name: importlib.metadata.version(name) for name in
                     ["torch", "numpy", "pandas", "scikit-learn"]},
    )
    (root / "recipe_ae_pca_mlp.json").write_text(json.dumps(recipe, indent=2),
                                                 encoding="utf-8")
    pd.DataFrame([row]).to_csv(root / "summary.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(root / "oof.csv", index=False)
    print(row, flush=True)
    print(f"Results and refit recipes: {root}")


def predict(args):
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    cfg = Config(raw=recipe["config"])
    if args.config:
        cfg.raw["paths"] = Config.load(args.config).raw["paths"]
    source_hash = hashlib.sha256(
        (Path(_bootstrap.SRC) / "libs2026" / "ae_pca_mlp.py").read_bytes()
        + (Path(_bootstrap.SRC) / "libs2026" / "autotransformer.py").read_bytes()
    ).hexdigest()
    if source_hash != recipe["source_sha256"]:
        raise ValueError("Model source changed since benchmarking; generate a new recipe")
    # Ensure classical n_groups comes from the recipe even if live config differs.
    cfg.raw.setdefault("ae_pca_mlp", {})
    cfg.raw["ae_pca_mlp"].update(recipe.get("classical", {}))
    X_depth, train = build_depth_sequences(cfg, **recipe["preparation"])
    test_depth, test = build_depth_sequences(cfg, split="test", **recipe["preparation"])
    classical_features, _ = _classical_matrix(cfg, n_jobs=8)
    train_c = classical_features.subset("train")
    test_c = classical_features.subset("test")
    train_ids = train["sample_id"].to_numpy()
    test_ids = test["sample_id"].to_numpy()
    row_train = row_sample_indices(train_ids, train_c.sample_ids)
    row_test = row_sample_indices(test_ids, test_c.sample_ids)
    y = train["label"].to_numpy(dtype=int)
    digest = hashlib.sha256(
        X_depth.tobytes() + train_c.X.tobytes() + y.tobytes()
    ).hexdigest()
    if list(train_ids) != recipe["training_ids"] or digest != recipe["input_sha256"]:
        raise ValueError("Training data/preprocessing changed since benchmarking")
    ae_params = recipe["ae_params"].copy()
    ae_params["val_fraction"] = 0
    ae_params["epochs_by_seed"] = recipe["epochs_by_seed"]
    model = AEPCAClassifier(
        ae_params=ae_params, seeds=recipe["seeds"], n_pca=recipe["n_pca"],
        mlp_hidden=tuple(recipe["mlp_hidden"]), mlp_alpha=recipe["mlp_alpha"],
        mlp_max_iter=recipe["mlp_max_iter"], random_state=recipe["seeds"][0],
        device=args.device,
    )
    model.fit(X_depth, train_c.X, train_c.y.astype(int), row_sample_idx=row_train)
    row_p = model.predict_proba(test_depth, test_c.X, row_sample_idx=row_test)
    p = _aggregate_sample_proba(row_p, row_test, len(test_ids), len(recipe["classes"]))
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
        raise FileExistsError("Choose a new output path; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=model, recipe=recipe, probabilities=p,
                     sample_ids=test["sample_id"].tolist()), checkpoint)
    submission.to_csv(out, index=False)
    print(f"Submission: {out}; checkpoint: {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--config")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--seeds", nargs="+", type=int)
    bench.add_argument("--folds", type=int)
    bench.add_argument("--repeats", type=int)
    bench.add_argument("--epochs", type=int)
    bench.add_argument("--n-jobs", type=int, default=8)
    bench.add_argument("--tag", default="g4_broadcast")
    pred = sub.add_parser("predict")
    pred.add_argument("--config")
    pred.add_argument("--device", default="cpu")
    pred.add_argument("--recipe", required=True)
    pred.add_argument("--output", default="submissions/ae_pca_mlp.csv")
    args = parser.parse_args()
    if args.command == "benchmark":
        if Path(args.tag).name != args.tag or args.tag in {".", ".."}:
            parser.error("tag must be a directory name, not a path")
        benchmark(Config.load(args.config), args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
