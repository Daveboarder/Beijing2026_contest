"""Fit the 2-D CNN on all training images and write a contest submission.

    uv run --extra cnn python scripts/10_predict_cnn.py --device cuda

Loads defaults from ``results/models/best_params_cnn2d.json`` when present.
Averages predictions across ``--ensemble-seeds`` (recommended for N=120).
"""

import argparse
import json
from datetime import datetime

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from dataclasses import replace

from libs2026 import Config, Preprocessor
from libs2026.cnn import SpectrumCNN
from libs2026.images import build_images


def validate_submission(frame: pd.DataFrame, cfg: Config) -> None:
    template = pd.read_csv(cfg.sample_submission)
    assert list(frame.columns) == ["filename", "predicted_label"], frame.columns
    assert len(frame) == len(template)
    assert not frame["filename"].duplicated().any()
    assert set(frame["filename"]) == set(template["filename"])
    assert frame["predicted_label"].between(1, 5).all()
    assert frame["predicted_label"].dtype.kind in "iu"


def _load_best(cfg: Config) -> dict:
    path = cfg.models_dir / "best_params_cnn2d.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    return blob.get("params", {})


def main() -> None:
    best = {}
    # Config needed early only for default path; parse once after.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bin-factor", type=int, default=None)
    parser.add_argument("--shot-bin", type=int, default=None)
    parser.add_argument("--n-shots", type=int, default=None,
                        help="keep only the first N pulses; 0 = all shots")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--spectral-pca", type=int, default=None)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--norm-reference", default=None, choices=["shot", "bulk"])
    parser.add_argument("--ensemble-seeds", default=None,
                        help="comma-separated seeds (empty string = single model)")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default=None)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    best = _load_best(cfg)
    cnn_cfg = cfg.get("cnn", {}) if hasattr(cfg, "get") else cfg["cnn"]

    def pick(name, cast=lambda x: x, default=None):
        cli = getattr(args, name.replace("-", "_"))
        if cli is not None and cli != "":
            return cast(cli) if not isinstance(cli, (list, tuple)) else cli
        if name in best:
            return best[name]
        if name in cnn_cfg:
            return cnn_cfg[name]
        return default

    bin_factor = int(pick("bin_factor", default=8))
    shot_bin = int(pick("shot_bin", default=4))
    n_shots_raw = args.n_shots if args.n_shots is not None else best.get("n_shots")
    n_shots = int(n_shots_raw) if n_shots_raw else None
    epochs = int(pick("epochs", default=180))
    batch_size = int(pick("batch_size", default=16))
    lr = float(pick("lr", default=1e-3))
    dropout = float(pick("dropout", default=0.35))
    spectral_pca = int(pick("spectral_pca", default=48))
    norm_reference = pick("norm_reference", default="shot")
    channels_raw = pick("channels", default=[32, 64, 128])
    if isinstance(channels_raw, str):
        channels = tuple(int(c) for c in channels_raw.split(",") if c.strip())
    else:
        channels = tuple(int(c) for c in channels_raw)

    seeds_raw = args.ensemble_seeds
    if seeds_raw is None:
        seeds = list(best.get("ensemble_seeds", cnn_cfg.get("ensemble_seeds", [42, 7, 123])))
    elif seeds_raw.strip() == "":
        seeds = [cfg["cv"]["random_state"]]
    else:
        seeds = [int(s) for s in seeds_raw.split(",") if s.strip()]

    pre = replace(Preprocessor.from_config(cfg), normalization_reference=norm_reference)
    images = build_images(cfg, pre, bin_factor=bin_factor, shot_bin=shot_bin,
                          n_shots=n_shots, n_jobs=args.n_jobs)
    train, test = images.subset("train"), images.subset("test")
    n_shots, n_wl = train.image_shape
    Xtr, ytr = train.as_flat(), train.y.astype(int)
    Xte = test.as_flat()

    print(f"fitting {len(seeds)} seed(s) on {train.X.shape} → predict {test.X.shape}")
    proba = None
    models = []
    classes = None
    for seed in seeds:
        model = SpectrumCNN(
            n_shots=n_shots, n_wavelengths=n_wl, spectral_pca=spectral_pca,
            channels=channels, dropout=dropout, epochs=epochs,
            batch_size=batch_size, lr=lr, device=args.device,
            verbose=args.verbose, random_state=int(seed),
        )
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)
        proba = p if proba is None else proba + p
        models.append(model)
        classes = model.classes_
    proba /= len(seeds)
    pred = classes[proba.argmax(axis=1)]

    submission = pd.DataFrame({
        "filename": test.sample_ids + ".csv",
        "predicted_label": pred.astype(int),
    }).sort_values("filename").reset_index(drop=True)
    validate_submission(submission, cfg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = args.name or f"predictions_cnn2d_{stamp}.csv"
    out = cfg.submissions_dir / name
    submission.to_csv(out, index=False)
    joblib.dump(
        {
            "models": models, "seeds": seeds, "bin_factor": bin_factor,
            "shot_bin": shot_bin, "n_shots": n_shots, "image_shape": (n_shots, n_wl),
            "norm_reference": norm_reference,
        },
        cfg.models_dir / f"fitted_cnn2d_{stamp}.joblib",
    )

    print("predicted class distribution:")
    print(submission["predicted_label"].value_counts().sort_index().to_string())
    print(f"mean top-class probability: {proba.max(axis=1).mean():.3f}")
    print(f"submission -> {out}")


if __name__ == "__main__":
    main()
