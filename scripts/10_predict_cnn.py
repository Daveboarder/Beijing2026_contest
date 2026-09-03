"""Fit the 2-D CNN on all training images and write a contest submission.

    uv run --extra cnn python scripts/10_predict_cnn.py --bin-factor 8
"""

import argparse
from datetime import datetime

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bin-factor", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--channels", default="16,32,64,128")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    channels = tuple(int(c) for c in args.channels.split(",") if c.strip())

    images = build_images(cfg, Preprocessor.from_config(cfg),
                          bin_factor=args.bin_factor, n_jobs=args.n_jobs)
    train, test = images.subset("train"), images.subset("test")
    n_shots, n_wl = train.image_shape

    model = SpectrumCNN(
        n_shots=n_shots, n_wavelengths=n_wl, channels=channels,
        dropout=args.dropout, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device, verbose=args.verbose,
        random_state=cfg["cv"]["random_state"],
    )
    model.fit(train.as_flat(), train.y.astype(int))
    proba = model.predict_proba(test.as_flat())
    pred = model.classes_[proba.argmax(axis=1)]

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
        {"model": model, "bin_factor": args.bin_factor, "image_shape": (n_shots, n_wl)},
        cfg.models_dir / f"fitted_cnn2d_{stamp}.joblib",
    )

    print("predicted class distribution:")
    print(submission["predicted_label"].value_counts().sort_index().to_string())
    print(f"mean top-class probability: {proba.max(axis=1).mean():.3f}")
    print(f"submission -> {out}")


if __name__ == "__main__":
    main()
