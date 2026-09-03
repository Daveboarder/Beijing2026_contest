"""Cross-validate the 2-D CNN that sees each sample as a depth-spectrum image.

    uv run --extra cnn python scripts/09_benchmark_cnn.py --bin-factor 8

Each sample is shaped ``(n_shots, n_wavelengths)`` — analogous to a single-channel
image — so the network can learn patterns that couple wavelength and depth.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from libs2026 import Config, Preprocessor, cross_validate_model, summarize
from libs2026 import plotting
from libs2026.cnn import SpectrumCNN
from libs2026.images import build_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bin-factor", type=int, default=8,
                        help="average neighbouring wavelength pixels (default 8)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--channels", default="32,64,128",
                        help="comma-separated conv channel widths")
    parser.add_argument("--norm-reference", default="shot", choices=["shot", "bulk"],
                        help="intensity normalisation reference for the images")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv_cfg = cfg["cv"]
    from dataclasses import replace

    channels = tuple(int(c) for c in args.channels.split(",") if c.strip())
    base = Preprocessor.from_config(cfg)
    pre = replace(base, normalization_reference=args.norm_reference)

    images = build_images(cfg, pre, bin_factor=args.bin_factor, n_jobs=args.n_jobs)
    train = images.subset("train")
    n_shots, n_wl = train.image_shape
    y = train.y.astype(int)
    X = train.as_flat()

    print(f"images: {train.X.shape}  (shots x wavelengths = {n_shots} x {n_wl})")
    print(f"device: {args.device or 'auto'}  channels={channels}  epochs={args.epochs}")

    model = SpectrumCNN(
        n_shots=n_shots,
        n_wavelengths=n_wl,
        channels=channels,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        random_state=cv_cfg["random_state"],
        verbose=args.verbose,
    )

    # Fewer repeats than the classical zoo: each fold trains a CNN.
    n_repeats = min(cv_cfg["n_repeats"], 2)
    result = cross_validate_model(
        model, X, y, train.groups, train.sample_ids, name="cnn2d",
        n_splits=cv_cfg["n_splits"], n_repeats=n_repeats,
        random_state=cv_cfg["random_state"],
    )
    print(f"cnn2d            acc={result.accuracy:.3f}+-{result.accuracy_std:.3f}  "
          f"bal_acc={result.balanced_accuracy:.3f}  "
          f"macroF1={result.macro_f1:.3f}  ({result.fit_seconds:.1f}s)")

    tag = args.tag or f"cnn2d_b{args.bin_factor}"
    summary = summarize([result])
    summary.insert(1, "bin_factor", args.bin_factor)
    summary.insert(2, "n_shots", n_shots)
    summary.insert(3, "n_wavelengths", n_wl)
    out_csv = cfg.metrics_dir / f"benchmark_{tag}.csv"
    summary.to_csv(out_csv, index=False)
    result.oof.to_csv(cfg.predictions_dir / f"oof_{tag}_cnn2d.csv", index=False)
    plotting.plot_confusion(
        result.confusion, np.unique(y),
        cfg.figures_dir / f"confusion_{tag}_cnn2d.png",
        title=f"cnn2d: out-of-fold confusion ({tag})",
    )
    with open(cfg.metrics_dir / f"benchmark_{tag}_settings.json", "w", encoding="utf-8") as fh:
        json.dump({
            "preprocessing": cfg["preprocessing"],
            "cv": {**cv_cfg, "n_repeats": n_repeats},
            "bin_factor": args.bin_factor,
            "image_shape": [n_shots, n_wl],
            "channels": list(channels),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dropout": args.dropout,
        }, fh, indent=2)

    print(f"\nleaderboard -> {out_csv}")


if __name__ == "__main__":
    main()
