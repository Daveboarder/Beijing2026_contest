"""GPU hyperparameter sweep for the depth-spectrum CNN.

    uv run --extra cnn python scripts/11_tune_cnn.py --device cuda

Evaluates a diverse hand-picked set of configs with 1 CV repeat, then
re-runs the winner with 2 repeats (same protocol as ``09_benchmark_cnn.py``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import _bootstrap  # noqa: F401
import pandas as pd

from libs2026 import Config, Preprocessor, cross_validate_model, plotting, summarize
from libs2026.cnn import SpectrumCNN
from libs2026.images import build_images

# Configs around the proven depth-aware baseline (~0.55–0.58).
TRIALS = [
    # shot_bin, spectral_pca, channels, lr, mixup, dropout
    (4, 48, "32,64,128", 1e-3, 0.0, 0.4),
    (4, 48, "32,64,128", 1e-3, 0.0, 0.35),
    (4, 48, "48,96,192", 1e-3, 0.0, 0.4),
    (4, 48, "48,96,192", 1e-3, 0.0, 0.35),
    (4, 64, "32,64,128", 1e-3, 0.0, 0.4),
    (4, 32, "32,64,128", 1e-3, 0.0, 0.4),
    (4, 48, "32,64,128", 1e-3, 0.1, 0.4),
    (4, 48, "32,64,128", 7e-4, 0.0, 0.4),
    (2, 48, "32,64,128", 1e-3, 0.0, 0.4),
    (4, 48, "24,48,96", 1e-3, 0.0, 0.35),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bin-factor", type=int, default=8)
    parser.add_argument("--norm-reference", default="shot", choices=["shot", "bulk"])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--final-epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    base = Preprocessor.from_config(cfg)
    pre = replace(base, normalization_reference=args.norm_reference)
    seed = cfg["cv"]["random_state"]

    image_cache = {}
    rows = []
    print(f"sweeping {len(TRIALS)} configs on {args.device}")

    for i, (shot_bin, pca, ch_str, lr, mix, drop) in enumerate(TRIALS, 1):
        if shot_bin not in image_cache:
            image_cache[shot_bin] = build_images(
                cfg, pre, bin_factor=args.bin_factor, shot_bin=shot_bin, n_jobs=args.n_jobs,
            )
        train = image_cache[shot_bin].subset("train")
        n_shots, n_wl = train.image_shape
        channels = tuple(int(c) for c in ch_str.split(","))

        model = SpectrumCNN(
            n_shots=n_shots, n_wavelengths=n_wl, spectral_pca=pca, channels=channels,
            dropout=drop, epochs=args.epochs, batch_size=args.batch_size, lr=lr,
            mixup_alpha=mix, device=args.device, random_state=seed, verbose=0,
        )
        result = cross_validate_model(
            model, train.as_flat(), train.y.astype(int), train.groups,
            train.sample_ids, name="cnn2d", n_splits=args.n_splits, n_repeats=1,
            random_state=seed,
        )
        rows.append({
            "shot_bin": shot_bin, "spectral_pca": pca, "channels": ch_str,
            "lr": lr, "mixup_alpha": mix, "dropout": drop,
            "bin_factor": args.bin_factor, "n_shots": n_shots,
            "accuracy": result.accuracy, "accuracy_std": result.accuracy_std,
            "balanced_accuracy": result.balanced_accuracy, "macro_f1": result.macro_f1,
            "fit_seconds": result.fit_seconds,
        })
        print(f"[{i:02d}/{len(TRIALS)}] shot_bin={shot_bin} pca={pca} ch={ch_str} "
              f"lr={lr} mix={mix} drop={drop}  "
              f"acc={result.accuracy:.3f}  ({result.fit_seconds:.0f}s)")

    board = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    sweep_path = cfg.metrics_dir / "cnn_sweep_cuda.csv"
    board.to_csv(sweep_path, index=False)
    print("\ntop 8")
    print(board.head(8).to_string(index=False))
    print(f"\nsweep -> {sweep_path}")

    best = board.iloc[0]
    shot_bin = int(best["shot_bin"])
    train = image_cache[shot_bin].subset("train")
    n_shots, n_wl = train.image_shape
    channels = tuple(int(c) for c in str(best["channels"]).split(","))

    print(f"\nre-evaluating winner with {args.final_epochs} epochs x 2 repeats")
    winner = SpectrumCNN(
        n_shots=n_shots, n_wavelengths=n_wl,
        spectral_pca=int(best["spectral_pca"]), channels=channels,
        dropout=float(best["dropout"]), epochs=args.final_epochs,
        batch_size=args.batch_size, lr=float(best["lr"]),
        mixup_alpha=float(best["mixup_alpha"]), device=args.device,
        random_state=seed, verbose=0,
    )
    final = cross_validate_model(
        winner, train.as_flat(), train.y.astype(int), train.groups,
        train.sample_ids, name="cnn2d", n_splits=args.n_splits, n_repeats=2,
        random_state=seed,
    )
    print(f"cnn2d (tuned)    acc={final.accuracy:.3f}+-{final.accuracy_std:.3f}  "
          f"bal_acc={final.balanced_accuracy:.3f}  macroF1={final.macro_f1:.3f}  "
          f"({final.fit_seconds:.1f}s)")

    tag = "cnn2d_cuda_tuned"
    summary = summarize([final])
    summary.insert(1, "bin_factor", args.bin_factor)
    summary.insert(2, "shot_bin", shot_bin)
    summary.insert(3, "spectral_pca", int(best["spectral_pca"]))
    out_csv = cfg.metrics_dir / f"benchmark_{tag}.csv"
    summary.to_csv(out_csv, index=False)
    final.oof.to_csv(cfg.predictions_dir / f"oof_{tag}_cnn2d.csv", index=False)
    plotting.plot_confusion(
        final.confusion, sorted(set(train.y.astype(int))),
        cfg.figures_dir / f"confusion_{tag}_cnn2d.png",
        title=f"cnn2d tuned on CUDA (acc {final.accuracy:.3f})",
    )
    with open(cfg.models_dir / "best_params_cnn2d.json", "w", encoding="utf-8") as fh:
        json.dump({
            "model": "cnn2d",
            "cv_accuracy": float(final.accuracy),
            "cv_accuracy_std": float(final.accuracy_std),
            "balanced_accuracy": float(final.balanced_accuracy),
            "params": {
                "bin_factor": args.bin_factor,
                "shot_bin": shot_bin,
                "spectral_pca": int(best["spectral_pca"]),
                "channels": list(channels),
                "dropout": float(best["dropout"]),
                "lr": float(best["lr"]),
                "mixup_alpha": float(best["mixup_alpha"]),
                "epochs": args.final_epochs,
                "batch_size": args.batch_size,
                "norm_reference": args.norm_reference,
            },
        }, fh, indent=2)

    print(f"benchmark -> {out_csv}")
    print(f"best params -> {cfg.models_dir / 'best_params_cnn2d.json'}")


if __name__ == "__main__":
    main()
