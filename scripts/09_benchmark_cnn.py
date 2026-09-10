"""Cross-validate the 2-D CNN that sees each sample as a depth-spectrum image.

    uv run --extra cnn python scripts/09_benchmark_cnn.py --device cuda

Each sample is shaped ``(n_shots, n_wavelengths)`` — analogous to a single-channel
image — so the network can learn patterns that couple wavelength and depth.

Single-seed CV is noisy with N=120; pass ``--ensemble-seeds 42,7,123`` to average
out-of-fold probabilities across seeds (typically lifts ~0.43 → ~0.55+).
"""

import argparse
import json
from dataclasses import replace

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from libs2026 import Config, Preprocessor, cross_validate_model, summarize
from libs2026 import plotting
from libs2026.cnn import SpectrumCNN
from libs2026.images import build_images


def _ensemble_oof(model_kw, X, y, groups, sample_ids, seeds, n_splits, n_repeats):
    classes = np.unique(y)
    oof_sum = None
    seed_accs = []
    for seed in seeds:
        proba = np.zeros((len(y), len(classes)))
        for repeat in range(n_repeats):
            fold_proba = np.zeros_like(proba)
            splitter = StratifiedGroupKFold(
                n_splits, shuffle=True, random_state=int(seed) + repeat,
            )
            for tr, te in splitter.split(X, y, groups):
                m = SpectrumCNN(**{**model_kw, "random_state": int(seed)})
                m.fit(X[tr], y[tr])
                fold_proba[te] = m.predict_proba(X[te])
            proba += fold_proba
        proba /= n_repeats
        seed_accs.append(float(accuracy_score(y, classes[proba.argmax(1)])))
        oof_sum = proba if oof_sum is None else oof_sum + proba
    oof = oof_sum / len(seeds)
    pred = classes[oof.argmax(1)]
    return {
        "proba": oof,
        "pred": pred,
        "classes": classes,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "seed_accuracies": seed_accs,
        "confusion": confusion_matrix(y, pred, labels=classes),
        "oof_frame": pd.DataFrame({
            "sample_id": sample_ids,
            "y_true": y,
            "y_pred": pred,
            **{f"p{int(c)}": oof[:, i] for i, c in enumerate(classes)},
        }),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bin-factor", type=int, default=8,
                        help="average neighbouring wavelength pixels (default 8)")
    parser.add_argument("--shot-bin", type=int, default=4,
                        help="average consecutive shots (default 4 → 50-row images)")
    parser.add_argument("--n-shots", type=int, default=None,
                        help="keep only the first N pulses (surface layer); "
                             "default = all recorded shots")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--spectral-pca", type=int, default=48,
                        help="project each shot onto this many PCA components (0=off)")
    parser.add_argument("--channels", default="32,64,128",
                        help="comma-separated conv channel widths")
    parser.add_argument("--norm-reference", default="shot", choices=["shot", "bulk"],
                        help="intensity normalisation reference for the images")
    parser.add_argument("--ensemble-seeds", default="42,7,123,99,2026",
                        help="comma-separated seeds to average (empty = single-seed CV)")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv_cfg = cfg["cv"]

    channels = tuple(int(c) for c in args.channels.split(",") if c.strip())
    seeds = [int(s) for s in args.ensemble_seeds.split(",") if s.strip()]
    base = Preprocessor.from_config(cfg)
    pre = replace(base, normalization_reference=args.norm_reference)

    images = build_images(cfg, pre, bin_factor=args.bin_factor,
                          shot_bin=args.shot_bin, n_shots=args.n_shots,
                          n_jobs=args.n_jobs)
    train = images.subset("train")
    n_shots, n_wl = train.image_shape
    y = train.y.astype(int)
    X = train.as_flat()

    print(f"images: {train.X.shape}  (shots x wavelengths = {n_shots} x {n_wl})")
    print(f"device: {args.device or 'auto'}  channels={channels}  epochs={args.epochs}")

    model_kw = dict(
        n_shots=n_shots, n_wavelengths=n_wl, spectral_pca=args.spectral_pca,
        channels=channels, dropout=args.dropout, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=args.device, verbose=args.verbose,
    )
    n_repeats = min(cv_cfg["n_repeats"], 2)
    tag = args.tag or f"cnn2d_b{args.bin_factor}_s{args.shot_bin}"

    if seeds:
        print(f"ensemble seeds: {seeds}  repeats={n_repeats}")
        ens = _ensemble_oof(
            model_kw, X, y, train.groups, train.sample_ids,
            seeds, cv_cfg["n_splits"], n_repeats,
        )
        print(
            f"cnn2d ensemble   acc={ens['accuracy']:.3f}  "
            f"bal_acc={ens['balanced_accuracy']:.3f}  "
            f"macroF1={ens['macro_f1']:.3f}  "
            f"seeds={[round(a, 3) for a in ens['seed_accuracies']]}"
        )
        summary = pd.DataFrame([{
            "model": "cnn2d_ensemble",
            "bin_factor": args.bin_factor,
            "n_shots": n_shots,
            "n_wavelengths": n_wl,
            "accuracy": ens["accuracy"],
            "accuracy_std": float(np.std(ens["seed_accuracies"])),
            "balanced_accuracy": ens["balanced_accuracy"],
            "macro_f1": ens["macro_f1"],
            "seed_mean": float(np.mean(ens["seed_accuracies"])),
        }])
        out_csv = cfg.metrics_dir / f"benchmark_{tag}.csv"
        summary.to_csv(out_csv, index=False)
        ens["oof_frame"].to_csv(cfg.predictions_dir / f"oof_{tag}_cnn2d.csv", index=False)
        plotting.plot_confusion(
            ens["confusion"], ens["classes"],
            cfg.figures_dir / f"confusion_{tag}_cnn2d.png",
            title=f"cnn2d ensemble: out-of-fold confusion ({tag})",
        )
        payload = {
            "model": "cnn2d",
            "cv_accuracy": ens["accuracy"],
            "balanced_accuracy": ens["balanced_accuracy"],
            "macro_f1": ens["macro_f1"],
            "seed_accuracies": ens["seed_accuracies"],
            "params": {
                "bin_factor": args.bin_factor, "shot_bin": args.shot_bin,
                "n_shots": args.n_shots,
                "norm_reference": args.norm_reference,
                "spectral_pca": args.spectral_pca, "channels": list(channels),
                "dropout": args.dropout, "epochs": args.epochs,
                "batch_size": args.batch_size, "lr": args.lr,
                "ensemble_seeds": seeds,
            },
        }
        with open(cfg.models_dir / f"best_params_{tag}.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        official = cfg.models_dir / "best_params_cnn2d.json"
        previous = {}
        if official.exists():
            with open(official, encoding="utf-8") as fh:
                previous = json.load(fh)
        if previous.get("cv_accuracy", -1) > payload["cv_accuracy"]:
            print(f"keeping existing {official} "
                  f"(acc {previous['cv_accuracy']:.3f} > {payload['cv_accuracy']:.3f})")
        else:
            with open(official, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
    else:
        model = SpectrumCNN(**model_kw, random_state=cv_cfg["random_state"])
        result = cross_validate_model(
            model, X, y, train.groups, train.sample_ids, name="cnn2d",
            n_splits=cv_cfg["n_splits"], n_repeats=n_repeats,
            random_state=cv_cfg["random_state"],
        )
        print(f"cnn2d            acc={result.accuracy:.3f}+-{result.accuracy_std:.3f}  "
              f"bal_acc={result.balanced_accuracy:.3f}  "
              f"macroF1={result.macro_f1:.3f}  ({result.fit_seconds:.1f}s)")
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
            "preprocessing": {**cfg["preprocessing"], "normalization_reference": args.norm_reference},
            "cv": {**cv_cfg, "n_repeats": n_repeats},
            "bin_factor": args.bin_factor,
            "shot_bin": args.shot_bin,
            "n_shots": args.n_shots,
            "image_shape": [n_shots, n_wl],
            "channels": list(channels),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dropout": args.dropout,
            "spectral_pca": args.spectral_pca,
            "ensemble_seeds": seeds,
        }, fh, indent=2)

    print(f"\nleaderboard -> {out_csv}")


if __name__ == "__main__":
    main()
