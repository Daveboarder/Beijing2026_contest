"""Cross-validate the multi-scale CNN encoder + depth transformer.

    uv run --extra cnn python scripts/15_benchmark_transformer.py --device cuda

Each shot is tokenised on its own by three parallel 1-D convolutions (kernel
widths 3, 7 and 15 over the wavelength axis) into a single ``d_model``
embedding, and the 50 resulting tokens — one per depth step — are fed to a
transformer encoder. Attention can therefore relate the aged surface to the
bulk plateau at any separation, which the fixed receptive field of the 2-D
pixel CNN cannot.

The protocol is identical to ``09_benchmark_cnn.py`` so the numbers are
comparable: grouped 5-fold, two repeats, out-of-fold probabilities averaged
over ``--ensemble-seeds``, and a blend grid against the classical ``pca_mlp``
out-of-fold file.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

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

from libs2026 import Config, Preprocessor, plotting
from libs2026.cnn import SpectrumTransformer
from libs2026.images import build_images

PIXEL_CNN_ACC = 0.575
TOKEN_CNN_ACC = 0.542
CLASSICAL_ACC = 0.715


def ensemble_oof(model_kw, X, y, groups, seeds, n_splits, n_repeats, verbose=True):
    """Seed-averaged out-of-fold probabilities over grouped folds."""
    classes = np.unique(y)
    oof_sum, seed_accs = None, []
    for seed in seeds:
        proba = np.zeros((len(y), len(classes)))
        for repeat in range(n_repeats):
            fold = np.zeros_like(proba)
            splitter = StratifiedGroupKFold(
                n_splits, shuffle=True, random_state=int(seed) + repeat,
            )
            for tr, te in splitter.split(X, y, groups):
                model = SpectrumTransformer(**{**model_kw, "random_state": int(seed)})
                model.fit(X[tr], y[tr])
                fold[te] = model.predict_proba(X[te])
            proba += fold
        proba /= n_repeats
        acc = float(accuracy_score(y, classes[proba.argmax(1)]))
        seed_accs.append(acc)
        if verbose:
            print(f"    seed {seed:<5} acc={acc:.3f}", flush=True)
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
    }


def _repeat_probas(frame: pd.DataFrame, sample_ids, classes) -> list[np.ndarray] | None:
    """One probability matrix per CV repeat, aligned to ``sample_ids``."""
    cols = [f"p{int(c)}" for c in classes]
    if any(c not in frame.columns for c in cols) or "sample_id" not in frame.columns:
        return None
    parts = ([frame] if "repeat" not in frame.columns
             else [part for _, part in frame.groupby("repeat", sort=True)])
    out = []
    for part in parts:
        grouped = part.groupby("sample_id", sort=False)[cols].mean()
        if any(s not in grouped.index for s in sample_ids):
            return None
        out.append(grouped.loc[list(sample_ids), cols].to_numpy(dtype=float))
    return out or None


def blend_grid(p_a, partners, y, classes, name_a, name_b):
    """Linear mix of two probability matrices, scanned over the mixing weight.

    The partner is a *list* of per-repeat matrices, so the blend is scored the
    same way the classical leaderboard is (mean over repeats) rather than on
    one optimistically averaged matrix.
    """
    rows, best = [], None
    for w in np.linspace(0.0, 1.0, 11):
        accs, bals, f1s, last = [], [], [], None
        for other in partners:
            proba = w * p_a + (1.0 - w) * other
            pred = classes[proba.argmax(axis=1)]
            accs.append(accuracy_score(y, pred))
            bals.append(balanced_accuracy_score(y, pred))
            f1s.append(f1_score(y, pred, average="macro"))
            last = pred
        row = {
            "pair": f"{name_a}+{name_b}",
            "w_a": float(w),
            "accuracy": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "balanced_accuracy": float(np.mean(bals)),
            "macro_f1": float(np.mean(f1s)),
        }
        rows.append(row)
        if best is None or row["accuracy"] > best["accuracy"]:
            best = {**row, "pred": last}
    return pd.DataFrame(rows), best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bin-factor", type=int, default=8,
                        help="average neighbouring wavelength pixels (default 8)")
    parser.add_argument("--shot-bin", type=int, default=4,
                        help="average consecutive shots (default 4 -> 50 depth tokens)")
    parser.add_argument("--n-shots", type=int, default=None,
                        help="keep only the first N pulses; default = all shots")
    parser.add_argument("--kernel-sizes", default="3,7,15",
                        help="parallel convolution widths of the tokeniser")
    parser.add_argument("--conv-channels", type=int, default=32,
                        help="channels per convolution branch")
    parser.add_argument("--d-model", type=int, default=128,
                        help="per-shot embedding width (transformer d_model)")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="transformer dropout; >0 collapses at N=120")
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--wl-shift", type=int, default=0)
    parser.add_argument("--shot-mask-p", type=float, default=0.0)
    parser.add_argument("--norm-reference", default="shot", choices=["shot", "bulk"])
    parser.add_argument("--ensemble-seeds", default="42,7,123,99,2026")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--classical-oof", default="results/predictions/oof_g4_b1_pca_mlp.csv")
    parser.add_argument("--pixel-oof", default="results/predictions/oof_cnn2d_pca48_cnn2d.csv")
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="transformer")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()

    kernel_sizes = tuple(int(k) for k in args.kernel_sizes.split(",") if k.strip())
    seeds = [int(s) for s in args.ensemble_seeds.split(",") if s.strip()]
    pre = replace(Preprocessor.from_config(cfg),
                  normalization_reference=args.norm_reference)

    images = build_images(cfg, pre, bin_factor=args.bin_factor,
                          shot_bin=args.shot_bin, n_shots=args.n_shots,
                          n_jobs=args.n_jobs)
    train = images.subset("train")
    n_shots, n_wl = train.image_shape
    y = train.y.astype(int)
    X = train.as_flat()

    print(f"images: {train.X.shape}  ({n_shots} depth tokens x {n_wl} wavelengths)")
    print(f"tokeniser: kernels {kernel_sizes} x {args.conv_channels} ch -> "
          f"d_model {args.d_model}")
    print(f"depth model: {args.n_layers} layers, {args.n_heads} heads, "
          f"ff x{args.ff_mult}, dropout {args.dropout}")

    model_kw = dict(
        n_shots=n_shots, n_wavelengths=n_wl,
        kernel_sizes=kernel_sizes, conv_channels=args.conv_channels,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        ff_mult=args.ff_mult, dropout=args.dropout,
        encoder_dropout=args.encoder_dropout,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, noise_std=args.noise_std,
        wl_shift=args.wl_shift, shot_mask_p=args.shot_mask_p,
        device=args.device, verbose=args.verbose,
    )

    print(f"ensemble seeds: {seeds}  repeats={args.n_repeats}", flush=True)
    ens = ensemble_oof(model_kw, X, y, train.groups, seeds,
                       args.n_splits, args.n_repeats)
    print(f"  transformer   acc={ens['accuracy']:.3f}  "
          f"bal_acc={ens['balanced_accuracy']:.3f}  "
          f"macroF1={ens['macro_f1']:.3f}  "
          f"(pixel CNN {PIXEL_CNN_ACC:.3f}, token CNN {TOKEN_CNN_ACC:.3f}, "
          f"classical {CLASSICAL_ACC:.3f})")

    rows = [{
        "model": "transformer",
        "bin_factor": args.bin_factor,
        "shot_bin": args.shot_bin,
        "n_shots": args.n_shots or int(cfg["data"].get("n_shots", 200)),
        "n_tokens": n_shots,
        "n_wavelengths": n_wl,
        "kernel_sizes": args.kernel_sizes,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "accuracy": ens["accuracy"],
        "accuracy_std": float(np.std(ens["seed_accuracies"])),
        "balanced_accuracy": ens["balanced_accuracy"],
        "macro_f1": ens["macro_f1"],
        "seed_mean": float(np.mean(ens["seed_accuracies"])),
    }]

    blend_rows, blend_best = [], None
    for label, path in (("pca_mlp", Path(args.classical_oof)),
                        ("pixel_cnn", Path(args.pixel_oof))):
        if not path.exists():
            print(f"blend skip {label}: {path} not found")
            continue
        repeats = _repeat_probas(pd.read_csv(path), train.sample_ids, ens["classes"])
        if repeats is None:
            print(f"blend skip {label}: could not align {path}")
            continue
        partner_acc = float(np.mean([
            accuracy_score(y, ens["classes"][p.argmax(1)]) for p in repeats
        ]))
        grid, winner = blend_grid(ens["proba"], repeats, y, ens["classes"],
                                  "transformer", label)
        print(f"\nblend vs {label} (partner OOF acc={partner_acc:.3f}):")
        print(grid[["w_a", "accuracy", "balanced_accuracy", "macro_f1"]].to_string(index=False))
        print(f"  best w_transformer={winner['w_a']:.1f}  acc={winner['accuracy']:.3f}")
        blend_rows.append(grid)
        if blend_best is None or winner["accuracy"] > blend_best["accuracy"]:
            blend_best = {**winner, "partner": label, "partner_acc": partner_acc}

    board = pd.DataFrame(rows)
    out_csv = cfg.metrics_dir / f"benchmark_{args.tag}.csv"
    board.to_csv(out_csv, index=False)
    print("\n" + board.to_string(index=False))

    if blend_rows:
        blend_csv = cfg.metrics_dir / f"blend_{args.tag}.csv"
        pd.concat(blend_rows, ignore_index=True).to_csv(blend_csv, index=False)
        print(f"blend grid -> {blend_csv}")

    pd.DataFrame({
        "sample_id": train.sample_ids, "y_true": y, "y_pred": ens["pred"],
        **{f"p{int(c)}": ens["proba"][:, i] for i, c in enumerate(ens["classes"])},
    }).to_csv(cfg.predictions_dir / f"oof_{args.tag}.csv", index=False)

    plotting.plot_confusion(
        ens["confusion"], ens["classes"],
        cfg.figures_dir / f"confusion_{args.tag}.png",
        title=f"multi-scale CNN + depth transformer, acc {ens['accuracy']:.3f}",
    )

    payload = {
        "model": "transformer",
        "cv_accuracy": ens["accuracy"],
        "balanced_accuracy": ens["balanced_accuracy"],
        "macro_f1": ens["macro_f1"],
        "seed_accuracies": ens["seed_accuracies"],
        "pixel_cnn_cv": PIXEL_CNN_ACC,
        "token_cnn_cv": TOKEN_CNN_ACC,
        "classical_pca_mlp_oof": None if blend_best is None else blend_best["partner_acc"],
        "blend_accuracy": None if blend_best is None else blend_best["accuracy"],
        "blend_w_transformer": None if blend_best is None else blend_best["w_a"],
        "blend_partner": None if blend_best is None else blend_best["partner"],
        "params": {
            "bin_factor": args.bin_factor, "shot_bin": args.shot_bin,
            "n_shots": args.n_shots, "norm_reference": args.norm_reference,
            "kernel_sizes": list(kernel_sizes),
            "conv_channels": args.conv_channels, "d_model": args.d_model,
            "n_layers": args.n_layers, "n_heads": args.n_heads,
            "ff_mult": args.ff_mult, "dropout": args.dropout,
            "encoder_dropout": args.encoder_dropout,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "noise_std": args.noise_std, "wl_shift": args.wl_shift,
            "shot_mask_p": args.shot_mask_p,
            "ensemble_seeds": seeds,
        },
    }
    tagged = cfg.models_dir / f"best_params_{args.tag}.json"
    with open(tagged, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    official = cfg.models_dir / "best_params_transformer.json"
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

    print(f"\nleaderboard -> {out_csv}")
    print(f"transformer {ens['accuracy']:.3f}  vs pixel CNN {PIXEL_CNN_ACC:.3f}  "
          f"vs token CNN {TOKEN_CNN_ACC:.3f}  vs classical {CLASSICAL_ACC:.3f}")


if __name__ == "__main__":
    main()
