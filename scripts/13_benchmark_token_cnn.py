"""Cross-validate the 2-D CNN over spectral-line tokens.

    uv run --extra cnn python scripts/13_benchmark_token_cnn.py --device cuda

Columns are physical transitions instead of wavelength bins, so no per-fold
spectral PCA is needed. Single-seed CV is very noisy at N=120; out-of-fold
probabilities are averaged over ``--ensemble-seeds`` as in
``09_benchmark_cnn.py``, which is what makes the numbers comparable.

``--ablation`` additionally scores a reduced line set and an amplitude-only
variant, to check whether the physics channels earn their place. The blend
grid against classical ``pca_mlp`` and the pixel CNN runs whenever those OOF
files are present.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from libs2026 import Config, Preprocessor, plotting
from libs2026.cnn import TokenCNN
from libs2026.evaluation import cross_validate_model
from libs2026.models import build_model_zoo
from libs2026.tokens import (
    F_MAX_INT,
    F_VALID,
    build_tokens,
    channel_head_kwargs,
    fit_config_from_config,
    line_dictionary_from_config,
    parse_sample_channels,
)

PIXEL_CNN_ACC = 0.575
CLASSICAL_ACC = 0.715


def ensemble_oof(tokens, model_kw, seeds, n_splits=5, n_repeats=2, verbose=True):
    """Seed-averaged out-of-fold probabilities over grouped folds."""
    X = tokens.as_flat()
    y = tokens.y.astype(int)
    classes = np.unique(y)
    oof_sum, seed_accs = None, []

    for seed in seeds:
        proba = np.zeros((len(y), len(classes)))
        for repeat in range(n_repeats):
            splitter = StratifiedGroupKFold(
                n_splits, shuffle=True, random_state=int(seed) + repeat,
            )
            fold = np.zeros_like(proba)
            for train_idx, test_idx in splitter.split(X, y, tokens.groups):
                model = TokenCNN(**{**model_kw, "random_state": int(seed)})
                model.fit(X[train_idx], y[train_idx])
                fold[test_idx] = model.predict_proba(X[test_idx])
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
    }


def model_kwargs(tokens, args, extra=None):
    rows, lines, feats = tokens.token_shape
    depth_bins = min(12, max(4, rows // 3)) if rows <= 32 else 12
    kw = dict(
        static=tokens.static,
        feature_mean=tokens.feature_mean,
        feature_std=tokens.feature_std,
        n_rows=rows, n_lines=lines, n_features=feats,
        channels=tuple(int(c) for c in args.channels.split(",") if c.strip()),
        dropout=args.dropout,
        depth_bins=depth_bins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        token_dropout=args.token_dropout,
        noise_std=args.noise_std,
        include_static=not args.no_static,
        device=args.device,
        verbose=0,
    )
    if extra:
        kw.update(extra)
    return kw


def _repeat_probas(frame: pd.DataFrame, sample_ids, classes) -> list[np.ndarray] | None:
    """One probability matrix per CV repeat, aligned to ``sample_ids``."""
    cols = [f"p{int(c)}" for c in classes]
    if any(c not in frame.columns for c in cols) or "sample_id" not in frame.columns:
        return None
    if "repeat" not in frame.columns:
        grouped = frame.groupby("sample_id", sort=False)[cols].mean()
        missing = [s for s in sample_ids if s not in grouped.index]
        if missing:
            return None
        return [grouped.loc[list(sample_ids), cols].to_numpy(dtype=float)]
    out = []
    for _, part in frame.groupby("repeat", sort=True):
        grouped = part.groupby("sample_id", sort=False)[cols].mean()
        missing = [s for s in sample_ids if s not in grouped.index]
        if missing:
            return None
        out.append(grouped.loc[list(sample_ids), cols].to_numpy(dtype=float))
    return out or None


def blend_grid(p_a, p_b, y, classes, name_a="token_cnn", name_b="other"):
    """Linear mix of two probability matrices, scanned over the mixing weight."""
    rows = []
    best = None
    partners = p_b if isinstance(p_b, list) else [p_b]
    for w in np.linspace(0.0, 1.0, 11):
        accs, bals, f1s, last = [], [], [], None
        for other in partners:
            proba = w * p_a + (1.0 - w) * other
            pred = classes[proba.argmax(axis=1)]
            accs.append(accuracy_score(y, pred))
            bals.append(balanced_accuracy_score(y, pred))
            f1s.append(f1_score(y, pred, average="macro"))
            last = (proba, pred)
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
            best = {**row, "proba": last[0], "pred": last[1]}
    return pd.DataFrame(rows), best


def stack_lr(p_a, p_b, y, classes, groups, n_splits=5, n_repeats=2):
    """Out-of-fold logistic stack on concatenated probabilities."""
    X = np.concatenate([p_a, p_b], axis=1)
    y = np.asarray(y)
    proba = np.zeros((len(y), len(classes)))
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits, shuffle=True, random_state=42 + repeat,
        )
        fold = np.zeros_like(proba)
        for tr, te in splitter.split(X, y, groups):
            scaler = StandardScaler()
            xs = scaler.fit_transform(X[tr])
            clf = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=0,
            )
            clf.fit(xs, y[tr])
            fold[te] = clf.predict_proba(scaler.transform(X[te]))
        proba += fold
    proba /= n_repeats
    pred = classes[proba.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "pred": pred,
        "proba": proba,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--shot-bin", type=int, default=None)
    parser.add_argument("--n-shots", type=int, default=None,
                        help="keep only the first N pulses (surface layer); "
                             "default from config, 0 = all shots")
    parser.add_argument("--norm-reference", default="shot", choices=["shot", "bulk", "both"],
                        help="'shot' won the token-CNN sweep; 'bulk' keeps the depth "
                             "intensity profile; 'both' compares the two")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--token-dropout", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--channels", default="32,64,128")
    parser.add_argument("--ensemble-seeds", default="42,7,123,99,2026")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--ablation", action="store_true",
                        help="also score a reduced line set and amplitude-only tokens")
    parser.add_argument("--sample-channels", default=None,
                        help="comma-separated per-sample token channels to keep "
                             "(e.g. r2,delta_lambda). Default: all seven.")
    parser.add_argument("--no-static", action="store_true",
                        help="drop the 9 dictionary physics channels; use only "
                             "the selected per-sample token channels")
    parser.add_argument("--classical-oof", default="results/predictions/oof_g4_b1_pca_mlp.csv",
                        help="classical pca_mlp OOF file to blend against")
    parser.add_argument("--pixel-oof", default="results/predictions/oof_cnn2d_pca48_cnn2d.csv",
                        help="pixel-CNN OOF file to blend against")
    parser.add_argument("--n-jobs", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="token_cnn")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    tok_cfg = cfg.get("tokens", {})
    shot_bin = args.shot_bin if args.shot_bin is not None else int(tok_cfg.get("shot_bin", 4))
    n_shots = args.n_shots if args.n_shots is not None else tok_cfg.get("n_shots")
    n_shots = int(n_shots) if n_shots else None
    seeds = [int(s) for s in args.ensemble_seeds.split(",") if s.strip()]
    fit_cfg = fit_config_from_config(cfg)
    dictionary = line_dictionary_from_config(cfg, verbose=False)

    references = ["shot", "bulk"] if args.norm_reference == "both" else [args.norm_reference]
    rows, results = [], {}

    for reference in references:
        pre = replace(Preprocessor.from_config(cfg), normalization_reference=reference)
        tokens = build_tokens(cfg, dictionary, pre=pre, shot_bin=shot_bin,
                              n_shots=n_shots, fit_cfg=fit_cfg,
                              n_jobs=args.n_jobs, verbose=True)
        train = tokens.subset("train")
        extra_kw = {}
        if args.sample_channels:
            keep = parse_sample_channels(args.sample_channels)
            train = train.select_channels(keep)
            extra_kw = channel_head_kwargs(keep)
            print(f"  sample channels: {args.sample_channels} -> indices {keep.tolist()}  "
                  f"static={'off' if args.no_static else 'on'}")
        print(f"\nnormalisation reference '{reference}': {train}  "
              f"fit_valid={train.valid_fraction():.1%}")
        result = ensemble_oof(
            train, model_kwargs(train, args, extra_kw), seeds,
            n_splits=args.n_splits, n_repeats=args.n_repeats,
        )
        print(f"  token_cnn ({reference})  acc={result['accuracy']:.3f}  "
              f"bal_acc={result['balanced_accuracy']:.3f}  "
              f"macroF1={result['macro_f1']:.3f}  "
              f"(pixel CNN {PIXEL_CNN_ACC:.3f}, classical pca_mlp {CLASSICAL_ACC:.3f})")
        rows.append({
            "model": f"token_cnn_{reference}",
            "norm_reference": reference,
            "n_shots": n_shots or int(cfg["data"].get("n_shots", 200)),
            "n_rows": train.token_shape[0],
            "n_lines": train.n_lines,
            "n_features": train.token_shape[2],
            "include_static": not args.no_static,
            "sample_channels": args.sample_channels or "all",
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "macro_f1": result["macro_f1"],
            "seed_mean": float(np.mean(result["seed_accuracies"])),
            "seed_std": float(np.std(result["seed_accuracies"])),
        })
        results[reference] = (train, result)

    best_ref = max(results, key=lambda r: results[r][1]["accuracy"])
    best_train, best = results[best_ref]
    y = best_train.y.astype(int)
    classes = best["classes"]

    if args.ablation and args.sample_channels:
        print("ablation skip: sample-channels already restricts the token tensor")
        args.ablation = False

    if args.ablation:
        print(f"\nablations on '{best_ref}'")
        strongest = np.sort(np.argsort(best_train.static[:, 6])[::-1][:250])
        for name, subset, extra_kw in (
            ("top250_lines", best_train.select_lines(strongest), {}),
            ("amplitude_only", best_train.select_channels([F_MAX_INT, F_VALID]),
             {"valid_index": 1, "n_dynamic": 1}),
        ):
            kw = {**model_kwargs(subset, args), **extra_kw}
            res = ensemble_oof(subset, kw, seeds[:3],
                               n_splits=args.n_splits, n_repeats=1, verbose=False)
            print(f"  {name:<16s} acc={res['accuracy']:.3f}  "
                  f"bal_acc={res['balanced_accuracy']:.3f}")
            rows.append({
                "model": name, "norm_reference": best_ref,
                "n_lines": subset.n_lines,
                "accuracy": res["accuracy"],
                "balanced_accuracy": res["balanced_accuracy"],
                "macro_f1": res["macro_f1"],
                "seed_mean": float(np.mean(res["seed_accuracies"])),
                "seed_std": float(np.std(res["seed_accuracies"])),
            })

        print("  classical pca_mlp on token amplitudes (depth x lines)")
        amp = best_train.X[..., F_MAX_INT].reshape(len(y), -1)
        classical = cross_validate_model(
            build_model_zoo()["pca_mlp"], amp, y, best_train.groups,
            best_train.sample_ids, name="pca_mlp_tokens",
            n_splits=args.n_splits, n_repeats=max(args.n_repeats, 4),
        )
        print(f"  pca_mlp_tokens   acc={classical.accuracy:.3f}+-{classical.accuracy_std:.3f}  "
              f"bal_acc={classical.balanced_accuracy:.3f}")
        rows.append({
            "model": "pca_mlp_tokens", "norm_reference": best_ref,
            "n_lines": best_train.n_lines,
            "accuracy": classical.accuracy,
            "balanced_accuracy": classical.balanced_accuracy,
            "macro_f1": classical.macro_f1,
            "seed_mean": classical.accuracy,
            "seed_std": classical.accuracy_std,
        })

    blend_rows, blend_best, stack = [], None, None
    partners = []
    for label, path in (
        ("pca_mlp", Path(args.classical_oof)),
        ("pixel_cnn", Path(args.pixel_oof)),
    ):
        if not path.exists():
            print(f"blend skip {label}: {path} not found")
            continue
        repeats = _repeat_probas(pd.read_csv(path), best_train.sample_ids, classes)
        if repeats is None:
            print(f"blend skip {label}: could not align {path}")
            continue
        other_accs = [accuracy_score(y, classes[p.argmax(1)]) for p in repeats]
        other_acc = float(np.mean(other_accs))
        grid, winner = blend_grid(best["proba"], repeats, y, classes, "token_cnn", label)
        print(f"\nblend vs {label} (partner OOF acc={other_acc:.3f}):")
        print(grid[["w_a", "accuracy", "balanced_accuracy", "macro_f1"]].to_string(index=False))
        print(f"  best w_token={winner['w_a']:.1f}  acc={winner['accuracy']:.3f}")
        blend_rows.append(grid)
        other_mean = np.mean(np.stack(repeats, axis=0), axis=0)
        partners.append((label, other_mean, other_acc, winner))
        if blend_best is None or winner["accuracy"] > blend_best["accuracy"]:
            blend_best = {**winner, "partner": label, "partner_acc": other_acc}

    if partners:
        name, other, other_acc, _ = next(
            (p for p in partners if p[0] == "pca_mlp"), partners[0],
        )
        stack = stack_lr(best["proba"], other, y, classes, best_train.groups,
                         n_splits=args.n_splits, n_repeats=args.n_repeats)
        print(f"  stack_lr vs {name}  acc={stack['accuracy']:.3f}")

    board = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    out_csv = cfg.metrics_dir / f"benchmark_{args.tag}.csv"
    board.to_csv(out_csv, index=False)
    print("\n" + board.to_string(index=False))

    if blend_rows:
        blend_frame = pd.concat(blend_rows, ignore_index=True)
        blend_csv = cfg.metrics_dir / f"blend_{args.tag}.csv"
        blend_frame.to_csv(blend_csv, index=False)
        print(f"blend grid -> {blend_csv}")

    oof_frame = pd.DataFrame({
        "sample_id": best_train.sample_ids, "y_true": y, "y_pred": best["pred"],
        **{f"p{int(c)}": best["proba"][:, i] for i, c in enumerate(classes)},
    })
    oof_frame.to_csv(cfg.predictions_dir / f"oof_{args.tag}.csv", index=False)
    plotting.plot_confusion(
        confusion_matrix(y, best["pred"], labels=classes), classes,
        cfg.figures_dir / f"confusion_{args.tag}.png",
        title=f"token CNN ({best_ref}), acc {best['accuracy']:.3f}",
    )
    if blend_best is not None:
        plotting.plot_confusion(
            confusion_matrix(y, blend_best["pred"], labels=classes), classes,
            cfg.figures_dir / f"confusion_{args.tag}_blend.png",
            title=(f"token CNN + {blend_best['partner']} "
                   f"w={blend_best['w_a']:.1f}, acc {blend_best['accuracy']:.3f}"),
        )

    payload = {
        "model": "token_cnn",
        "cv_accuracy": best["accuracy"],
        "balanced_accuracy": best["balanced_accuracy"],
        "macro_f1": best["macro_f1"],
        "seed_accuracies": best["seed_accuracies"],
        "pixel_cnn_cv": PIXEL_CNN_ACC,
        "classical_pca_mlp_oof": None if blend_best is None else blend_best.get("partner_acc"),
        "blend_accuracy": None if blend_best is None else blend_best["accuracy"],
        "blend_w_cnn": None if blend_best is None else blend_best["w_a"],
        "blend_partner": None if blend_best is None else blend_best.get("partner"),
        "stack_lr_accuracy": None if stack is None else stack["accuracy"],
        "params": {
            "norm_reference": best_ref,
            "shot_bin": shot_bin,
            "n_shots": n_shots,
            "n_rows": best_train.token_shape[0],
            "n_lines": best_train.n_lines,
            "n_features": best_train.token_shape[2],
            "sample_channels": args.sample_channels,
            "include_static": not args.no_static,
            "channels": [int(c) for c in args.channels.split(",") if c.strip()],
            "dropout": args.dropout,
            "token_dropout": args.token_dropout,
            "noise_std": args.noise_std,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "ensemble_seeds": seeds,
            "blend_oof": (
                "pca_mlp" if (
                    blend_best is not None
                    and blend_best.get("partner") == "pca_mlp"
                    and blend_best["w_a"] > 0
                    and blend_best["accuracy"] > best["accuracy"]
                ) else None
            ),
            "blend_weight": None if blend_best is None else blend_best["w_a"],
        },
    }
    tagged_path = cfg.models_dir / f"best_params_{args.tag}.json"
    official_path = cfg.models_dir / "best_params_token_cnn.json"
    previous = {}
    if official_path.exists():
        with open(official_path, encoding="utf-8") as fh:
            previous = json.load(fh)
    with open(tagged_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    if previous.get("cv_accuracy", -1) > payload["cv_accuracy"]:
        print(f"keeping existing {official_path} "
              f"(acc {previous['cv_accuracy']:.3f} > {payload['cv_accuracy']:.3f})")
        print(f"wrote {tagged_path}")
    else:
        with open(official_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"updated {official_path} and {tagged_path}")

    print(f"\nleaderboard -> {out_csv}")
    print(f"token CNN {best['accuracy']:.3f}  vs pixel CNN {PIXEL_CNN_ACC:.3f}  "
          f"vs classical {CLASSICAL_ACC:.3f}")


if __name__ == "__main__":
    main()
