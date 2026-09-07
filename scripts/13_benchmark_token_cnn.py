"""Cross-validate the 2-D CNN over spectral-line tokens.

    uv run --extra cnn python scripts/13_benchmark_token_cnn.py --device cuda

Columns are physical transitions instead of wavelength bins, so no per-fold
spectral PCA is needed. Single-seed CV is very noisy at N=120; out-of-fold
probabilities are averaged over ``--ensemble-seeds`` as in
``09_benchmark_cnn.py``, which is what makes the numbers comparable.

``--ablation`` additionally scores a reduced line set and an amplitude-only
variant, to check whether the physics channels earn their place.
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

from libs2026 import Config, Preprocessor, plotting
from libs2026.cnn import TokenCNN
from libs2026.tokens import (
    build_tokens,
    fit_config_from_config,
    line_dictionary_from_config,
)


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


def model_kwargs(tokens, args):
    rows, lines, feats = tokens.token_shape
    return dict(
        static=tokens.static,
        feature_mean=tokens.feature_mean,
        feature_std=tokens.feature_std,
        n_rows=rows, n_lines=lines, n_features=feats,
        channels=tuple(int(c) for c in args.channels.split(",") if c.strip()),
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        token_dropout=args.token_dropout,
        noise_std=args.noise_std,
        device=args.device,
        verbose=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--shot-bin", type=int, default=None)
    parser.add_argument("--norm-reference", default="bulk", choices=["shot", "bulk", "both"],
                        help="'bulk' keeps the depth intensity profile that token "
                             "amplitudes depend on; 'both' sweeps the two")
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
    parser.add_argument("--n-jobs", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="token_cnn")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    tok_cfg = cfg.get("tokens", {})
    shot_bin = args.shot_bin if args.shot_bin is not None else int(tok_cfg.get("shot_bin", 4))
    seeds = [int(s) for s in args.ensemble_seeds.split(",") if s.strip()]
    fit_cfg = fit_config_from_config(cfg)
    dictionary = line_dictionary_from_config(cfg, verbose=False)

    references = ["shot", "bulk"] if args.norm_reference == "both" else [args.norm_reference]
    rows, results = [], {}

    for reference in references:
        pre = replace(Preprocessor.from_config(cfg), normalization_reference=reference)
        tokens = build_tokens(cfg, dictionary, pre=pre, shot_bin=shot_bin,
                              fit_cfg=fit_cfg, n_jobs=args.n_jobs, verbose=True)
        train = tokens.subset("train")
        print(f"\nnormalisation reference '{reference}': {train}  "
              f"fit_valid={train.valid_fraction():.1%}")
        result = ensemble_oof(
            train, model_kwargs(train, args), seeds,
            n_splits=args.n_splits, n_repeats=args.n_repeats,
        )
        print(f"  token_cnn ({reference})  acc={result['accuracy']:.3f}  "
              f"bal_acc={result['balanced_accuracy']:.3f}  "
              f"macroF1={result['macro_f1']:.3f}")
        rows.append({
            "model": f"token_cnn_{reference}",
            "norm_reference": reference,
            "n_lines": train.n_lines,
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "macro_f1": result["macro_f1"],
            "seed_mean": float(np.mean(result["seed_accuracies"])),
            "seed_std": float(np.std(result["seed_accuracies"])),
        })
        results[reference] = (train, result)

    best_ref = max(results, key=lambda r: results[r][1]["accuracy"])
    best_train, best = results[best_ref]

    if args.ablation:
        print(f"\nablations on '{best_ref}'")
        strongest = np.sort(np.argsort(dictionary.theoretical_intensity)[::-1][:250])
        strongest = strongest[strongest < best_train.n_lines]
        for name, subset, extra in (
            ("top250_lines", best_train.select_lines(strongest), {}),
            ("amplitude_only", best_train, {"n_features_used": 1}),
        ):
            kw = model_kwargs(subset, args)
            if extra.get("n_features_used"):
                # Amplitude and mask only: does the rest of the Voigt fit help?
                keep = np.array([0, 5])
                sub = subset.X[..., keep]
                subset = type(subset)(
                    sub, subset.y, subset.sample_ids, subset.split, subset.static,
                    subset.line_wavelength, subset.line_element, subset.line_ion_state,
                    subset.feature_mean[:, keep], subset.feature_std[:, keep],
                )
                kw = model_kwargs(subset, args)
                kw["valid_index"] = 1
                kw["n_dynamic"] = 1
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

    board = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    out_csv = cfg.metrics_dir / f"benchmark_{args.tag}.csv"
    board.to_csv(out_csv, index=False)
    print("\n" + board.to_string(index=False))

    y = best_train.y.astype(int)
    oof_frame = pd.DataFrame({
        "sample_id": best_train.sample_ids, "y_true": y, "y_pred": best["pred"],
        **{f"p{int(c)}": best["proba"][:, i] for i, c in enumerate(best["classes"])},
    })
    oof_frame.to_csv(cfg.predictions_dir / f"oof_{args.tag}.csv", index=False)
    plotting.plot_confusion(
        confusion_matrix(y, best["pred"], labels=best["classes"]), best["classes"],
        cfg.figures_dir / f"confusion_{args.tag}.png",
        title=f"token CNN ({best_ref}), acc {best['accuracy']:.3f}",
    )
    with open(cfg.models_dir / "best_params_token_cnn.json", "w", encoding="utf-8") as fh:
        json.dump({
            "model": "token_cnn",
            "cv_accuracy": best["accuracy"],
            "balanced_accuracy": best["balanced_accuracy"],
            "macro_f1": best["macro_f1"],
            "seed_accuracies": best["seed_accuracies"],
            "params": {
                "norm_reference": best_ref,
                "shot_bin": shot_bin,
                "n_lines": best_train.n_lines,
                "channels": [int(c) for c in args.channels.split(",") if c.strip()],
                "dropout": args.dropout,
                "token_dropout": args.token_dropout,
                "noise_std": args.noise_std,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "ensemble_seeds": seeds,
            },
        }, fh, indent=2)

    print(f"\nleaderboard -> {out_csv}")


if __name__ == "__main__":
    main()
