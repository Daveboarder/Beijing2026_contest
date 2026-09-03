"""Sweep preprocessing variants against a few fast reference classifiers.

For LIBS the preprocessing chain usually matters as much as the classifier, so
this script crosses baseline removal, normalisation and shot-block averaging
and reports the cross-validated accuracy of each combination.
"""

import argparse
import itertools

import _bootstrap  # noqa: F401
import pandas as pd

from libs2026 import (
    Config,
    Preprocessor,
    build_features,
    build_model_zoo,
    cross_validate_model,
)

BASELINES = ["none", "snip"]
NORMALIZATIONS = ["none", "tic", "max", "snv", "l2"]
REFERENCE_MODELS = ["pca_lda", "pca_logreg", "plsda", "pca_mlp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="*", default=REFERENCE_MODELS)
    parser.add_argument("--baselines", nargs="*", default=BASELINES)
    parser.add_argument("--normalizations", nargs="*", default=NORMALIZATIONS)
    parser.add_argument("--n-groups", type=int, default=1)
    parser.add_argument("--bin-factor", type=int, default=1)
    parser.add_argument("--n-pca", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv_cfg = cfg["cv"]
    zoo = build_model_zoo(n_pca=args.n_pca)
    base = Preprocessor.from_config(cfg)

    rows = []
    for baseline, norm in itertools.product(args.baselines, args.normalizations):
        pre = Preprocessor(
            baseline=baseline,
            snip_iterations=base.snip_iterations,
            normalization=norm,
            per_channel=base.per_channel,
            smooth_window=base.smooth_window,
            drop_outlier_shots=base.drop_outlier_shots,
            outlier_z=base.outlier_z,
            channel_bounds=base.channel_bounds,
        )
        features = build_features(cfg, pre, n_groups=args.n_groups,
                                  bin_factor=args.bin_factor, n_jobs=args.n_jobs)
        train = features.subset("train")
        y = train.y.astype(int)
        for name in args.models:
            result = cross_validate_model(
                zoo[name], train.X, y, train.groups, train.sample_ids, name=name,
                n_splits=cv_cfg["n_splits"], n_repeats=cv_cfg["n_repeats"],
                random_state=cv_cfg["random_state"],
            )
            rows.append({"baseline": baseline, "normalization": norm, "model": name,
                         "accuracy": result.accuracy, "accuracy_std": result.accuracy_std,
                         "balanced_accuracy": result.balanced_accuracy,
                         "macro_f1": result.macro_f1})
            print(f"baseline={baseline:5s} norm={norm:5s} {name:12s} "
                  f"acc={result.accuracy:.3f}+-{result.accuracy_std:.3f}")

    frame = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    out = cfg.metrics_dir / f"preprocessing_sweep_g{args.n_groups}_b{args.bin_factor}.csv"
    frame.to_csv(out, index=False)

    print("\ntop 10 combinations")
    print(frame.head(10).to_string(index=False))
    print("\nmean accuracy per preprocessing combination")
    pivot = frame.pivot_table(index="baseline", columns="normalization", values="accuracy")
    print(pivot.round(3).to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
