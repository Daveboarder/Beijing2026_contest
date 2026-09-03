"""Compare every classifier in the model zoo under identical cross-validation.

    python scripts/03_benchmark_models.py --n-groups 4

Results land in ``results/metrics/benchmark_<tag>.csv`` and the corresponding
out-of-fold predictions in ``results/predictions/``.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from libs2026 import (
    ENCODINGS,
    Config,
    Preprocessor,
    build_features,
    build_model_zoo,
    cross_validate_model,
    summarize,
)
from libs2026 import plotting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of model names (default: all)")
    parser.add_argument("--n-groups", type=int, default=1,
                        help="feature rows per sample (interleaved shot subsets)")
    parser.add_argument("--bin-factor", type=int, default=1,
                        help="average neighbouring wavelength channels")
    parser.add_argument("--encoding", default=None, choices=ENCODINGS,
                        help="how the depth profile is encoded (default: from config)")
    parser.add_argument("--n-bins", type=int, default=None,
                        help="number of log-spaced depth bins")
    parser.add_argument("--n-pca", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv_cfg = cfg["cv"]
    feat_cfg = cfg.get("features", {})
    encoding = args.encoding or feat_cfg.get("encoding", "mean")
    n_bins = args.n_bins or feat_cfg.get("n_bins", 8)
    pre = Preprocessor.from_config(cfg)

    features = build_features(cfg, pre, n_groups=args.n_groups,
                              bin_factor=args.bin_factor, encoding=encoding,
                              n_bins=n_bins, n_jobs=args.n_jobs)
    train = features.subset("train")
    y = train.y.astype(int)
    print(f"encoding: {encoding} ({n_bins} depth bins)")
    print(f"training matrix: {train.X.shape}  "
          f"({len(np.unique(train.groups))} samples, {args.n_groups} row(s) each)")

    zoo = build_model_zoo(n_pca=args.n_pca)
    names = args.models or list(zoo)
    results = []
    for name in names:
        result = cross_validate_model(
            zoo[name], train.X, y, train.groups, train.sample_ids, name=name,
            n_splits=cv_cfg["n_splits"], n_repeats=cv_cfg["n_repeats"],
            random_state=cv_cfg["random_state"],
        )
        results.append(result)
        print(f"{name:16s} acc={result.accuracy:.3f}+-{result.accuracy_std:.3f}  "
              f"bal_acc={result.balanced_accuracy:.3f}  "
              f"macroF1={result.macro_f1:.3f}  ({result.fit_seconds:.1f}s)")

    tag = args.tag or f"{encoding}_g{args.n_groups}_b{args.bin_factor}"
    summary = summarize(results)
    summary.insert(1, "encoding", encoding)
    summary.insert(2, "n_groups", args.n_groups)
    summary.insert(3, "bin_factor", args.bin_factor)
    out_csv = cfg.metrics_dir / f"benchmark_{tag}.csv"
    summary.to_csv(out_csv, index=False)

    for result in results:
        result.oof.to_csv(cfg.predictions_dir / f"oof_{tag}_{result.name}.csv", index=False)

    best = results[int(np.argmax([r.accuracy for r in results]))]
    plotting.plot_leaderboard(summary, cfg.figures_dir / f"leaderboard_{tag}.png")
    plotting.plot_confusion(best.confusion, np.unique(y),
                            cfg.figures_dir / f"confusion_{tag}_{best.name}.png",
                            title=f"{best.name}: out-of-fold confusion ({tag})")

    with open(cfg.metrics_dir / f"benchmark_{tag}_settings.json", "w", encoding="utf-8") as fh:
        json.dump({"preprocessing": cfg["preprocessing"], "cv": cv_cfg,
                   "encoding": encoding, "n_bins": n_bins,
                   "n_groups": args.n_groups, "bin_factor": args.bin_factor,
                   "n_pca": args.n_pca}, fh, indent=2)

    print(f"\nbest model: {best.name} (accuracy {best.accuracy:.3f})")
    print(f"leaderboard -> {out_csv}")


if __name__ == "__main__":
    main()
