"""Compare depth-resolved encodings against plain shot averaging.

The 200 shots of a sample form a depth profile through one spot rather than 200
replicates, so this asks the central question directly: does keeping the depth
axis beat averaging it away?

    python scripts/07_compare_encodings.py --n-groups 4 --bin-factor 4
"""

import argparse
from dataclasses import replace
from itertools import product

import _bootstrap  # noqa: F401
import pandas as pd

from libs2026 import (
    AUGMENTATIONS,
    Config,
    Preprocessor,
    build_depth_zoo,
    build_features,
    build_model_zoo,
    cross_validate_model,
    log_bin_edges,
)
from libs2026 import plotting

REFERENCE_MODELS = ["pca_lda", "pca_logreg", "pca_svm_rbf", "pca_mlp", "plsda"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--encodings", nargs="*",
                        default=["mean", "mean_lines", "lines", "mean_depth", "depth_profile"])
    parser.add_argument("--references", nargs="*", default=["bulk", "shot"],
                        choices=["shot", "bulk"],
                        help="normalisation reference; 'shot' flattens the depth "
                             "intensity profile, 'bulk' preserves it")
    parser.add_argument("--models", nargs="*", default=REFERENCE_MODELS)
    parser.add_argument("--augmentations", nargs="*", default=["blocks", "surface"],
                        choices=list(AUGMENTATIONS))
    parser.add_argument("--depth-models", action="store_true",
                        help="also run the shared-basis BinnedPCA models on depth_bins")
    parser.add_argument("--n-groups", type=int, default=1)
    parser.add_argument("--bin-factor", type=int, default=1)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--n-pca", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv_cfg = cfg["cv"]
    base = Preprocessor.from_config(cfg)
    zoo = build_model_zoo(n_pca=args.n_pca)

    rows, best = [], {}
    for reference, augment, encoding in product(args.references, args.augmentations,
                                                args.encodings):
        pre = replace(base, normalization_reference=reference)
        features = build_features(cfg, pre, n_groups=args.n_groups,
                                  bin_factor=args.bin_factor, encoding=encoding,
                                  n_bins=args.n_bins, augment=augment, n_jobs=args.n_jobs)
        train = features.subset("train")
        y = train.y.astype(int)

        models = dict(zoo)
        names = list(args.models)
        if args.depth_models and encoding == "depth_bins":
            # Rows are one spectrum per depth bin, so the shared-basis models
            # need to know how many bins were actually produced.
            shots_per_row = len(train.sample_ids) and cfg["data"]["n_shots"] // args.n_groups
            n_spectra = len(log_bin_edges(shots_per_row, args.n_bins))
            models.update(build_depth_zoo(n_bins=n_spectra, n_components=args.n_pca))
            names += [n for n in models if n.startswith("binpca")]

        print(f"\n{encoding} (norm={reference}, augment={augment}): "
              f"{train.X.shape[1]} features")
        for name in names:
            result = cross_validate_model(
                models[name], train.X, y, train.groups, train.sample_ids, name=name,
                n_splits=cv_cfg["n_splits"], n_repeats=cv_cfg["n_repeats"],
                random_state=cv_cfg["random_state"],
            )
            rows.append({"encoding": encoding, "reference": reference,
                         "augment": augment, "model": name,
                         "n_features": train.X.shape[1],
                         "accuracy": result.accuracy,
                         "accuracy_std": result.accuracy_std,
                         "balanced_accuracy": result.balanced_accuracy,
                         "macro_f1": result.macro_f1})
            print(f"  {name:15s} acc={result.accuracy:.3f}+-{result.accuracy_std:.3f}  "
                  f"bal_acc={result.balanced_accuracy:.3f}")
            if not best or result.accuracy > best["accuracy"]:
                best = {"accuracy": result.accuracy, "encoding": encoding,
                        "name": name, "result": result, "classes": sorted(set(y))}

    frame = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    out = cfg.metrics_dir / f"encoding_comparison_g{args.n_groups}_b{args.bin_factor}.csv"
    frame.to_csv(out, index=False)

    print("\nbest accuracy per encoding")
    print(frame.pivot_table(index="encoding", columns=["reference", "augment"],
                            values="accuracy", aggfunc="max").round(3).to_string())
    print("\ntop 10")
    print(frame.head(10).to_string(index=False))

    plotting.plot_confusion(
        best["result"].confusion, best["classes"],
        cfg.figures_dir / f"confusion_{best['encoding']}_{best['name']}.png",
        title=f"{best['name']} on {best['encoding']} (acc {best['accuracy']:.3f})",
    )
    print(f"\nbest: {best['name']} on {best['encoding']} -> {best['accuracy']:.3f}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
