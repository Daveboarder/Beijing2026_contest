"""Hyper-parameter search for a single model family.

    python scripts/05_tune_model.py --model pca_svm_rbf

The search uses the same grouped, stratified splitter as the benchmark, so the
reported score stays comparable with ``03_benchmark_models.py``.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from libs2026 import Config, Preprocessor, build_features, get_model

PARAM_GRIDS = {
    "plsda": {"clf__n_components": [5, 10, 15, 20, 30, 40]},
    "pca_lda": {"pca__n_components": [10, 20, 30, 50, 80],
                "clf__solver": ["svd", "lsqr"], "clf__shrinkage": [None]},
    "pca_logreg": {"pca__n_components": [10, 20, 30, 50, 80],
                   "clf__C": [0.01, 0.1, 1.0, 10.0]},
    "pca_svm_rbf": {"pca__n_components": [10, 20, 30, 50, 80],
                    "clf__C": [1.0, 10.0, 100.0, 1000.0],
                    "clf__gamma": ["scale", 0.001, 0.01]},
    "svm_linear": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "pca_knn": {"pca__n_components": [10, 20, 30, 50],
                "clf__n_neighbors": [1, 3, 5, 7, 11]},
    "pca_mlp": {"pca__n_components": [20, 30, 50],
                "clf__hidden_layer_sizes": [(64,), (128, 64), (256, 128)],
                "clf__alpha": [1e-4, 1e-3, 1e-2]},
    "random_forest": {"clf__max_features": ["sqrt", 0.05, 0.2],
                      "clf__min_samples_leaf": [1, 2, 4]},
    "extra_trees": {"clf__max_features": ["sqrt", 0.05, 0.2],
                    "clf__min_samples_leaf": [1, 2, 4]},
    "hist_gbdt": {"pca__n_components": [20, 30, 50],
                  "clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [15, 31]},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", required=True, choices=sorted(PARAM_GRIDS))
    parser.add_argument("--n-groups", type=int, default=1)
    parser.add_argument("--bin-factor", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    features = build_features(cfg, Preprocessor.from_config(cfg), n_groups=args.n_groups,
                              bin_factor=args.bin_factor, n_jobs=args.n_jobs)
    train = features.subset("train")
    y = train.y.astype(int)

    splitter = StratifiedGroupKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True,
                                    random_state=cfg["cv"]["random_state"])
    search = GridSearchCV(
        get_model(args.model), PARAM_GRIDS[args.model], scoring="accuracy",
        cv=splitter.split(train.X, y, train.groups), n_jobs=args.n_jobs, verbose=1,
    )
    search.fit(train.X, y)

    frame = (pd.DataFrame(search.cv_results_)
             [["params", "mean_test_score", "std_test_score", "rank_test_score"]]
             .sort_values("rank_test_score"))
    out = cfg.metrics_dir / f"tuning_{args.model}_g{args.n_groups}.csv"
    frame.to_csv(out, index=False)

    best = {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in search.best_params_.items()}
    with open(cfg.models_dir / f"best_params_{args.model}.json", "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "n_groups": args.n_groups,
                   "bin_factor": args.bin_factor, "params": best,
                   "cv_accuracy": float(search.best_score_)}, fh, indent=2)

    print(f"\nbest CV accuracy: {search.best_score_:.3f}")
    print("best parameters:")
    for key, value in search.best_params_.items():
        print(f"  {key} = {value}")
    print(f"\nfull grid -> {out}")
    print(frame.head(8).to_string(index=False))
    np.set_printoptions(suppress=True)


if __name__ == "__main__":
    main()
