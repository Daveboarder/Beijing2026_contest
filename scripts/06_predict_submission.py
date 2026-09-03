"""Fit the chosen model on all 120 training samples and predict the test set.

    python scripts/06_predict_submission.py --model pca_mlp \
        --params results/models/best_params_pca_mlp.json

Writes a contest-format CSV to ``submissions/`` and validates it against
``sample_submission/predictions.csv``.
"""

import argparse
import json
from datetime import datetime

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd

from libs2026 import (
    Config,
    Preprocessor,
    aggregate_predictions,
    build_features,
    cross_validate_model,
    get_model,
    predict_scores,
)


def validate_submission(frame: pd.DataFrame, cfg: Config) -> None:
    """Fail loudly if the file would be rejected by the organisers."""
    template = pd.read_csv(cfg.sample_submission)
    assert list(frame.columns) == ["filename", "predicted_label"], frame.columns
    assert len(frame) == len(template), f"{len(frame)} rows, expected {len(template)}"
    assert not frame["filename"].duplicated().any(), "duplicate filenames"
    assert set(frame["filename"]) == set(template["filename"]), "filename mismatch"
    assert frame["predicted_label"].between(1, 5).all(), "labels outside 1-5"
    assert frame["predicted_label"].dtype.kind in "iu", "labels must be integers"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="pca_mlp")
    parser.add_argument("--params", default=None, help="JSON file from 05_tune_model.py")
    parser.add_argument("--n-groups", type=int, default=1)
    parser.add_argument("--bin-factor", type=int, default=1)
    parser.add_argument("--n-pca", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--name", default=None, help="output file name")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()

    n_groups, bin_factor = args.n_groups, args.bin_factor
    params = {}
    if args.params:
        blob = json.loads(open(args.params, encoding="utf-8").read())
        params = {k: (tuple(v) if isinstance(v, list) else v)
                  for k, v in blob["params"].items()}
        n_groups = blob.get("n_groups", n_groups)
        bin_factor = blob.get("bin_factor", bin_factor)
        print(f"loaded tuned parameters ({blob['model']}, CV {blob['cv_accuracy']:.3f})")

    features = build_features(cfg, Preprocessor.from_config(cfg), n_groups=n_groups,
                              bin_factor=bin_factor, n_jobs=args.n_jobs)
    train, test = features.subset("train"), features.subset("test")
    y = train.y.astype(int)

    model = get_model(args.model, n_pca=args.n_pca)
    if params:
        model.set_params(**params)

    # Honest estimate of what this exact configuration scores, before refitting.
    cv = cross_validate_model(model, train.X, y, train.groups, train.sample_ids,
                              name=args.model, n_splits=cfg["cv"]["n_splits"],
                              n_repeats=cfg["cv"]["n_repeats"],
                              random_state=cfg["cv"]["random_state"])
    print(f"cross-validated accuracy: {cv.accuracy:.3f} +- {cv.accuracy_std:.3f}")

    model.fit(train.X, y)
    proba = predict_scores(model, test.X, model.classes_)
    predictions = aggregate_predictions(test.sample_ids, proba, model.classes_)

    submission = pd.DataFrame({
        "filename": predictions["sample_id"] + ".csv",
        "predicted_label": predictions["predicted_label"].astype(int),
    }).sort_values("filename").reset_index(drop=True)
    validate_submission(submission, cfg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = args.name or f"predictions_{args.model}_{stamp}.csv"
    out = cfg.submissions_dir / name
    submission.to_csv(out, index=False)
    predictions.to_csv(cfg.predictions_dir / f"test_proba_{args.model}_{stamp}.csv", index=False)
    joblib.dump(model, cfg.models_dir / f"fitted_{args.model}_{stamp}.joblib")

    print("\npredicted class distribution:")
    print(submission["predicted_label"].value_counts().sort_index().to_string())
    print("\ntraining class distribution:")
    print(pd.Series(y).value_counts().sort_index().to_string())
    confidence = predictions[[c for c in predictions.columns if c in model.classes_]]
    print(f"\nmean top-class probability: {np.max(confidence.to_numpy(), axis=1).mean():.3f}")
    print(f"submission -> {out}")


if __name__ == "__main__":
    main()
