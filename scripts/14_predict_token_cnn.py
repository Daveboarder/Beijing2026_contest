"""Fit the token CNN on all training samples and write a contest submission.

    uv run --extra cnn python scripts/14_predict_token_cnn.py --device cuda

Defaults come from ``results/models/best_params_token_cnn.json`` when present.
``--blend-oof`` mixes the CNN probabilities with a classical model, which is
what produced the best cross-validated score so far: the two families make
different mistakes, so averaging them beats either alone.
"""

import argparse
import json
from dataclasses import replace
from datetime import datetime

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from libs2026 import Config, Preprocessor, build_features, build_model_zoo
from libs2026.cnn import TokenCNN
from libs2026.evaluation import predict_scores
from libs2026.tokens import (
    build_tokens,
    fit_config_from_config,
    line_dictionary_from_config,
)


def validate_submission(frame: pd.DataFrame, cfg: Config) -> None:
    template = pd.read_csv(cfg.sample_submission)
    assert list(frame.columns) == ["filename", "predicted_label"], frame.columns
    assert len(frame) == len(template)
    assert not frame["filename"].duplicated().any()
    assert set(frame["filename"]) == set(template["filename"])
    assert frame["predicted_label"].between(1, 5).all()
    assert frame["predicted_label"].dtype.kind in "iu"


def _classical_proba(cfg, classes, test_ids, model_name):
    """Test-set probabilities from a classical model fitted on all training rows."""
    features = build_features(cfg)
    train, test = features.subset("train"), features.subset("test")
    model = clone(build_model_zoo()[model_name])
    model.fit(train.X, train.y.astype(int))
    scores = predict_scores(model, test.X, classes)

    frame = pd.DataFrame(scores, columns=[f"p{int(c)}" for c in classes])
    frame["sample_id"] = test.sample_ids
    agg = frame.groupby("sample_id", as_index=False).mean()
    order = agg.set_index("sample_id").loc[test_ids]
    return order[[f"p{int(c)}" for c in classes]].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--shot-bin", type=int, default=None)
    parser.add_argument("--norm-reference", default=None, choices=["shot", "bulk"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--ensemble-seeds", default=None)
    parser.add_argument("--blend-oof", default=None,
                        help="classical model to blend with, e.g. pca_mlp")
    parser.add_argument("--blend-weight", type=float, default=0.5,
                        help="weight of the token CNN in the blend")
    parser.add_argument("--n-jobs", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    tok_cfg = cfg.get("tokens", {})

    best_path = cfg.models_dir / "best_params_token_cnn.json"
    best = {}
    if best_path.exists():
        with open(best_path, encoding="utf-8") as fh:
            best = json.load(fh).get("params", {})

    def pick(name, cast, default):
        cli = getattr(args, name)
        if cli is not None:
            return cast(cli)
        if name in best:
            return cast(best[name])
        return default

    shot_bin = pick("shot_bin", int, int(tok_cfg.get("shot_bin", 4)))
    norm_reference = pick("norm_reference", str, "bulk")
    epochs = pick("epochs", int, 150)
    batch_size = pick("batch_size", int, 16)
    lr = pick("lr", float, 1e-3)
    dropout = pick("dropout", float, 0.4)

    channels_raw = args.channels or best.get("channels", [32, 64, 128])
    channels = (tuple(int(c) for c in channels_raw.split(",") if c.strip())
                if isinstance(channels_raw, str) else tuple(int(c) for c in channels_raw))
    if args.ensemble_seeds is not None:
        seeds = [int(s) for s in args.ensemble_seeds.split(",") if s.strip()]
    else:
        seeds = list(best.get("ensemble_seeds", [42, 7, 123, 99, 2026]))

    pre = replace(Preprocessor.from_config(cfg), normalization_reference=norm_reference)
    dictionary = line_dictionary_from_config(cfg, verbose=False)
    tokens = build_tokens(cfg, dictionary, pre=pre, shot_bin=shot_bin,
                          fit_cfg=fit_config_from_config(cfg), n_jobs=args.n_jobs)
    train, test = tokens.subset("train"), tokens.subset("test")
    rows, lines, feats = train.token_shape
    x_train, y_train = train.as_flat(), train.y.astype(int)
    x_test = test.as_flat()

    print(f"fitting {len(seeds)} seed(s) on {train.X.shape} -> predict {test.X.shape}")
    proba, models, classes = None, [], None
    for seed in seeds:
        model = TokenCNN(
            static=train.static, feature_mean=train.feature_mean,
            feature_std=train.feature_std, n_rows=rows, n_lines=lines, n_features=feats,
            channels=channels, dropout=dropout, epochs=epochs,
            batch_size=batch_size, lr=lr, device=args.device, random_state=int(seed),
        )
        model.fit(x_train, y_train)
        p = model.predict_proba(x_test)
        proba = p if proba is None else proba + p
        models.append(model)
        classes = model.classes_
    proba /= len(seeds)

    if args.blend_oof:
        classical = _classical_proba(cfg, classes, test.sample_ids, args.blend_oof)
        w = args.blend_weight
        print(f"blending with {args.blend_oof} at w_cnn={w:.2f}")
        proba = w * proba + (1.0 - w) * classical

    pred = classes[proba.argmax(axis=1)]
    submission = pd.DataFrame({
        "filename": test.sample_ids + ".csv",
        "predicted_label": pred.astype(int),
    }).sort_values("filename").reset_index(drop=True)
    validate_submission(submission, cfg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = args.name or f"predictions_token_cnn_{stamp}.csv"
    out = cfg.submissions_dir / name
    submission.to_csv(out, index=False)
    joblib.dump(
        {"models": models, "seeds": seeds, "shot_bin": shot_bin,
         "norm_reference": norm_reference, "n_lines": lines,
         "blend": args.blend_oof, "blend_weight": args.blend_weight},
        cfg.models_dir / f"fitted_token_cnn_{stamp}.joblib",
    )

    print("predicted class distribution:")
    print(submission["predicted_label"].value_counts().sort_index().to_string())
    print(f"mean top-class probability: {proba.max(axis=1).mean():.3f}")
    print(f"submission -> {out}")


if __name__ == "__main__":
    main()
