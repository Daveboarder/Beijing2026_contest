"""Sweep the depth-transformer regularisation over single-seed grouped CV.

    uv run --extra cnn python scripts/16_tune_transformer.py --device cuda

The first run of this architecture collapsed to a constant output: with
``dropout 0.3`` plus AdamW ``weight_decay 1e-2`` on 96 training samples every
sample received the same uniform class posterior (mean top probability 0.20 =
1/5). Turning the regularisation off recovered a normal fit, so the knob that
matters here is *how little* regularisation the model needs, not how much.

Each trial is one seed of grouped 5-fold CV, which is noisy — treat it as a
screen and re-score the winner with ``15_benchmark_transformer.py`` and its
full seed ensemble.
"""

import argparse
from dataclasses import replace

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from libs2026 import Config, Preprocessor
from libs2026.cnn import SpectrumTransformer
from libs2026.images import build_images

# name -> overrides on top of the SpectrumTransformer defaults.
TRIALS: dict[str, dict] = {
    "no_reg": dict(dropout=0.0, encoder_dropout=0.0, noise_std=0.0,
                   wl_shift=0, shot_mask_p=0.0, weight_decay=1e-4),
    "aug_only": dict(dropout=0.0, encoder_dropout=0.0, weight_decay=1e-4),
    "drop10": dict(dropout=0.1, encoder_dropout=0.05, weight_decay=1e-4),
    "drop20": dict(dropout=0.2, encoder_dropout=0.1, weight_decay=1e-4),
    "drop10_mask": dict(dropout=0.1, encoder_dropout=0.05, weight_decay=1e-4,
                        shot_mask_p=0.1),
    "drop10_wd1e-3": dict(dropout=0.1, encoder_dropout=0.05, weight_decay=1e-3),
    "drop10_ep250": dict(dropout=0.1, encoder_dropout=0.05, weight_decay=1e-4,
                         epochs=250),
    "drop10_lr1e-3": dict(dropout=0.1, encoder_dropout=0.05, weight_decay=1e-4,
                          lr=1e-3),
}


def score_cv(model_kw, X, y, groups, seed, n_splits, n_repeats):
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits, shuffle=True,
                                        random_state=seed + repeat)
        fold = np.zeros_like(proba)
        for tr, te in splitter.split(X, y, groups):
            model = SpectrumTransformer(**{**model_kw, "random_state": seed})
            model.fit(X[tr], y[tr])
            fold[te] = model.predict_proba(X[te])
        proba += fold
    proba /= n_repeats
    pred = classes[proba.argmax(1)]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "mean_top_p": float(proba.max(1).mean()),
        "n_predicted_classes": int(len(np.unique(pred))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bin-factor", type=int, default=8)
    parser.add_argument("--shot-bin", type=int, default=4)
    parser.add_argument("--norm-reference", default="shot", choices=["shot", "bulk"])
    parser.add_argument("--trials", default=None,
                        help="comma-separated subset of the trial names")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="transformer")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    pre = replace(Preprocessor.from_config(cfg),
                  normalization_reference=args.norm_reference)
    images = build_images(cfg, pre, bin_factor=args.bin_factor,
                          shot_bin=args.shot_bin, n_jobs=args.n_jobs)
    train = images.subset("train")
    n_shots, n_wl = train.image_shape
    X, y = train.as_flat(), train.y.astype(int)

    names = ([t.strip() for t in args.trials.split(",") if t.strip()]
             if args.trials else list(TRIALS))
    base = dict(n_shots=n_shots, n_wavelengths=n_wl, device=args.device, verbose=0)

    rows = []
    for i, name in enumerate(names, 1):
        overrides = TRIALS[name]
        res = score_cv({**base, **overrides}, X, y, train.groups,
                       args.seed, args.n_splits, args.n_repeats)
        rows.append({"trial": name, **res, **overrides})
        collapsed = " (collapsed)" if res["n_predicted_classes"] <= 1 else ""
        print(f"[{i:02d}/{len(names)}] {name:<16s} acc={res['accuracy']:.3f}  "
              f"bal_acc={res['balanced_accuracy']:.3f}  "
              f"top_p={res['mean_top_p']:.2f}{collapsed}", flush=True)

    board = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    out = cfg.metrics_dir / f"tuning_{args.tag}.csv"
    board.to_csv(out, index=False)
    print("\n" + board[["trial", "accuracy", "balanced_accuracy",
                        "macro_f1", "mean_top_p"]].to_string(index=False))
    print(f"\nsweep -> {out}")


if __name__ == "__main__":
    main()
