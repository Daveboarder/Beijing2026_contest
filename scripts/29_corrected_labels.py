"""Retrain pca_mlp and a linear SVM with the suspect training labels corrected.

    python scripts/29_corrected_labels.py --n-repeats 10 --n-jobs 5

The corrections are the consensus of five model types (scripts/28): each of
these samples was predicted as the same other class by at least four of
pca_mlp, pca_svm_rbf, svm_linear, pca_lda and plsda, none of which had seen
it. train_079 is kept, since four of five models supported its label.

Rows: the best setting -- 10 consecutive blocks of 20 shots per sample.

Scoring corrected labels on corrected labels is circular (the models chose
them), so the CV reports two numbers for each label set:
* all 120 samples, against the label set used for training;
* the 110 samples whose label is unchanged -- the fair comparison: does
  training on the corrected labels help predict the samples nobody touched?
Folds are stratified on the original labels for both label sets, so the two
runs use identical splits.

Final models are fitted on all 120 samples with each label set and predict the
60 test samples; contest-format CSVs are written to ``submissions/`` (not sent).
"""

import argparse
from importlib import import_module

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from libs2026 import Config, Preprocessor, build_features, get_model
from libs2026.evaluation import predict_scores

CLASSES = np.array([1, 2, 3, 4, 5])
PCOLS = [f"p{c}" for c in CLASSES]
CORRECTIONS = {  # sample_id: (given, corrected)
    "train_022": (4, 5), "train_042": (2, 5), "train_050": (4, 3), "train_083": (1, 2),
    "train_088": (2, 3), "train_098": (1, 3), "train_101": (2, 3), "train_080": (1, 3),
    "train_078": (3, 2), "train_113": (3, 4),
}


def _repeat(model, X, y_fit, y_strat, groups, n_splits, seed):
    proba = np.zeros((len(y_fit), len(CLASSES)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X, y_strat, groups):
        fitted = clone(model).fit(X[tr], y_fit[tr])
        proba[te] = predict_scores(fitted, X[te], CLASSES)
    return proba


def sample_table(proba, sample_ids, y):
    f = pd.DataFrame(proba, columns=PCOLS)
    f["sample_id"], f["y"] = sample_ids, y
    s = f.groupby("sample_id").agg(y=("y", "first"), **{c: (c, "mean") for c in PCOLS})
    s["pred"] = CLASSES[s[PCOLS].to_numpy().argmax(1)]
    return s


def scores(s):
    return {"accuracy": accuracy_score(s["y"], s["pred"]),
            "balanced_accuracy": recall_score(s["y"], s["pred"], average="macro"),
            "f1_macro": f1_score(s["y"], s["pred"], average="macro")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", default="pca_mlp,svm_linear")
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=5)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv = cfg["cv"]
    validate_submission = import_module("06_predict_submission").validate_submission
    pre = Preprocessor.from_config(cfg)
    pre.normalization_reference = "shot"
    features = build_features(cfg, pre, n_groups=args.n_groups, bin_factor=1, encoding="mean",
                              augment="blocks", n_jobs=8)
    train, test = features.subset("train"), features.subset("test")
    y_orig = train.y.astype(int)
    corrected = pd.Series(y_orig, index=train.sample_ids)
    for sid, (given, new) in CORRECTIONS.items():
        assert (corrected.loc[sid] == given).all(), f"{sid}: expected given label {given}"
    y_corr = np.array([CORRECTIONS.get(s, (None, y))[1] for s, y in zip(train.sample_ids, y_orig)])
    changed = np.isin(train.sample_ids, list(CORRECTIONS))
    pd.DataFrame([{"sample_id": s, "given_label": g, "corrected_label": c} for s, (g, c) in CORRECTIONS.items()]) \
        .to_csv(cfg.metrics_dir / "label_corrections.csv", index=False)
    before = np.bincount(y_orig[::args.n_groups], minlength=6)[1:]
    after = np.bincount(y_corr[::args.n_groups], minlength=6)[1:]
    print(f"rows {train.X.shape}; class counts original {before.tolist()} -> corrected {after.tolist()}")

    summary, submissions = [], {}
    for name in args.models.split(","):
        model = get_model(name)
        for label_set, y_fit in (("original", y_orig), ("corrected", y_corr)):
            probas = Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(_repeat)(model, train.X, y_fit, y_orig, train.groups, cv["n_splits"],
                                 cv["random_state"] + r)
                for r in range(args.n_repeats)
            )
            per_all, per_kept, frames = [], [], []
            for r, p in enumerate(probas):
                s = sample_table(p, train.sample_ids, y_fit)
                kept = s.loc[~s.index.isin(list(CORRECTIONS))]
                per_all.append(scores(s))
                per_kept.append(scores(kept))
                frames.append(s.assign(repeat=r).reset_index())
            oof = pd.concat(frames, ignore_index=True)
            oof.to_csv(cfg.predictions_dir / f"oof_{name}_g{args.n_groups}_{label_set}_labels.csv", index=False)
            a, k = pd.DataFrame(per_all), pd.DataFrame(per_kept)
            summary.append({"model": name, "labels": label_set,
                            **{f"all120_{m}": a[m].mean() for m in a},
                            "all120_accuracy_std": a["accuracy"].std(ddof=0),
                            **{f"unchanged110_{m}": k[m].mean() for m in k},
                            "unchanged110_accuracy_std": k["accuracy"].std(ddof=0)})
            print(f"{name} / {label_set}: all120 acc {a['accuracy'].mean():.3f}, "
                  f"unchanged110 acc {k['accuracy'].mean():.3f}", flush=True)

            # Final fit on all 120 samples -> test predictions.
            fitted = clone(model).fit(train.X, y_fit)
            t = sample_table(predict_scores(fitted, test.X, CLASSES), test.sample_ids,
                             np.zeros(len(test.sample_ids), dtype=int))
            sub = pd.DataFrame({"filename": [f"{s}.csv" for s in t.index],
                                "predicted_label": t["pred"].astype(int).to_numpy()})
            validate_submission(sub, cfg)
            path = cfg.submissions_dir / f"predictions_{name}_g{args.n_groups}_{label_set}_labels.csv"
            sub.to_csv(path, index=False)
            submissions[(name, label_set)] = sub.set_index("filename")["predicted_label"]

    table = pd.DataFrame(summary)
    table.to_csv(cfg.metrics_dir / f"corrected_labels_g{args.n_groups}.csv", index=False)
    cols = ["model", "labels", "all120_accuracy", "all120_balanced_accuracy", "all120_f1_macro",
            "unchanged110_accuracy", "unchanged110_accuracy_std", "unchanged110_balanced_accuracy",
            "unchanged110_f1_macro"]
    print()
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    comp = pd.DataFrame({f"{m}_{ls}": s for (m, ls), s in submissions.items()})
    comp.to_csv(cfg.metrics_dir / f"test_predictions_corrected_labels_g{args.n_groups}.csv")
    print("\ntest set (60 samples):")
    for name in args.models.split(","):
        o, c = comp[f"{name}_original"], comp[f"{name}_corrected"]
        print(f"  {name}: {int((o != c).sum())} predictions change; class counts original "
              f"{np.bincount(o, minlength=6)[1:].tolist()} -> corrected {np.bincount(c, minlength=6)[1:].tolist()}")
    names = args.models.split(",")
    if len(names) == 2:
        for ls in ("original", "corrected"):
            agree = (comp[f"{names[0]}_{ls}"] == comp[f"{names[1]}_{ls}"]).mean()
            print(f"  {names[0]} vs {names[1]} agreement on test ({ls} labels): {agree:.2f}")


if __name__ == "__main__":
    main()
