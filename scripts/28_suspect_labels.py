"""Retrain without the surface block and the "always wrong" samples, then re-predict those samples.

    python scripts/28_suspect_labels.py --n-repeats 10 --n-jobs 5

Pipeline: best pca_mlp with 10 depth blocks of 20 shots (scripts/27), but the
first block (shots 1-20) is dropped, leaving 9 rows per sample.

1. CV on all 120 samples without block 1 -- isolates the effect of dropping the
   surface block (compare with 0.752 for all 10 blocks).
2. CV on the 109 remaining samples ("clean" set) -- the scores are optimistic
   by construction, since the hardest samples were removed.
3. The 11 suspect samples are never trained on. They are predicted by every
   CV fold model of step 2 (n_repeats x 5 models, each trained on ~80 % of the
   clean set) and by models fitted on the whole clean set with different MLP
   seeds. For each suspect the file records how consistently the models agree,
   which class they pick, and how much probability they leave for the given
   label -- next to the same statistics for correctly classified clean samples
   (out-of-fold), as a reference for what an unambiguous sample looks like.

Consistent, confident disagreement with the given label supports a labelling
(or sample/measurement) problem; it cannot prove it. That needs a check of the
physical samples.
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from libs2026 import Config, Preprocessor, build_features, get_model, plotting  # noqa: E402
from libs2026.evaluation import predict_scores  # noqa: E402

CLASSES = np.array([1, 2, 3, 4, 5])
PCOLS = [f"p{c}" for c in CLASSES]
SUSPECTS = ["train_022", "train_042", "train_050", "train_078", "train_079", "train_080",
            "train_083", "train_088", "train_098", "train_101", "train_113"]


def aligned_proba(model, X):
    """Class scores summing to one; SVMs without predict_proba go through a decision softmax."""
    return predict_scores(model, X, CLASSES)


def _repeat(model, X, y, groups, X_extra, n_splits, seed):
    """OOF row probabilities on (X, y) and every fold model's probabilities on X_extra."""
    oof = np.zeros((len(y), len(CLASSES)))
    extra = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X, y, groups):
        fitted = clone(model).fit(X[tr], y[tr])
        oof[te] = aligned_proba(fitted, X[te])
        if X_extra is not None:
            extra.append(aligned_proba(fitted, X_extra))
    return oof, extra


def _full_fit(model, X, y, X_extra, seed):
    fitted = clone(model)
    if "random_state" in fitted.named_steps["clf"].get_params():
        fitted.set_params(clf__random_state=seed)
    fitted.fit(X, y)
    return aligned_proba(fitted, X_extra)


def sample_scores(frame):
    """Per-repeat sample-level metrics from row probabilities."""
    out = []
    for _, f in frame.groupby("repeat"):
        s = f.groupby("sample_id").agg(y_true=("y_true", "first"), **{c: (c, "mean") for c in PCOLS})
        pred = CLASSES[s[PCOLS].to_numpy().argmax(1)]
        out.append({"accuracy": accuracy_score(s["y_true"], pred),
                    "balanced_accuracy": recall_score(s["y_true"], pred, average="macro"),
                    "f1_macro": f1_score(s["y_true"], pred, average="macro")})
    per = pd.DataFrame(out)
    return {**per.mean().to_dict(), "accuracy_std": per["accuracy"].std(ddof=0)}


def run_cv(model, X, y, groups, sample_ids, block, X_extra, n_repeats, cv, n_jobs):
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_repeat)(model, X, y, groups, X_extra, cv["n_splits"], cv["random_state"] + r)
        for r in range(n_repeats)
    )
    frames = []
    for r, (oof, _) in enumerate(res):
        f = pd.DataFrame(oof, columns=PCOLS)
        f.insert(0, "block", block)
        f.insert(0, "y_true", y)
        f.insert(0, "sample_id", sample_ids)
        f.insert(0, "repeat", r)
        frames.append(f)
    return pd.concat(frames, ignore_index=True), [e for _, extra in res for e in extra]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--drop-blocks", default="1", help="1-based blocks to omit")
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-full-seeds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument("--model", default="pca_mlp",
                        help="model zoo name, e.g. pca_mlp, pca_svm_rbf, svm_linear")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if args.tag is None:
        blocks = f"_noB{args.drop_blocks.replace(',', '-')}" if args.drop_blocks else ""
        args.tag = f"{args.model}_g{args.n_groups}{blocks}"
    # Deterministic classifiers (SVMs) give identical full fits, so one is enough.
    if args.model != "pca_mlp":
        args.n_full_seeds = min(args.n_full_seeds, 1)

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv = cfg["cv"]
    pre = Preprocessor.from_config(cfg)
    pre.normalization_reference = "shot"
    features = build_features(cfg, pre, n_groups=args.n_groups, bin_factor=1, encoding="mean",
                              augment="blocks", n_jobs=8)
    train = features.subset("train")
    n_shots = cfg["data"]["n_shots"]
    shot_blocks = np.array_split(np.arange(1, n_shots + 1), args.n_groups)
    block_all = np.tile(np.arange(1, args.n_groups + 1), len(train.y) // args.n_groups)
    dropped = [int(b) for b in args.drop_blocks.split(",")] if args.drop_blocks else []
    keep_rows = ~np.isin(block_all, dropped)
    X, y = train.X[keep_rows], train.y[keep_rows].astype(int)
    ids, groups, block = train.sample_ids[keep_rows], train.groups[keep_rows], block_all[keep_rows]
    suspect = np.isin(ids, SUSPECTS)
    model = get_model(args.model)
    print(f"rows per sample: {args.n_groups - len(dropped)} (dropped blocks {dropped}); "
          f"clean samples {len(np.unique(ids[~suspect]))}, suspects {len(np.unique(ids[suspect]))}")

    # ---- 1. all 120 samples without the dropped blocks --------------------------------
    rows_all, _ = run_cv(model, X, y, groups, ids, block, None, args.n_repeats, cv, args.n_jobs)
    # ---- 2. clean set, with fold models also predicting the suspects --------------------
    rows_clean, extra = run_cv(model, X[~suspect], y[~suspect], groups[~suspect], ids[~suspect],
                               block[~suspect], X[suspect], args.n_repeats, cv, args.n_jobs)
    rows_clean.to_csv(cfg.predictions_dir / f"rows_{args.tag}_clean.csv", index=False)
    full = Parallel(n_jobs=args.n_jobs)(
        delayed(_full_fit)(model, X[~suspect], y[~suspect], X[suspect], seed)
        for seed in range(args.n_full_seeds)
    )

    summary = pd.DataFrame([
        {"setting": f"120 samples, blocks {sorted(set(block))[0]}-{args.n_groups}", "n_samples": 120,
         **sample_scores(rows_all)},
        {"setting": "109 clean samples (suspects removed; optimistic)", "n_samples": 109,
         **sample_scores(rows_clean)},
    ])
    summary.to_csv(cfg.metrics_dir / f"cv_{args.tag}.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- 3. suspects: every model x row ------------------------------------------------
    s_ids, s_y, s_block = ids[suspect], y[suspect], block[suspect]
    records = []
    for m, (kind, proba) in enumerate([("cv_fold", p) for p in extra] + [("full_fit", p) for p in full]):
        f = pd.DataFrame(proba, columns=PCOLS)
        f.insert(0, "row_pred", CLASSES[proba.argmax(1)])
        f.insert(0, "shots", [f"{shot_blocks[b - 1][0]}-{shot_blocks[b - 1][-1]}" for b in s_block])
        f.insert(0, "block", s_block)
        f.insert(0, "given_label", s_y)
        f.insert(0, "sample_id", s_ids)
        f.insert(0, "model", f"{kind}_{m}")
        records.append(f)
    s_rows = pd.concat(records, ignore_index=True)
    s_rows.to_csv(cfg.predictions_dir / f"suspect_labels_rows_{args.tag}.csv", index=False)

    per_model = s_rows.groupby(["model", "sample_id"]).agg(
        given_label=("given_label", "first"), **{c: (c, "mean") for c in PCOLS}).reset_index()
    per_model["sample_pred"] = CLASSES[per_model[PCOLS].to_numpy().argmax(1)]
    modal_rows = s_rows.groupby(["sample_id", "block"])["row_pred"].agg(lambda v: v.value_counts().idxmax())

    # Reference: clean samples, out-of-fold, per repeat.
    ref = rows_clean.groupby(["repeat", "sample_id"]).agg(
        y_true=("y_true", "first"), **{c: (c, "mean") for c in PCOLS}).reset_index()
    ref["pred"] = CLASSES[ref[PCOLS].to_numpy().argmax(1)]
    ref["p_label"] = ref[PCOLS].to_numpy()[np.arange(len(ref)), np.searchsorted(CLASSES, ref["y_true"])]
    ref["p_max"] = ref[PCOLS].max(axis=1)
    right = ref[ref["pred"] == ref["y_true"]]
    ref_line = {"sample_id": "REFERENCE: clean samples predicted correctly (OOF)", "given_label": np.nan,
                "mean_p_given_label": right["p_label"].mean(), "mean_p_predicted": right["p_max"].mean()}

    table = []
    for sid, g in per_model.groupby("sample_id"):
        label = int(g["given_label"].iloc[0])
        counts = g["sample_pred"].value_counts()
        top = int(counts.index[0])
        mean_p = g[PCOLS].mean()
        rows_right = int((modal_rows.loc[sid] == label).sum())
        table.append({
            "sample_id": sid, "given_label": label, "predicted_label": top,
            "agreement": counts.iloc[0] / len(g),
            "share_models_predicting_given_label": (g["sample_pred"] == label).mean(),
            "mean_p_given_label": mean_p[f"p{label}"], "mean_p_predicted": mean_p[f"p{top}"],
            "label_distance": abs(top - label),
            "rows_supporting_given_label": f"{rows_right}/{modal_rows.loc[sid].size}",
            "modal_row_classes": " ".join(str(int(v)) for v in modal_rows.loc[sid].to_numpy()),
            **{f"votes_L{c}": int((g["sample_pred"] == c).sum()) for c in CLASSES},
            **{f"mean_p{c}": mean_p[f"p{c}"] for c in CLASSES},
            "n_models": len(g),
        })
    table = pd.DataFrame(table).sort_values(["share_models_predicting_given_label", "sample_id"])
    table["verdict"] = np.select(
        [(table["share_models_predicting_given_label"] < 0.1) & (table["agreement"] >= 0.8),
         table["share_models_predicting_given_label"] < 0.3],
        ["consistently predicted as another class", "mostly another class, less consistent"],
        default="given label plausible",
    )
    out = pd.concat([table, pd.DataFrame([ref_line])], ignore_index=True)
    path = cfg.metrics_dir / f"suspect_labels_{args.tag}.csv"
    out.to_csv(path, index=False)
    cols = ["sample_id", "given_label", "predicted_label", "agreement", "share_models_predicting_given_label",
            "mean_p_given_label", "mean_p_predicted", "label_distance", "rows_supporting_given_label",
            "modal_row_classes", "verdict"]
    with pd.option_context("display.width", 250):
        print(out[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"-> {path}")

    # ---- figure: mean class probabilities per suspect ------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(table))
    bottom = np.zeros(len(table))
    for c, color in zip(CLASSES, plotting.CLASS_COLORS):
        vals = table[f"mean_p{c}"].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, label=f"level {c}")
        bottom += vals
    for i, (lab, pred) in enumerate(zip(table["given_label"], table["predicted_label"])):
        ax.text(i, 1.02, f"given {lab}\npred {pred}", ha="center", fontsize=8)
    ax.set_xticks(x, table["sample_id"], rotation=45)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("mean class probability over all models")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title(f"Suspect samples: {args.model}, {int(table['n_models'].iloc[0])} models never trained on them "
                 f"(blocks {sorted(set(block))[0]}-{args.n_groups})")
    plotting.save(fig, cfg.figures_dir / f"suspect_labels_{args.tag}.png")


if __name__ == "__main__":
    main()
