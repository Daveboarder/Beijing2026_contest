"""Best pca_mlp pipeline with 10 depth blocks: per-row predictions and where the information sits.

    python scripts/27_pca_mlp_rows.py --n-repeats 10 --n-jobs 5

Pipeline (identical to the best ``pca_mlp`` except for the number of rows):
outlier-shot repair, no baseline, per-shot per-channel L2; 10 consecutive
blocks of 20 shots, block-mean spectrum per row; StandardScaler -> PCA(30) ->
StandardScaler -> MLP(128, 64); row probabilities averaged per sample;
StratifiedGroupKFold 5 folds grouped by sample, repeated.

Outputs
* every out-of-fold row with its own predicted class and the sample decision;
* depth importance: row accuracy of each block, accuracy when a sample is
  decided by one block only, and when one block is left out of the vote;
* spectral x depth importance: occlusion inside every fold -- a 10 nm window is
  set to the training mean (i.e. removed after scaling) in the test rows and
  the drop of the true-class probability is recorded per window and block.
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from libs2026 import Config, Preprocessor, build_features, get_model, plotting  # noqa: E402

CLASSES = np.array([1, 2, 3, 4, 5])


def _one_repeat(model, X, y, groups, windows, n_splits, seed):
    """Out-of-fold row probabilities and occlusion drops for one CV repeat."""
    proba = np.zeros((len(y), len(CLASSES)))
    drop = np.zeros((len(windows), len(y)), dtype=np.float32)
    true_col = np.searchsorted(CLASSES, y)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X, y, groups):
        fitted = clone(model).fit(X[tr], y[tr])
        cols = np.searchsorted(CLASSES, fitted.classes_)
        base = np.zeros((len(te), len(CLASSES)))
        base[:, cols] = fitted.predict_proba(X[te])
        proba[te] = base
        mean = fitted.named_steps["scale"].mean_
        for w, pixels in enumerate(windows):
            occluded = X[te].copy()
            occluded[:, pixels] = mean[pixels]
            p = np.zeros_like(base)
            p[:, cols] = fitted.predict_proba(occluded)
            drop[w, te] = base[np.arange(len(te)), true_col[te]] - p[np.arange(len(te)), true_col[te]]
    return proba, drop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--window-nm", type=float, default=10.0)
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument("--tag", default="pca_mlp_g10")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv = cfg["cv"]
    pre = Preprocessor.from_config(cfg)
    pre.normalization_reference = "shot"
    features = build_features(cfg, pre, n_groups=args.n_groups, bin_factor=1, encoding="mean",
                              augment="blocks", n_jobs=8)
    train = features.subset("train")
    X, y, groups = train.X, train.y.astype(int), train.groups
    wavelength = np.asarray(features.wavelength, dtype=float)
    n_shots = cfg["data"]["n_shots"]
    shot_blocks = np.array_split(np.arange(1, n_shots + 1), args.n_groups)
    # Rows of one sample are consecutive, in depth order.
    block = np.concatenate([np.arange(args.n_groups)] * (len(y) // args.n_groups))
    edges = np.arange(np.floor(wavelength.min()), wavelength.max() + args.window_nm, args.window_nm)
    windows = [np.flatnonzero((wavelength >= a) & (wavelength < b)) for a, b in zip(edges[:-1], edges[1:])]
    keep = [i for i, w in enumerate(windows) if w.size]
    windows, win_lo = [windows[i] for i in keep], edges[:-1][keep]
    print(f"rows {X.shape}; {len(windows)} occlusion windows of {args.window_nm:g} nm")

    out = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_one_repeat)(get_model("pca_mlp"), X, y, groups, windows, cv["n_splits"],
                             cv["random_state"] + r)
        for r in range(args.n_repeats)
    )

    # ---- per-row table ----------------------------------------------------------------
    frames, sample_metrics = [], []
    for r, (proba, _) in enumerate(out):
        f = pd.DataFrame(proba, columns=[f"p{c}" for c in CLASSES])
        f.insert(0, "row_pred", CLASSES[proba.argmax(1)])
        f.insert(0, "shots", [f"{shot_blocks[b][0]}-{shot_blocks[b][-1]}" for b in block])
        f.insert(0, "block", block + 1)
        f.insert(0, "y_true", y)
        f.insert(0, "sample_id", train.sample_ids)
        f.insert(0, "repeat", r)
        sample_p = f.groupby("sample_id")[[f"p{c}" for c in CLASSES]].transform("mean").to_numpy()
        f["sample_pred"] = CLASSES[sample_p.argmax(1)]
        frames.append(f)
        s = f.drop_duplicates("sample_id")
        sample_metrics.append({"accuracy": accuracy_score(s["y_true"], s["sample_pred"]),
                               "balanced_accuracy": recall_score(s["y_true"], s["sample_pred"], average="macro"),
                               "f1_macro": f1_score(s["y_true"], s["sample_pred"], average="macro")})
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(cfg.predictions_dir / f"rows_{args.tag}.csv", index=False)
    sm = pd.DataFrame(sample_metrics)
    print(f"\nsample level ({args.n_repeats} repeats): accuracy {sm['accuracy'].mean():.3f} +- "
          f"{sm['accuracy'].std(ddof=0):.3f}, balanced {sm['balanced_accuracy'].mean():.3f}, "
          f"macro-F1 {sm['f1_macro'].mean():.3f}")

    # Wide view: modal row prediction over repeats, one column per block.
    modal = rows.groupby(["sample_id", "block"])["row_pred"].agg(lambda v: v.value_counts().idxmax())
    wide = modal.unstack("block")
    wide.columns = [f"b{b} ({shot_blocks[b - 1][0]}-{shot_blocks[b - 1][-1]})" for b in wide.columns]
    info = rows.groupby("sample_id").agg(y_true=("y_true", "first"),
                                         sample_pred=("sample_pred", lambda v: v.value_counts().idxmax()),
                                         sample_correct=("sample_pred", lambda v: 0))
    info["sample_correct"] = rows.assign(ok=rows["sample_pred"] == rows["y_true"]) \
        .drop_duplicates(["repeat", "sample_id"]).groupby("sample_id")["ok"].mean()
    wide = info.join(wide).sort_values(["y_true", "sample_id"])
    wide.to_csv(cfg.metrics_dir / f"rows_{args.tag}_by_sample.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print("\nmodal predicted class per row (columns = depth blocks); sample_correct = share of repeats")
        print(wide.to_string(float_format=lambda v: f"{v:.1f}"))

    # ---- depth importance ------------------------------------------------------------
    rows["row_ok"] = rows["row_pred"] == rows["y_true"]
    depth = rows.groupby("block")["row_ok"].mean().rename("row_accuracy").to_frame()
    depth["shots"] = [f"{s[0]}-{s[-1]}" for s in shot_blocks]
    depth_by_class = rows.pivot_table(index="block", columns="y_true", values="row_ok", aggfunc="mean")
    only, without = [], []
    for b in range(1, args.n_groups + 1):
        acc_only, acc_wo = [], []
        for _, f in rows.groupby("repeat"):
            p_cols = [f"p{c}" for c in CLASSES]
            one = f[f["block"] == b].set_index("sample_id")
            acc_only.append(accuracy_score(one["y_true"], CLASSES[one[p_cols].to_numpy().argmax(1)]))
            rest = f[f["block"] != b].groupby("sample_id")
            mean = rest[p_cols].mean()
            acc_wo.append(accuracy_score(rest["y_true"].first(), CLASSES[mean.to_numpy().argmax(1)]))
        only.append(np.mean(acc_only))
        without.append(np.mean(acc_wo))
    depth["sample_acc_this_block_only"] = only
    depth["sample_acc_without_block"] = without
    depth["delta_without_block"] = depth["sample_acc_without_block"] - sm["accuracy"].mean()
    depth = depth.join(depth_by_class.add_prefix("row_acc_level"))
    depth.to_csv(cfg.metrics_dir / f"depth_importance_{args.tag}.csv")
    print("\ndepth importance")
    print(depth.to_string(float_format=lambda v: f"{v:.3f}"))

    # ---- spectral x depth importance (occlusion) ------------------------------------------
    drops = np.mean([d for _, d in out], axis=0)            # (windows, rows)
    heat = np.stack([drops[:, block == b].mean(axis=1) for b in range(args.n_groups)], axis=1)
    occ = pd.DataFrame(heat, columns=[f"b{b + 1}" for b in range(args.n_groups)])
    occ.insert(0, "wl_from", win_lo)
    occ.insert(1, "wl_to", win_lo + args.window_nm)
    occ["mean_all_blocks"] = heat.mean(axis=1)
    occ.to_csv(cfg.metrics_dir / f"occlusion_{args.tag}.csv", index=False)
    top = occ.sort_values("mean_all_blocks", ascending=False).head(12)
    print("\nmost informative spectral windows (mean drop of true-class probability when removed)")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- figures -------------------------------------------------------------------------
    figs = cfg.figures_dir
    blocks_cols = [c for c in wide.columns if c.startswith("b")]
    mat = wide[blocks_cols].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 16), gridspec_kw={"width_ratios": [10, 1.6]})
    cmap = ListedColormap(plotting.CLASS_COLORS)
    axes[0].imshow(mat, aspect="auto", cmap=cmap, vmin=0.5, vmax=5.5, interpolation="nearest")
    wrong = mat != wide["y_true"].to_numpy()[:, None]
    yy, xx = np.nonzero(wrong)
    axes[0].scatter(xx, yy, marker="x", s=10, color="red", linewidths=0.7)
    axes[0].set_xticks(range(len(blocks_cols)), [c.split(" ")[1].strip("()") for c in blocks_cols], rotation=45)
    axes[0].set_yticks(range(len(wide)), [f"{s} (L{t})" for s, t in zip(wide.index, wide["y_true"])], fontsize=5)
    axes[0].set_xlabel("depth block (shots)")
    axes[0].set_title("modal predicted class per row (red x = differs from true level)")
    side = np.c_[wide["y_true"], wide["sample_pred"]].astype(float)
    axes[1].imshow(side, aspect="auto", cmap=cmap, vmin=0.5, vmax=5.5, interpolation="nearest")
    axes[1].set_xticks([0, 1], ["true", "sample\npred"])
    axes[1].set_yticks([])
    for i, t in enumerate(wide["y_true"].to_numpy()):
        if i and t != wide["y_true"].to_numpy()[i - 1]:
            for ax in axes:
                ax.axhline(i - 0.5, color="black", lw=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in plotting.CLASS_COLORS]
    axes[0].legend(handles, [f"level {c}" for c in CLASSES], ncol=5, fontsize=8, loc="upper center",
                   bbox_to_anchor=(0.5, -0.03))
    plotting.save(fig, figs / f"rows_{args.tag}_predictions.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(1, args.n_groups + 1)
    axes[0].plot(x, depth["row_accuracy"], marker="o", color="black", label="single row (all levels)")
    for c, color in zip(CLASSES, plotting.CLASS_COLORS):
        axes[0].plot(x, depth[f"row_acc_level{c}"], marker=".", color=color, lw=1, label=f"level {c}")
    axes[0].set_ylabel("row accuracy")
    axes[1].plot(x, depth["sample_acc_this_block_only"], marker="o", label="decided by this block only")
    axes[1].plot(x, depth["sample_acc_without_block"], marker="s", label="vote without this block")
    axes[1].axhline(sm["accuracy"].mean(), color="grey", ls="--", label="all 10 blocks")
    axes[1].set_ylabel("sample accuracy")
    for ax in axes:
        ax.set_xticks(x, depth["shots"], rotation=45)
        ax.set_xlabel("depth block (shots)")
        ax.legend(fontsize=7)
    fig.suptitle("Which depth carries the information")
    fig.tight_layout()
    plotting.save(fig, figs / f"depth_importance_{args.tag}.png")

    fig, ax = plt.subplots(figsize=(15, 5))
    lim = np.abs(heat).max()
    im = ax.imshow(heat.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower",
                   extent=[edges[0], edges[0] + len(win_lo) * args.window_nm, 0.5, args.n_groups + 0.5])
    ax.set_yticks(x, [f"b{b} ({s})" for b, s in zip(x, depth["shots"])], fontsize=8)
    ax.set_xlabel(f"occluded wavelength window ({args.window_nm:g} nm)")
    ax.set_ylabel("depth block")
    fig.colorbar(im, ax=ax, label="drop of true-class probability")
    ax.set_title("Spectral x depth importance by occlusion (red = removing it hurts)")
    plotting.save(fig, figs / f"occlusion_{args.tag}.png")
    print(f"\nfigures -> {figs}")


if __name__ == "__main__":
    main()
