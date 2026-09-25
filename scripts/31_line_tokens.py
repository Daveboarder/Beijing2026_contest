"""Select the informative spectral lines and extract them as tokens.

    python scripts/31_line_tokens.py --n-repeats 10 --n-jobs 10

Builds on scripts/27: the best pca_mlp (10 depth blocks of 20 shots, per-shot
L2, no baseline) and its 10 nm occlusion map, which showed *where* in the
spectrum the information sits. Here the 12282 pixels are replaced by
data-driven line tokens (src/libs2026/line_tokens.py) and the lines are ranked
individually:

1. tokens -- detect the lines of the training mean spectrum, assign them
   tentatively against the air line database, and describe every line in
   every depth block with 8 descriptors (area, height, width, shift,
   asymmetry, detected, continuum, continuum slope).
2. benchmark -- identical grouped CV splits for the full spectrum, all line
   tokens, and descriptor subsets (which part of a line carries the signal).
3. nested selection -- lines are ranked inside every training fold only
   (recursive elimination on inner-CV occlusion) and the K survivors are
   scored on the held-out fold; K random lines are the control. This is the
   honest accuracy of a K-line token set.
4. final ranking -- recursive elimination on all training samples, averaged
   over several inner-split seeds, plus out-of-fold occlusion per depth
   block. How often a line survived in the nested folds measures stability.

Outputs
* results/metrics/line_tokens_lines.csv      every detected line: geometry,
  assignment, importance, rank, stability, selected flag
* results/metrics/line_tokens_benchmark.csv  CV accuracy of each representation
* results/metrics/line_tokens_selection.csv  nested accuracy per line budget
* cache/line_tokens/selected_top{K}.npz      LineTokens of the selected lines
* results/figures/line_tokens_*.png
"""

import argparse
import time
import warnings
from functools import partial
from importlib import import_module

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.feature_selection import f_classif  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from libs2026 import Config, Preprocessor, build_features, get_model, plotting  # noqa: E402
from libs2026.evaluation import predict_scores  # noqa: E402
from libs2026.line_tokens import (  # noqa: E402
    DESCRIPTOR_NAMES,
    LineTokenConfig,
    build_line_tokens,
    eliminate_lines,
    line_occlusion,
    line_token_model,
    usable_ranges,
)

CLASSES = np.array([1, 2, 3, 4, 5])
SERIES = ("#2a78d6", "#eb6834")   # categorical slots 1-2: selected lines, random lines
INK, MUTED = "#0b0b0b", "#8a8984"
DESCRIPTOR_SETS = {
    "all descriptors": DESCRIPTOR_NAMES,
    "line only (no continuum)": ("area", "height", "width", "shift", "asymmetry", "detected"),
    "intensity (area, height)": ("area", "height"),
    "shape (width, shift, asymmetry)": ("width", "shift", "asymmetry"),
    "continuum (level, slope)": ("continuum", "continuum_slope"),
}


def sample_scores(proba, y, n_rows):
    """Metrics of one CV repeat after averaging the rows of each sample."""
    p = proba.reshape(-1, n_rows, len(CLASSES)).mean(axis=1)
    y_s, pred = y[::n_rows], CLASSES[p.argmax(axis=1)]
    return {"accuracy": accuracy_score(y_s, pred),
            "balanced_accuracy": balanced_accuracy_score(y_s, pred),
            "f1_macro": f1_score(y_s, pred, average="macro")}


def _folds(X, y, groups, n_splits, seed):
    return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups)


def _cv_proba(model, X, y, groups, n_splits, seed):
    proba = np.zeros((len(y), len(CLASSES)))
    for tr, te in _folds(X, y, groups, n_splits, seed):
        proba[te] = predict_scores(model.fit(X[tr], y[tr]), X[te], CLASSES)
    return proba


def _nested_repeat(X, y, groups, budgets, n_splits, seed):
    """One outer CV repeat: select lines on the training fold, score on the held-out fold."""
    n, n_lines, n_desc = X.shape
    rng = np.random.default_rng(seed)
    proba = {(kind, k): np.zeros((n, len(CLASSES)))
             for kind in ("selected", "random") for k in budgets}
    counts = {k: np.zeros(n_lines, dtype=int) for k in budgets}
    for tr, te in _folds(X, y, groups, n_splits, seed):
        subsets, _ = eliminate_lines(X[tr], y[tr], groups[tr], budgets, seed=seed)
        for k in budgets:
            counts[k][subsets[k]] += 1
            random_lines = np.sort(rng.permutation(n_lines)[:k])
            for kind, keep in (("selected", subsets[k]), ("random", random_lines)):
                model = line_token_model(k, n_desc).fit(X[tr][:, keep].reshape(len(tr), -1), y[tr])
                held = X[te][:, keep].reshape(len(te), -1)
                proba[(kind, k)][te] = predict_scores(model, held, CLASSES)
    return proba, counts


def _label(row):
    name = row["assignment"] if row["assignment"] != "?" else "unassigned"
    return f"{name} ({row['wavelength_nm']:.2f})"


# ---- figures -------------------------------------------------------------------------------


def plot_spectrum(path, wavelength, mean, bounds, lines, n_annotate=30):
    ranges = usable_ranges(wavelength, bounds)
    fig, axes = plt.subplots(3, 1, figsize=(18, 11))
    top = set(lines.nsmallest(n_annotate, "rank")["line_id"])
    for c, (ax, (a, b), (lo, hi)) in enumerate(zip(axes, zip(bounds[:-1], bounds[1:]), ranges)):
        m = (wavelength[a:b] >= lo) & (wavelength[a:b] < hi)
        ax.plot(wavelength[a:b][m], mean[a:b][m], color=INK, lw=0.5)
        ch = lines[lines["channel"] == c + 1]
        ax.plot(ch["wavelength_nm"], np.full(len(ch), 0.02), "|", color=MUTED, ms=8,
                transform=ax.get_xaxis_transform(), label="detected line")
        sel = ch[ch["selected"]]
        peak = mean[sel["peak_px"].to_numpy()]
        ax.plot(sel["wavelength_nm"], peak * 1.04, "v", color=SERIES[0], ms=5,
                label="selected line")
        for (_, row), h in zip(sel.iterrows(), peak):
            if row["line_id"] in top:
                ax.annotate(f"{row['assignment']} #{int(row['rank'])}",
                            (row["wavelength_nm"], h * 1.08), rotation=90, fontsize=6.5,
                            color=INK, ha="center", va="bottom")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, float(mean[a:b][m].max()) * 1.45)
        ax.set_ylabel(f"channel {c + 1}\nmean intensity (a.u.)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    axes[-1].set_xlabel("wavelength (nm)")
    axes[0].set_title(f"Training mean spectrum with the {int(lines['selected'].sum())} selected "
                      f"lines (labels: tentative assignment and rank, top {n_annotate})")
    plotting.save(fig, path)


def plot_selection(path, bench, curve, full_acc, all_acc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    order = bench.sort_values("accuracy")
    colors = [MUTED if r.startswith("full spectrum") else SERIES[0]
              for r in order["representation"]]
    axes[0].barh(order["representation"], order["accuracy"], xerr=order["accuracy_std"],
                 color=colors, height=0.6, error_kw={"lw": 1, "ecolor": INK})
    for y, (acc, feats) in enumerate(zip(order["accuracy"], order["n_features"])):
        axes[0].text(0.01, y, f"{acc:.3f}  ({feats} features)", va="center", fontsize=8,
                     color="white")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("sample accuracy (grouped CV)")
    axes[0].set_title("What part of a line carries the signal")
    ax = axes[1]
    for kind, color, name in (("selected", SERIES[0], "lines ranked inside each training fold"),
                              ("random", SERIES[1], "random lines (control)")):
        c = curve[curve["selection"] == kind]
        ax.errorbar(c["n_lines"], c["accuracy"], yerr=c["accuracy_std"], color=color, marker="o",
                    ms=5, lw=2, capsize=3, label=name)
    n_total = curve["n_lines_total"].iloc[0]
    for value, text in ((full_acc, "full spectrum"), (all_acc, f"all {n_total} lines")):
        ax.axhline(value, color=MUTED, ls="--", lw=1)
        ax.text(curve["n_lines"].min(), value + 0.004, text, fontsize=8, color=INK)
    ticks = sorted(curve["n_lines"].unique())
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(k) for k in ticks])
    ax.set_xlabel("number of lines in the token set")
    ax.set_ylabel("sample accuracy (nested CV)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Accuracy of a K-line token set")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plotting.save(fig, path)


def plot_depth_importance(path, sel, block_cols, shots):
    heat = sel[block_cols].to_numpy()
    lim = float(np.abs(heat).max()) or 1.0
    fig, ax = plt.subplots(figsize=(9, 0.16 * len(sel) + 2))
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    ax.set_yticks(range(len(sel)), [_label(r) for _, r in sel.iterrows()], fontsize=6)
    ax.set_xticks(range(len(block_cols)), shots, rotation=45, fontsize=8)
    ax.set_xlabel("depth block (shots)")
    fig.colorbar(im, ax=ax, label="drop of true-class probability", fraction=0.04)
    ax.set_title("Out-of-fold line occlusion by depth (red = removing the line hurts)")
    plotting.save(fig, path)


def plot_profiles(path, tokens_train, y, top, shots):
    cols, n = 4, len(top)
    n_panel_rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(n_panel_rows, cols, figsize=(16, 3.1 * n_panel_rows), squeeze=False)
    x = np.arange(tokens_train.shape[1])
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        d = DESCRIPTOR_NAMES.index(row["best_descriptor"])
        values = tokens_train[:, :, int(row["token_index"]), d]
        for c, color in zip(CLASSES, plotting.CLASS_COLORS):
            ax.plot(x, values[y == c].mean(axis=0), color=color, lw=2, marker="o", ms=3,
                    label=f"level {c}")
        ax.set_title(f"#{int(row['rank'])} {_label(row)}\n{row['best_descriptor']}", fontsize=8)
        ax.set_xticks(x[::3], shots[::3], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Depth profile of the most separating descriptor of the top-ranked lines, "
                 "mean per aging level")
    fig.tight_layout()
    plotting.save(fig, path)


# ---- main ----------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--budgets", default="10,20,40,80,150,250",
                        help="line counts scored in the nested selection")
    parser.add_argument("--n-select", type=int, default=None,
                        help="lines in the exported token set (default: line_tokens.n_select)")
    parser.add_argument("--rank-seeds", type=int, default=5,
                        help="inner-split seeds averaged for the final ranking")
    parser.add_argument("--labels", default="original", choices=["original", "corrected"],
                        help="'corrected' applies the consensus corrections of scripts/29")
    parser.add_argument("--skip-nested", action="store_true",
                        help="skip the (slowest) nested selection")
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv = cfg["cv"]
    settings = LineTokenConfig.from_config(cfg)
    n_select = args.n_select or int(cfg.get("line_tokens", {}).get("n_select", 80))
    budgets = sorted({int(b) for b in args.budgets.split(",")} | {n_select})
    tag = "" if args.labels == "original" else "_corrected_labels"
    started = time.perf_counter()

    # ---- 1. tokens -------------------------------------------------------------------------
    pre = Preprocessor.from_config(cfg)
    pre.normalization_reference = "shot"
    tokens = build_line_tokens(cfg, settings, pre=pre, n_jobs=8)
    features = build_features(cfg, pre, n_groups=settings.n_groups, bin_factor=1,
                              encoding="mean", augment="blocks", n_jobs=8)
    train = tokens.subset("train")
    spectrum = features.subset("train")
    y_samples = train.y.astype(int)
    if args.labels == "corrected":
        corrections = import_module("29_corrected_labels").CORRECTIONS
        y_samples = np.array([corrections.get(s, (None, v))[1]
                              for s, v in zip(train.sample_ids, y_samples)])
    n_rows, n_lines = train.n_rows, train.n_lines
    if not np.array_equal(np.repeat(train.sample_ids, n_rows), spectrum.sample_ids):
        raise RuntimeError("token rows and spectrum rows are not aligned")
    X_tok = train.X.reshape(len(y_samples) * n_rows, n_lines, -1)
    y = np.repeat(y_samples, n_rows)
    groups = np.repeat(np.arange(len(y_samples)), n_rows)
    budgets = [b for b in budgets if b < n_lines]
    seeds = [cv["random_state"] + r for r in range(args.n_repeats)]
    lines = train.lines.copy()
    lines["token_index"] = np.arange(n_lines)
    print(f"{train}; labels: {args.labels}; budgets {budgets}; exporting top {n_select}")

    # ---- 2. benchmark ----------------------------------------------------------------------
    reps = {"full spectrum (12282 px)": (spectrum.X, partial(get_model, "pca_mlp"))}
    for name, names in DESCRIPTOR_SETS.items():
        idx = [DESCRIPTOR_NAMES.index(d) for d in names]
        reps[f"{n_lines} lines: {name}"] = (X_tok[:, :, idx].reshape(len(y), -1),
                                            partial(line_token_model, n_lines, len(idx)))
    jobs = [(name, s) for name in reps for s in seeds]
    probas = Parallel(n_jobs=args.n_jobs)(
        delayed(_cv_proba)(reps[name][1](), reps[name][0], y, groups, cv["n_splits"], s)
        for name, s in jobs)
    bench = []
    for name in reps:
        per = pd.DataFrame([sample_scores(p, y, n_rows)
                            for (nm, _), p in zip(jobs, probas) if nm == name])
        bench.append({"representation": name, "n_features": reps[name][0].shape[1],
                      "n_repeats": len(per), **per.mean().to_dict(),
                      "accuracy_std": per["accuracy"].std(ddof=0)})
    bench = pd.DataFrame(bench)
    bench.to_csv(cfg.metrics_dir / f"line_tokens_benchmark{tag}.csv", index=False)
    print("\nbenchmark (sample level, identical grouped folds)")
    print(bench.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    representation = bench.set_index("representation")["accuracy"]
    full_acc = float(representation.filter(like="full spectrum").iloc[0])
    all_acc = float(representation.filter(like="all descriptors").iloc[0])
    print(f"[{time.perf_counter() - started:.0f}s]")

    # ---- 3. nested selection ---------------------------------------------------------------
    curve = None
    if not args.skip_nested:
        out = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_nested_repeat)(X_tok, y, groups, budgets, cv["n_splits"], s) for s in seeds)
        rows = []
        for kind in ("selected", "random"):
            for k in budgets:
                per = pd.DataFrame([sample_scores(p[(kind, k)], y, n_rows) for p, _ in out])
                rows.append({"selection": kind, "n_lines": k,
                             "n_features": k * len(DESCRIPTOR_NAMES), **per.mean().to_dict(),
                             "accuracy_std": per["accuracy"].std(ddof=0),
                             "n_lines_total": n_lines})
        curve = pd.DataFrame(rows)
        curve.to_csv(cfg.metrics_dir / f"line_tokens_selection{tag}.csv", index=False)
        print("\nnested selection (lines ranked on the training folds only)")
        table = curve.pivot(index="n_lines", columns="selection", values="accuracy")
        table = table.join(curve.pivot(index="n_lines", columns="selection",
                                       values="accuracy_std"), rsuffix="_std")
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))
        n_folds = len(seeds) * cv["n_splits"]
        for k in budgets:
            lines[f"selection_freq_top{k}"] = sum(c[k] for _, c in out) / n_folds
        print(f"[{time.perf_counter() - started:.0f}s]")

    # ---- 4. final ranking and depth importance ---------------------------------------------
    ranked = Parallel(n_jobs=args.n_jobs)(
        delayed(eliminate_lines)(X_tok, y, groups, budgets, seed=cv["random_state"] + 1000 + s)
        for s in range(args.rank_seeds))
    ranks = np.stack([r for _, r in ranked])
    lines["rank_mean"], lines["rank_sd"] = ranks.mean(axis=0), ranks.std(axis=0)
    lines["rank"] = lines["rank_mean"].rank(method="first").astype(int)
    lines["selected"] = lines["rank"] <= n_select

    occ = Parallel(n_jobs=args.n_jobs)(
        delayed(line_occlusion)(X_tok, y, groups, n_splits=cv["n_splits"], seed=s) for s in seeds)
    lines["occlusion"] = np.mean([imp for imp, _ in occ], axis=0)
    drop = np.mean([d for _, d in occ], axis=0)
    drop = drop.reshape(len(y_samples), n_rows, n_lines).mean(axis=0)
    shot_blocks = np.array_split(np.arange(1, cfg["data"]["n_shots"] + 1), n_rows)
    shots = [f"{b[0]}-{b[-1]}" for b in shot_blocks]
    block_cols = [f"occlusion_b{b + 1}" for b in range(n_rows)]
    lines[block_cols] = drop.T

    # Univariate context: which descriptor of a line tracks the aging level.
    sample_means = train.X.mean(axis=1)                                 # (samples, lines, desc)
    with warnings.catch_warnings():  # "detected" is constant (always 1) for strong lines
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        F, _ = f_classif(sample_means.reshape(len(y_samples), -1), y_samples)
    F = np.nan_to_num(F).reshape(n_lines, -1)
    lines["best_descriptor"] = [DESCRIPTOR_NAMES[i] for i in F.argmax(axis=1)]
    lines["best_F"] = F.max(axis=1)
    for d in ("area", "shift", "continuum"):
        i = DESCRIPTOR_NAMES.index(d)
        lines[f"rho_{d}"] = [spearmanr(sample_means[:, j, i], y_samples)[0]
                             for j in range(n_lines)]

    lines = lines.sort_values("rank")
    lines.to_csv(cfg.metrics_dir / f"line_tokens_lines{tag}.csv", index=False)
    export = tokens.select(lines["token_index"].to_numpy()[:n_select])
    export.lines = lines.iloc[:n_select].reset_index(drop=True)   # same order, with the rankings
    export_path = export.save(cfg.cache_dir / "line_tokens" / f"selected_top{n_select}{tag}.npz")

    cols = ["rank", "wavelength_nm", "channel", "assignment", "n_competing", "occlusion",
            "rank_sd", "best_descriptor", "best_F", "rho_area", "rho_continuum"]
    if f"selection_freq_top{n_select}" in lines:
        cols.insert(6, f"selection_freq_top{n_select}")
    shown = min(n_select, 40)
    with pd.option_context("display.width", 250, "display.max_rows", 100):
        print(f"\ntop {shown} lines (of {n_select} selected)")
        print(lines[cols].head(shown).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    chosen = lines[lines["selected"]]
    print(f"\nselected lines per (tentative) element: "
          f"{chosen['element'].replace('', '?').value_counts().to_dict()}")
    print(f"selected lines per channel: {chosen['channel'].value_counts().sort_index().to_dict()}")

    # Consistency with the 10 nm occlusion of the full-spectrum model (scripts/27).
    ref = cfg.metrics_dir / "occlusion_pca_mlp_g10.csv"
    if ref.exists():
        win = pd.read_csv(ref)
        inside = [(lines["wavelength_nm"] >= a) & (lines["wavelength_nm"] < b)
                  for a, b in zip(win["wl_from"], win["wl_to"])]
        rho = spearmanr([lines.loc[m, "occlusion"].sum() for m in inside],
                        win["mean_all_blocks"])[0]
        print(f"10 nm windows: Spearman between summed line occlusion and the pixel occlusion "
              f"of scripts/27: {rho:+.2f}")

    # ---- figures ---------------------------------------------------------------------------
    figs = cfg.figures_dir
    wavelength = np.asarray(features.wavelength, dtype=float)
    bounds = tuple(cfg["data"]["channel_bounds"])
    plot_spectrum(figs / f"line_tokens_spectrum{tag}.png", wavelength, spectrum.X.mean(axis=0),
                  bounds, lines)
    if curve is not None:
        plot_selection(figs / f"line_tokens_selection{tag}.png", bench, curve, full_acc, all_acc)
    plot_depth_importance(figs / f"line_tokens_depth_importance{tag}.png",
                          chosen.sort_values("wavelength_nm"), block_cols, shots)
    plot_profiles(figs / f"line_tokens_profiles{tag}.png", train.X, y_samples, lines.head(12),
                  shots)
    print(f"\nselected tokens -> {export_path}  {export}")
    print(f"figures -> {figs}  [{time.perf_counter() - started:.0f}s]")


if __name__ == "__main__":
    main()
