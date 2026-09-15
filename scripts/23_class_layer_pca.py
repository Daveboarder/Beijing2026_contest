"""PCA over the depth layers of each aging level's mean profile, with outlier removal.

    python scripts/23_class_layer_pca.py

1. Every training sample is preprocessed (config defaults) and the samples of
   one aging level are averaged shot by shot: ``(200 layers, 12282)`` per level.
2. Per level, the 200 layer spectra are autoscaled and a PCA is fitted.
3. The 2 layers with the largest Mahalanobis distance in PC1-PC2 are flagged
   and highlighted, removed, and the PCA is refitted on the remaining 198.
4. Component signs are aligned to level 1 so loadings compare across levels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from libs2026 import Config, Preprocessor, load_index, load_shots, load_wavelength, plotting  # noqa: E402
from libs2026.preprocessing import normalize  # noqa: E402


def _preprocessed(cfg, pre, sample_id):
    return pre(load_shots(cfg, sample_id, mmap=False)).astype(np.float64)


def class_layer_means(cfg, pre, train, n_jobs):
    """``{level: (n_shots, n_wl)}`` shot-by-shot mean over the level's samples (cached)."""
    key = hashlib.md5(json.dumps(asdict(pre), sort_keys=True, default=str).encode()).hexdigest()[:10]
    path = cfg.cache_dir / f"class_layer_means_{key}.npz"
    if path.exists():
        blob = np.load(path)
        return {int(k[1:]): blob[k] for k in blob.files}
    means = {}
    for level, group in train.groupby("label"):
        blocks = Parallel(n_jobs=n_jobs, verbose=2)(
            delayed(_preprocessed)(cfg, pre, s) for s in group["sample_id"]
        )
        means[int(level)] = np.mean(blocks, axis=0)
        print(f"level {int(level)}: {len(blocks)} samples -> {means[int(level)].shape}")
    np.savez(path, **{f"L{k}": v for k, v in means.items()})
    return means


def fit_pca(spectra, n_components, scaling="auto"):
    """Centre (and for ``auto`` also scale) every pixel over the layers, then fit a PCA.

    ``auto`` gives weak and noisy pixels the same weight as strong lines;
    ``center`` keeps the intensity scale, which is the usual choice after SNV.
    """
    mean = spectra.mean(axis=0)
    std = spectra.std(axis=0) if scaling == "auto" else np.ones(spectra.shape[1])
    scaled = (spectra - mean) / np.where(std > 0, std, 1.0)
    pca = PCA(n_components=n_components, random_state=0).fit(scaled)
    return pca, pca.transform(scaled)


def mahalanobis_pc12(scores):
    """Squared Mahalanobis distance in PC1-PC2 (scores are centred and uncorrelated)."""
    var = scores[:, :2].var(axis=0, ddof=1)
    return (scores[:, :2] ** 2 / var).sum(axis=1)


def align_signs(pca, scores, reference):
    """Flip components to correlate positively with the reference loadings."""
    for k in range(pca.n_components_):
        if np.dot(pca.components_[k], reference[k]) < 0:
            pca.components_[k] *= -1
            scores[:, k] *= -1
    return pca, scores


def plot_scores(results, key, path, title):
    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4.3), squeeze=False)
    for ax, (level, res) in zip(axes[0], results.items()):
        scores, shots, pca = res[key]["scores"], res[key]["shots"], res[key]["pca"]
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=shots, cmap="viridis",
                        norm=LogNorm(1, 200), s=16, alpha=0.9)
        if key == "all":
            for i in res["outliers"]:
                ax.scatter(scores[i, 0], scores[i, 1], s=160, facecolors="none",
                           edgecolors="red", linewidths=2)
                ax.annotate(f"shot {shots[i]}", (scores[i, 0], scores[i, 1]), color="red",
                            fontsize=8, xytext=(6, -10), textcoords="offset points")
        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f} %)")
        ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f} %)")
        ax.set_title(f"level {level} (n={res['n_samples']}, {len(shots)} layers)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
    fig.colorbar(sc, ax=axes[0].tolist(), fraction=0.02, pad=0.01, label="shot number (depth)")
    fig.suptitle(title)
    return plotting.save(fig, path)


def plot_loadings(results, wavelength, bounds, path, how, n_show=3):
    n_ch = len(bounds) - 1
    fig, axes = plt.subplots(n_show, n_ch, figsize=(6 * n_ch, 2.9 * n_show), squeeze=False)
    for k in range(n_show):
        for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            ax = axes[k, j]
            for (level, res), color in zip(results.items(), plotting.CLASS_COLORS):
                pca = res["clean"]["pca"]
                ax.plot(wavelength[a:b], pca.components_[k, a:b], lw=0.6, color=color,
                        label=f"level {level} ({pca.explained_variance_ratio_[k] * 100:.1f} %)")
            ax.axhline(0, color="grey", lw=0.5)
            if j == 0:
                ax.set_ylabel(f"PC{k + 1} loading")
            if k == n_show - 1:
                ax.set_xlabel("wavelength (nm)")
            if k == 0:
                ax.set_title(f"channel {j + 1}")
        axes[k, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Loadings of PC1-PC3, {how} (signs aligned to level 1)")
    fig.tight_layout()
    return plotting.save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--n-outliers", type=int, default=2)
    parser.add_argument("--drop-shots", default=None,
                        help="comma-separated shot numbers to remove instead of the Mahalanobis pick, e.g. 1,2")
    parser.add_argument("--snv", action="store_true",
                        help="standard normal variate on every layer spectrum (per channel) before PCA")
    parser.add_argument("--scaling", choices=("auto", "center"), default=None,
                        help="pixel scaling before PCA; default: center with --snv, auto otherwise")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    bounds = tuple(cfg["data"]["channel_bounds"])
    wavelength = load_wavelength(cfg).astype(np.float64)
    index = load_index(cfg)
    train = index[index["split"] == "train"].reset_index(drop=True)
    pre = Preprocessor.from_config(cfg)
    means = class_layer_means(cfg, pre, train, args.n_jobs)
    counts = train["label"].astype(int).value_counts().to_dict()
    limit = stats.chi2.ppf(0.975, df=2)
    drop = [int(s) for s in args.drop_shots.split(",")] if args.drop_shots else None
    scaling = args.scaling or ("center" if args.snv else "auto")
    suffix = (f"_drop{'-'.join(map(str, drop))}" if drop else "") + ("_snv" if args.snv else "") \
        + ("" if scaling == ("center" if args.snv else "auto") else f"_{scaling}")
    how = (f"shots {', '.join(map(str, drop))} removed" if drop
           else f"{args.n_outliers} most outlying layers per level (Mahalanobis, PC1-PC2)")
    how += ", SNV" if args.snv else ""
    how += ", mean-centred" if scaling == "center" else ", autoscaled"
    if args.snv:
        # Per channel, like Preprocessor's normalisation: the channels differ in response.
        means = {k: normalize(v, method="snv", per_channel=True, bounds=bounds).astype(np.float64)
                 for k, v in means.items()}

    results, rows = {}, []
    reference = {}
    for level in sorted(means):
        spectra = means[level]
        shots = np.arange(1, spectra.shape[0] + 1)
        pca, scores = fit_pca(spectra, args.n_components, scaling)
        d2 = mahalanobis_pc12(scores)
        outliers = (np.array([s - 1 for s in drop]) if drop
                    else np.argsort(d2)[::-1][: args.n_outliers])
        keep = np.setdiff1d(np.arange(len(shots)), outliers)
        pca_c, scores_c = fit_pca(spectra[keep], args.n_components, scaling)

        if not reference:
            reference = {"all": pca.components_.copy(), "clean": pca_c.components_.copy()}
        pca, scores = align_signs(pca, scores, reference["all"])
        pca_c, scores_c = align_signs(pca_c, scores_c, reference["clean"])

        results[level] = {
            "n_samples": counts[level],
            "outliers": outliers,
            "all": {"pca": pca, "scores": scores, "shots": shots},
            "clean": {"pca": pca_c, "scores": scores_c, "shots": shots[keep]},
        }
        ev, ev_c = pca.explained_variance_ratio_, pca_c.explained_variance_ratio_
        for i in outliers:
            rows.append({"level": level, "shot": int(shots[i]), "d2": d2[i], "chi2_975": limit,
                         "PC1": scores[i, 0], "PC2": scores[i, 1],
                         **{f"ev_PC{k + 1}_before": ev[k] for k in range(3)},
                         **{f"ev_PC{k + 1}_after": ev_c[k] for k in range(3)}})
        peaks = [f"PC{k + 1}@{wavelength[np.argmax(np.abs(pca_c.components_[k]))]:.2f}nm" for k in range(3)]
        print(f"level {level} ({counts[level]} samples): removed shots "
              f"{', '.join(f'{shots[i]} (d2={d2[i]:.1f})' for i in outliers)} [chi2 97.5% = {limit:.2f}]; "
              f"EV PC1-3 {np.round(ev[:3] * 100, 1)} -> {np.round(ev_c[:3] * 100, 1)} %; "
              f"peak |loading| {', '.join(peaks)}")

        plotting.plot_depth_shot_pca(
            wavelength, pca_c.components_.T, scores_c, ev_c,
            cfg.figures_dir / f"class_layer_pca_level{level}{suffix}.png",
            sample_id=f"class mean, {counts[level]} samples, shots "
                      f"{', '.join(str(shots[i]) for i in sorted(outliers))} removed",
            label=level, channel_bounds=bounds, shots=shots[keep],
        )

    pd.DataFrame(rows).to_csv(cfg.metrics_dir / f"class_layer_pca_outliers{suffix}.csv", index=False)
    figs = cfg.figures_dir
    plot_scores(results, "all", figs / f"class_layer_pca_scores{suffix}.png",
                f"PCA of class-mean depth layers — {how}, in red")
    plot_scores(results, "clean", figs / f"class_layer_pca_scores_clean{suffix}.png",
                f"PCA refitted — {how}")
    plot_loadings(results, wavelength, bounds, figs / f"class_layer_pca_loadings{suffix}.png", how)
    print(f"figures -> {figs / 'class_layer_pca_*.png'}")


if __name__ == "__main__":
    main()
