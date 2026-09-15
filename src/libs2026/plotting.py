"""Figures for exploratory analysis and model comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CLASS_COLORS = plt.get_cmap("viridis")(np.linspace(0, 0.9, 5))


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_mean_spectra(wavelength, X, y, path: Path, channel_bounds=None):
    """Mean spectrum per aging level, one panel per spectrometer channel."""
    bounds = channel_bounds or (0, len(wavelength))
    n_panels = len(bounds) - 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels))
    axes = np.atleast_1d(axes)
    classes = np.unique(y)
    for ax, (a, b) in zip(axes, zip(bounds[:-1], bounds[1:])):
        for c, color in zip(classes, CLASS_COLORS):
            ax.plot(wavelength[a:b], X[y == c, a:b].mean(axis=0),
                    lw=0.7, color=color, label=f"level {int(c)}")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("intensity (a.u.)")
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].set_title("Class-average spectra per spectrometer channel")
    return save(fig, path)


def plot_pca_scores(scores, y, explained, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    classes = np.unique(y)
    for c, color in zip(classes, CLASS_COLORS):
        m = y == c
        axes[0].scatter(scores[m, 0], scores[m, 1], s=28, color=color,
                        label=f"level {int(c)}", alpha=0.85)
        axes[1].scatter(scores[m, 1], scores[m, 2], s=28, color=color, alpha=0.85)
    axes[0].set_xlabel(f"PC1 ({explained[0]*100:.1f} %)")
    axes[0].set_ylabel(f"PC2 ({explained[1]*100:.1f} %)")
    axes[1].set_xlabel(f"PC2 ({explained[1]*100:.1f} %)")
    axes[1].set_ylabel(f"PC3 ({explained[2]*100:.1f} %)")
    axes[0].legend(fontsize=8)
    fig.suptitle("PCA of sample-average spectra")
    return save(fig, path)


def plot_confusion(matrix: np.ndarray, classes, path: Path, title: str = ""):
    normed = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), [int(c) for c in classes])
    ax.set_yticks(range(len(classes)), [int(c) for c in classes])
    ax.set_xlabel("predicted aging level")
    ax.set_ylabel("true aging level")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:d}", ha="center", va="center",
                    color="white" if normed[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised share")
    ax.set_title(title or "Out-of-fold confusion matrix")
    return save(fig, path)


def plot_leaderboard(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(summary) + 1.5))
    order = summary.sort_values("accuracy")
    ax.barh(order["model"], order["accuracy"], xerr=order["accuracy_std"],
            color="#4c72b0", height=0.6)
    ax.set_xlabel("cross-validated sample accuracy")
    ax.set_xlim(0, 1)
    ax.axvline(0.333, color="grey", ls="--", lw=1, label="majority-class baseline")
    ax.legend(fontsize=8)
    ax.set_title("Model comparison (repeated stratified group CV)")
    return save(fig, path)


def plot_depth_profiles(profiles: dict[str, np.ndarray], y: np.ndarray, path: Path,
                        bin_edges=None):
    """Class-average depth profiles: signal versus shot number, i.e. versus depth.

    Plotted on a log shot axis because the aged layer is consumed within the
    first few pulses while the bulk plateau spans the remaining ~150.
    """
    classes = np.unique(y)
    fig, axes = plt.subplots(1, len(profiles), figsize=(6 * len(profiles), 4.5),
                             squeeze=False)
    for ax, (label, values) in zip(axes[0], profiles.items()):
        shots = np.arange(1, values.shape[1] + 1)
        for c, color in zip(classes, CLASS_COLORS):
            ax.plot(shots, values[y == c].mean(axis=0), color=color, lw=1.4,
                    label=f"level {int(c)}")
        if bin_edges is not None:
            for sl in bin_edges[:-1]:
                ax.axvline(sl.stop + 0.5, color="grey", lw=0.5, alpha=0.4)
        ax.set_xscale("log")
        ax.axhline(1.0, color="black", lw=0.8, ls=":")
        ax.set_xlabel("shot number (increasing depth)")
        ax.set_ylabel("intensity / bulk")
        ax.set_title(label)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Depth profiles by aging level (vertical lines = depth bins)")
    return save(fig, path)


def plot_shot_variability(totals: dict[str, np.ndarray], path: Path):
    """Total intensity of each shot, for a few example samples."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, values in totals.items():
        ax.plot(values, lw=0.9, label=name)
    ax.set_xlabel("shot number")
    ax.set_ylabel("total intensity (a.u.)")
    ax.set_title("Shot-to-shot variability")
    ax.legend(fontsize=8, ncol=3)
    return save(fig, path)


def plot_depth_shot_pca(wavelength, loadings, scores, explained, path: Path,
                        sample_id: str = "", label=None, channel_bounds=None, shots=None):
    """PCA across shots of one sample: loadings + PC1–PC2 / PC1–PC3 by depth.

    ``loadings`` is ``(n_wavelengths, n_components)``, ``scores`` is
    ``(n_shots, n_components)``. Points are coloured by shot number (depth);
    pass ``shots`` when some shots were removed so the colours stay true.
    """
    bounds = channel_bounds or (0, len(wavelength))
    n_panels = len(bounds) - 1
    shots = np.arange(1, len(scores) + 1) if shots is None else np.asarray(shots)
    fig = plt.figure(figsize=(12, 10))
    # Loadings: one panel per spectrometer channel, PC1–3 overlaid.
    gs = fig.add_gridspec(n_panels + 1, 2, height_ratios=[1.1] * n_panels + [1.4],
                          hspace=0.35, wspace=0.28)
    colors = ("#4c72b0", "#dd8452", "#55a868")
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        ax = fig.add_subplot(gs[i, :])
        for k, color in enumerate(colors):
            ax.plot(wavelength[a:b], loadings[a:b, k], lw=0.7, color=color,
                    label=f"PC{k + 1} ({explained[k] * 100:.1f}%)")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_ylabel("loading")
        if i == 0:
            title = f"Depth-PCA loadings — {sample_id}"
            if label is not None:
                title += f" (aging level {int(label)})"
            ax.set_title(title)
            ax.legend(fontsize=8, ncol=3, loc="upper right")
        if i == n_panels - 1:
            ax.set_xlabel("wavelength (nm)")

    ax01 = fig.add_subplot(gs[-1, 0])
    sc = ax01.scatter(scores[:, 0], scores[:, 1], c=shots, cmap="viridis",
                      s=22, alpha=0.9)
    ax01.set_xlabel(f"PC1 ({explained[0] * 100:.1f} %)")
    ax01.set_ylabel(f"PC2 ({explained[1] * 100:.1f} %)")
    ax01.set_title("Scores: PC1 vs PC2 (colour = shot / depth)")
    fig.colorbar(sc, ax=ax01, fraction=0.046, label="shot number")

    ax02 = fig.add_subplot(gs[-1, 1])
    sc2 = ax02.scatter(scores[:, 0], scores[:, 2], c=shots, cmap="viridis",
                       s=22, alpha=0.9)
    ax02.set_xlabel(f"PC1 ({explained[0] * 100:.1f} %)")
    ax02.set_ylabel(f"PC3 ({explained[2] * 100:.1f} %)")
    ax02.set_title("Scores: PC1 vs PC3 (colour = shot / depth)")
    fig.colorbar(sc2, ax=ax02, fraction=0.046, label="shot number")
    return save(fig, path)


def _centred(y: np.ndarray) -> np.ndarray:
    """Remove each spectrum's mean level; slopes (temperatures) are unchanged."""
    return y - np.nanmean(y, axis=1, keepdims=True)


def plot_boltzmann_by_class(panels, labels: np.ndarray, path: Path, offset_step: float = 0.8):
    """Class-mean (Saha-)Boltzmann plots with the mean temperature of each level.

    ``panels`` holds ``(title, x, y, T_per_sample, lines)`` tuples; ``y`` is
    ``(n_samples, n_lines)`` and ``x`` either shared or per sample. Each
    spectrum is centred on its own mean before averaging, which removes
    intensity (number density) differences but leaves the slope untouched.
    Classes are offset vertically for legibility. The right-hand panel shows
    the per-sample temperatures behind each legend entry.
    """
    classes = np.unique(labels)
    fig, axes = plt.subplots(len(panels), 2, figsize=(14, 5.2 * len(panels)),
                             gridspec_kw={"width_ratios": [2.3, 1]}, squeeze=False)
    for (ax, ax_t), (title, x, y, temps, lines) in zip(axes, panels):
        x_line = np.asarray(x)[0] if np.ndim(x) == 2 else np.asarray(x)
        ionic = (lines["ion_state"].to_numpy() != "I")
        yc = _centred(y)
        for k, (c, color) in enumerate(zip(classes, CLASS_COLORS)):
            m = labels == c
            mean = np.nanmean(yc[m], axis=0) + k * offset_step
            sem = np.nanstd(yc[m], axis=0) / np.sqrt(np.isfinite(yc[m]).sum(axis=0))
            t = temps[m][np.isfinite(temps[m])]
            slope, intercept = np.polyfit(x_line[np.isfinite(mean)], mean[np.isfinite(mean)], 1)
            label = (f"level {int(c)} (n={m.sum()}): T = {t.mean():,.0f} ± {t.std(ddof=1) / np.sqrt(t.size):.0f} K"
                     f"  [fit of mean {-1 / (8.617333262e-5 * slope):,.0f} K]")
            for sel, marker in ((~ionic, "o"), (ionic, "^")):
                if sel.any():
                    ax.errorbar(x_line[sel], mean[sel], yerr=sem[sel], fmt=marker, ms=5,
                                color=color, capsize=2, lw=0.8)
            xx = np.linspace(x_line.min(), x_line.max(), 10)
            ax.plot(xx, intercept + slope * xx, color=color, lw=1.4, label=label)

            jitter = np.random.default_rng(int(c)).uniform(-0.18, 0.18, t.size)
            ax_t.scatter(np.full(t.size, c) + jitter, t, s=12, color=color, alpha=0.45)
            ax_t.errorbar(c, t.mean(), yerr=t.std(ddof=1) / np.sqrt(t.size), fmt="s", ms=8,
                          color="black", mfc=color, capsize=4, lw=1.5)
        xlabel = "E_k + E_ion (ionic lines), eV" if ionic.any() else "upper level energy E_k (eV)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"ln($I\lambda / g_k A_k$), centred  (+ offset per level)")
        n_i, n_ii = int((~ionic).sum()), int(ionic.sum())
        ax.set_title(f"{title}: {n_i} Fe I" + (f" (o) + {n_ii} Fe II (^)" if n_ii else "") + " lines")
        ax.legend(fontsize=8, loc="upper right")
        means = [temps[labels == c][np.isfinite(temps[labels == c])].mean() for c in classes]
        ax_t.plot(classes, means, color="grey", lw=1, ls="--", zorder=0)
        ax_t.set_xticks(classes, [int(c) for c in classes])
        ax_t.set_xlabel("aging level")
        ax_t.set_ylabel("temperature (K)")
        lo, hi = np.nanpercentile(temps, [2, 98])
        ax_t.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax_t.set_title("per-sample T, mean ± SEM")
    fig.suptitle("Fe plasma temperature by aging level (all 200 shots, training samples)")
    fig.tight_layout()
    return save(fig, path)


def plot_temperature_by_class(samples: pd.DataFrame, depth: pd.DataFrame, path: Path):
    """Temperature versus depth per aging level, plus the H-alpha electron density."""
    columns = [c for c in ("T_boltz", "T_saha") if c in depth]
    titles = {"T_boltz": "Boltzmann T (Fe I)", "T_saha": "Saha-Boltzmann T (Fe I + II)"}
    fig, axes = plt.subplots(1, len(columns) + 1, figsize=(5.5 * (len(columns) + 1), 4.5))
    classes = np.sort(samples["label"].astype(int).unique())
    mid = depth.groupby("depth_bin")[["shot_start", "shot_stop"]].first()
    mid = np.sqrt(mid["shot_start"] * mid["shot_stop"])
    for ax, col in zip(axes, columns):
        for c, color in zip(classes, CLASS_COLORS):
            g = depth[depth["label"].astype(int) == c].groupby("depth_bin")[col]
            ax.errorbar(mid.to_numpy(), g.mean().to_numpy(), yerr=g.sem().to_numpy(), color=color,
                        marker="o", ms=4, capsize=2, lw=1.2, label=f"level {int(c)}")
        ax.set_xscale("log")
        ax.set_xlabel("shot number (depth bin centre)")
        ax.set_ylabel("temperature (K)")
        ax.set_title(titles[col])
    axes[0].legend(fontsize=8)
    ax = axes[-1]
    data = [samples.loc[samples["label"].astype(int) == c, "ne"].dropna() for c in classes]
    parts = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch, color in zip(parts["boxes"], CLASS_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(classes) + 1), [int(c) for c in classes])
    ax.set_yscale("log")
    ax.set_xlabel("aging level")
    ax.set_ylabel("n_e (cm$^{-3}$)")
    ax.set_title("Electron density, H-alpha Stark width")
    fig.suptitle("Plasma state versus depth and aging level (mean ± SEM)")
    fig.tight_layout()
    return save(fig, path)


def plot_line_selection(candidates: pd.DataFrame, path_df: pd.DataFrame, x: np.ndarray,
                        y: np.ndarray, mask: np.ndarray, path: Path,
                        y_raw: np.ndarray | None = None, response_ln: np.ndarray | None = None):
    """Greedy elimination path, grand-mean Boltzmann plot of every Fe I candidate
    and, when a response was fitted, the per-line evidence for it."""
    show_response = y_raw is not None and response_ln is not None
    n_cols = 3 if show_response else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7.5 * n_cols, 5),
                             gridspec_kw={"width_ratios": [1, 2, 1.4][:n_cols]})
    ax = axes[0]
    ax.plot(path_df["n_lines"], path_df["max_abs_residual"], marker="o", ms=3,
            label="worst line |median residual|")
    ax.axvline(int(mask.sum()), color="crimson", ls="--", lw=1, label=f"selected: {int(mask.sum())} lines")
    ax.set_xlabel("lines kept")
    ax.set_ylabel("ln units")
    ax.invert_xaxis()
    ax2 = ax.twinx()
    ax2.plot(path_df["n_lines"], path_df["median_T"], color="grey", lw=1, marker=".")
    ax2.set_ylabel("median T of the set (K, grey)")
    ax.legend(fontsize=8)
    ax.set_title("Backward elimination of off-line lines")

    ax = axes[1]
    mean = np.nanmean(_centred(y), axis=0)
    sd = np.nanstd(_centred(y), axis=0)
    ax.errorbar(x[~mask], mean[~mask], yerr=sd[~mask], fmt="o", color="lightgrey", ms=4, lw=0.6,
                label="rejected candidate")
    ax.errorbar(x[mask], mean[mask], yerr=sd[mask], fmt="o", color="#c44e52", ms=5, lw=0.8,
                label="selected")
    slope, intercept = np.polyfit(x[mask], mean[mask], 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, intercept + slope * xx, color="#c44e52", lw=1.2,
            label=f"fit of selected: T = {-1 / (8.617333262e-5 * slope):,.0f} K")
    ok = candidates[candidates["status"] == "ok"]
    for xi, yi, wl in zip(x[mask], mean[mask], ok["wavelength"].to_numpy()[mask]):
        ax.annotate(f"{wl:.2f}", (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    counts = candidates["status"].value_counts().to_dict()
    ax.set_xlabel("E_k (eV)")
    ax.set_ylabel(r"ln($I\lambda / g_k A_k$), centred  (mean ± SD over samples)")
    ax.set_title("Fe I candidates — screening: " + ", ".join(f"{k} {v}" for k, v in counts.items()),
                 fontsize=9)
    ax.legend(fontsize=8)

    if show_response:
        from .boltzmann import fit_rows

        ax = axes[2]
        fit = fit_rows(x[mask], y[:, mask])
        resid = y_raw - fit.intercept[:, None] - fit.slope[:, None] * x
        mean_resid = np.nanmean(resid, axis=0)
        wl = ok["wavelength"].to_numpy()
        shift = np.nanmean(mean_resid[mask] - response_ln[mask])
        ax.scatter(wl[~mask], mean_resid[~mask] - shift, color="lightgrey", s=18, label="rejected")
        ax.scatter(wl[mask], mean_resid[mask] - shift, color="#c44e52", s=22, label="selected")
        order = np.argsort(wl)
        ax.plot(wl[order], response_ln[order], color="black", lw=1.2, label="fitted ln R(λ)")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("mean residual of uncorrected plot")
        ax.set_title("Spectral response (shared by all spectra)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, path)
