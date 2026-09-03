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
