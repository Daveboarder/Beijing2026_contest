"""Turning 200 raw shots per sample into model-ready feature matrices.

The 200 shots of a sample are a depth profile through one spot, not 200
replicates, so how the spectrum is collapsed into features is a modelling
choice rather than a detail. ``encoding`` selects it (see :mod:`.depth`):
``mean`` throws depth away, the ``depth_*`` options keep it.

``n_groups=k`` returns k feature rows per sample instead of one, built from
interleaved shot subsets that each span the full depth range. This augments the
tiny training set, and the returned ``groups`` array keeps rows of the same
sample together during cross-validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Config
from .data import load_index, load_shots, load_wavelength
from .depth import encode_sample, line_indices
from .preprocessing import Preprocessor


class FeatureSet:
    """Feature matrix with sample bookkeeping."""

    def __init__(self, X, y, sample_ids, groups, split, wavelength):
        self.X = X
        self.y = y
        self.sample_ids = np.asarray(sample_ids)
        self.groups = np.asarray(groups)
        self.split = np.asarray(split)
        self.wavelength = wavelength

    def subset(self, split: str) -> FeatureSet:
        mask = self.split == split
        y = self.y[mask] if self.y is not None else None
        return FeatureSet(
            self.X[mask], y, self.sample_ids[mask], self.groups[mask],
            self.split[mask], self.wavelength,
        )

    def __repr__(self) -> str:
        return f"FeatureSet(X={self.X.shape}, n_samples={len(set(self.sample_ids))})"


def bin_spectrum(x: np.ndarray, factor: int, bounds) -> np.ndarray:
    """Average neighbouring detector pixels by ``factor``, per spectrometer channel."""
    if factor <= 1:
        return x
    blocks = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        block = x[..., a:b]
        n = (block.shape[-1] // factor) * factor
        block = block[..., :n]
        blocks.append(block.reshape(block.shape[:-1] + (n // factor, factor)).mean(axis=-1))
    return np.concatenate(blocks, axis=-1)


def _bin_axis(wavelength: np.ndarray, factor: int, bounds) -> np.ndarray:
    if factor <= 1:
        return wavelength
    return bin_spectrum(wavelength[None, :], factor, bounds)[0]


def _sample_rows(cfg: Config, sample_id: str, pre: Preprocessor, n_groups: int,
                 bin_factor: int, encoding: str, n_bins: int, augment: str,
                 bounds, line_idx: np.ndarray | None) -> np.ndarray:
    """Preprocess one sample's shots and encode them into ``n_groups`` rows."""
    processed = pre(load_shots(cfg, sample_id, mmap=False))
    rows = encode_sample(
        processed, encoding=encoding, n_bins=n_bins,
        bulk_fraction=pre.bulk_fraction, n_groups=n_groups, augment=augment,
        line_idx=line_idx,
    )

    if bin_factor > 1:
        # Only the spectral prefix is wavelength-ordered; line descriptors (if
        # any) sit after it and must not be averaged as if they were pixels.
        n_wl = bounds[-1] - bounds[0]
        n_spectra = max(rows.shape[1] // n_wl, 0)
        if n_spectra == 0:
            return rows
        spectral = rows[:, : n_spectra * n_wl]
        extra = rows[:, n_spectra * n_wl :]
        parts = np.array_split(spectral, n_spectra, axis=1)
        binned = np.concatenate([bin_spectrum(p, bin_factor, bounds) for p in parts], axis=1)
        rows = np.concatenate([binned, extra], axis=1) if extra.size else binned
    return rows


def build_features(
    cfg: Config,
    pre: Preprocessor | None = None,
    n_groups: int = 1,
    bin_factor: int = 1,
    encoding: str | None = None,
    n_bins: int | None = None,
    augment: str | None = None,
    n_jobs: int = 8,
    use_cache: bool = True,
) -> FeatureSet:
    """Build (and cache) the feature matrix for all cached samples.

    ``encoding`` and ``n_bins`` fall back to the ``features`` section of the
    config, so every script uses the same representation unless it overrides it.
    """
    pre = pre or Preprocessor.from_config(cfg)
    bounds = tuple(cfg["data"].get("channel_bounds", (0, 4094, 8188, 12282)))
    feat_cfg = cfg.get("features", {})
    encoding = encoding or feat_cfg.get("encoding", "mean")
    n_bins = n_bins or feat_cfg.get("n_bins", 8)
    augment = augment or feat_cfg.get("augment", "blocks")

    wavelength = load_wavelength(cfg)
    line_idx = line_indices(wavelength) if encoding in ("mean_lines", "lines") else None

    key = json.dumps(
        {"pre": asdict(pre), "n_groups": n_groups, "bin": bin_factor,
         "encoding": encoding, "n_bins": n_bins, "augment": augment,
         "lines": None if line_idx is None else line_idx.tolist()},
        sort_keys=True, default=str,
    )
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_file: Path = cfg.cache_dir / "features" / f"{digest}.npz"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_file.exists():
        blob = np.load(cache_file, allow_pickle=True)
        y = blob["y"]
        return FeatureSet(blob["X"], np.where(y < 0, np.nan, y), blob["sample_ids"],
                          blob["groups"], blob["split"], blob["wavelength"])

    index = load_index(cfg)
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_sample_rows)(cfg, sid, pre, n_groups, bin_factor, encoding, n_bins,
                              augment, bounds, line_idx)
        for sid in index["sample_id"]
    )

    X = np.vstack(results)
    counts = [r.shape[0] for r in results]
    sample_ids = np.repeat(index["sample_id"].to_numpy(), counts)
    split = np.repeat(index["split"].to_numpy(), counts)
    labels = index["label"].to_numpy(dtype=float)
    y = np.repeat(labels, counts)
    groups = np.repeat(np.arange(len(index)), counts)

    # Wavelength axis covers the spectral prefix only; trailing descriptors are
    # not wavelengths, so the stored axis may be shorter than X.shape[1].
    axis = _bin_axis(wavelength, bin_factor, bounds)
    n_wl = axis.size
    if X.shape[1] >= n_wl and X.shape[1] % n_wl == 0:
        axis = np.tile(axis, X.shape[1] // n_wl)
    elif X.shape[1] > n_wl:
        # mean_lines: one spectrum + line descriptors
        n_spectra = X.shape[1] // n_wl
        if n_spectra >= 1:
            axis = np.concatenate([np.tile(axis, n_spectra),
                                  np.full(X.shape[1] - n_spectra * n_wl, np.nan)])
        else:
            axis = np.full(X.shape[1], np.nan)
    else:
        axis = np.full(X.shape[1], np.nan)

    np.savez_compressed(
        cache_file, X=X, y=np.where(np.isnan(y), -1, y), sample_ids=sample_ids,
        groups=groups, split=split, wavelength=axis,
    )
    with open(cache_file.with_suffix(".json"), "w", encoding="utf-8") as fh:
        fh.write(key)
    return FeatureSet(X, y, sample_ids, groups, split, axis)


def aggregate_predictions(sample_ids: np.ndarray, proba: np.ndarray,
                          classes: np.ndarray) -> pd.DataFrame:
    """Average per-row class probabilities into one prediction per sample."""
    frame = pd.DataFrame(proba, columns=classes)
    frame["sample_id"] = sample_ids
    mean = frame.groupby("sample_id").mean()
    mean["predicted_label"] = mean[list(classes)].to_numpy().argmax(axis=1)
    mean["predicted_label"] = [classes[i] for i in mean["predicted_label"]]
    return mean.reset_index()
