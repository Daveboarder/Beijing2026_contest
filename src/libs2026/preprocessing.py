"""Spectral preprocessing: shot screening, baseline removal, normalisation.

Everything operates on 2-D arrays shaped ``(n_spectra, n_wavelengths)`` and is
channel-aware: the Avantes spectrometer delivers three concatenated channels
with overlapping ranges, so baselines and intensity scales must be handled
separately for each of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

DEFAULT_CHANNEL_BOUNDS = (0, 4094, 8188, 12282)


def channel_slices(bounds=DEFAULT_CHANNEL_BOUNDS) -> list[slice]:
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def snip_baseline(spectra: np.ndarray, iterations: int = 40) -> np.ndarray:
    """SNIP baseline estimate (Ryan et al.) for a batch of spectra.

    Works on the log-log-sqrt transformed signal so that intense emission lines
    are clipped away while the slowly varying continuum survives.
    """
    x = np.asarray(spectra, dtype=np.float64)
    offset = np.minimum(x.min(axis=1, keepdims=True), 0.0)
    v = np.log(np.log(np.sqrt(x - offset + 1.0) + 1.0) + 1.0)

    for p in range(iterations, 0, -1):
        shifted = 0.5 * (np.roll(v, p, axis=1) + np.roll(v, -p, axis=1))
        shifted[:, :p] = v[:, :p]
        shifted[:, v.shape[1] - p:] = v[:, v.shape[1] - p:]
        v = np.minimum(v, shifted)

    baseline = (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0 + offset
    return baseline.astype(spectra.dtype)


def remove_baseline(
    spectra: np.ndarray, iterations: int = 40, bounds=DEFAULT_CHANNEL_BOUNDS
) -> np.ndarray:
    """Subtract a per-channel SNIP baseline."""
    out = np.array(spectra, dtype=np.float32, copy=True)
    for sl in channel_slices(bounds):
        block = out[:, sl]
        out[:, sl] = block - snip_baseline(block, iterations)
    return out


def smooth(spectra: np.ndarray, window: int = 9, polyorder: int = 3,
           bounds=DEFAULT_CHANNEL_BOUNDS) -> np.ndarray:
    """Per-channel Savitzky-Golay smoothing (channels are not contiguous)."""
    out = np.array(spectra, dtype=np.float32, copy=True)
    for sl in channel_slices(bounds):
        out[:, sl] = savgol_filter(out[:, sl], window, polyorder, axis=1)
    return out


def normalize(
    spectra: np.ndarray,
    method: str = "tic",
    per_channel: bool = True,
    bounds=DEFAULT_CHANNEL_BOUNDS,
    reference: slice | None = None,
) -> np.ndarray:
    """Intensity normalisation, optionally applied to each channel separately.

    ``tic`` divides by the total emission, ``max`` by the strongest line,
    ``l2`` by the Euclidean norm and ``snv`` applies a standard normal variate
    transform (centre and scale each spectrum).

    ``reference`` selects how the scale is derived. ``None`` scales every shot
    by its own norm, which is right when shots are replicates but destroys the
    depth profile: it forces every depth to the same total intensity. Passing a
    slice of bulk shots instead derives one scale per sample from those shots
    and applies it to all of them, so differences between depths survive while
    sample-to-sample coupling differences are still removed.
    """
    if method in (None, "none"):
        return np.asarray(spectra, dtype=np.float32)

    out = np.array(spectra, dtype=np.float32, copy=True)
    blocks = channel_slices(bounds) if per_channel else [slice(None)]
    for sl in blocks:
        block = out[:, sl]
        # Statistics come either from each shot itself or from the bulk shots.
        stat_src = block if reference is None else block[reference].mean(axis=0, keepdims=True)
        if method == "tic":
            scale = np.abs(stat_src).sum(axis=1, keepdims=True)
            block = block / np.where(scale > 0, scale, 1.0) * block.shape[1]
        elif method == "max":
            scale = stat_src.max(axis=1, keepdims=True)
            block = block / np.where(scale > 0, scale, 1.0)
        elif method == "l2":
            scale = np.linalg.norm(stat_src, axis=1, keepdims=True)
            block = block / np.where(scale > 0, scale, 1.0) * np.sqrt(block.shape[1])
        elif method == "snv":
            mean = stat_src.mean(axis=1, keepdims=True)
            std = stat_src.std(axis=1, keepdims=True)
            block = (block - mean) / np.where(std > 0, std, 1.0)
        else:
            raise ValueError(f"Unknown normalization '{method}'")
        out[:, sl] = block
    return out


def shot_outlier_mask(spectra: np.ndarray, z_threshold: float = 4.0,
                      window: int = 11) -> np.ndarray:
    """Flag shots that depart from the *local* depth trend of their neighbours.

    A misfire has to be judged against the shots either side of it, not against
    the sample as a whole. The first shots legitimately carry 1.4-1.9x the bulk
    emission because they ablate the surface layer, so any global median/MAD
    rule flags the entire surface region -- that is, precisely the shots that
    separate the aging levels -- as outliers.
    """
    total = np.asarray(spectra, dtype=np.float64).sum(axis=1)
    trend = median_filter(total, size=min(window, total.size), mode="nearest")
    residual = total - trend
    mad = np.median(np.abs(residual - np.median(residual)))
    if mad <= 0:
        return np.ones(total.shape[0], dtype=bool)
    return np.abs(0.6745 * (residual - np.median(residual)) / mad) <= z_threshold


def repair_outlier_shots(spectra: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Replace flagged shots in place instead of deleting them.

    Shot index *is* the depth coordinate, so dropping a shot would shift every
    later shot to a shallower depth than it really is. A rejected shot is
    therefore overwritten by the mean of its nearest surviving neighbours,
    which keeps the depth axis intact.
    """
    keep = np.asarray(keep, dtype=bool)
    if keep.all() or keep.sum() < 5:
        return spectra

    out = np.array(spectra, copy=True)
    good = np.flatnonzero(keep)
    for i in np.flatnonzero(~keep):
        left = good[good < i]
        right = good[good > i]
        if left.size and right.size:
            out[i] = 0.5 * (out[left[-1]] + out[right[0]])
        else:
            out[i] = out[left[-1] if left.size else right[0]]
    return out


@dataclass
class Preprocessor:
    """Configurable preprocessing chain applied to one sample at a time.

    The output always has the same number of rows as the input, in the original
    firing order, because row index doubles as the depth coordinate.
    """

    baseline: str = "none"
    snip_iterations: int = 40
    normalization: str = "l2"
    per_channel: bool = True
    smooth_window: int = 0
    repair_outlier_shots: bool = True
    outlier_z: float = 4.0
    # Neighbourhood used to establish the local depth trend for outlier scoring.
    outlier_window: int = 11
    # Shots treated as bulk material, as a fraction of the shot sequence.
    bulk_fraction: float = 0.3
    # Derive the normalisation scale from the bulk shots ("bulk") or from each
    # shot separately ("shot"). Only "bulk" preserves the depth profile.
    normalization_reference: str = "bulk"
    channel_bounds: tuple = DEFAULT_CHANNEL_BOUNDS

    def bulk_slice(self, n_shots: int) -> slice:
        """The trailing shots that represent unaffected bulk material."""
        start = int(round(n_shots * (1.0 - self.bulk_fraction)))
        return slice(min(start, n_shots - 1), n_shots)

    def __call__(self, shots: np.ndarray) -> np.ndarray:
        x = np.asarray(shots, dtype=np.float32)
        bulk = self.bulk_slice(x.shape[0])

        if self.repair_outlier_shots:
            x = repair_outlier_shots(x, shot_outlier_mask(x, self.outlier_z,
                                                          self.outlier_window))
        if self.smooth_window and self.smooth_window > 3:
            x = smooth(x, self.smooth_window, bounds=self.channel_bounds)
        if self.baseline == "snip":
            x = remove_baseline(x, self.snip_iterations, self.channel_bounds)
        elif self.baseline not in (None, "none"):
            raise ValueError(f"Unknown baseline method '{self.baseline}'")

        reference = bulk if self.normalization_reference == "bulk" else None
        x = normalize(x, self.normalization, self.per_channel, self.channel_bounds, reference)
        return x

    @classmethod
    def from_config(cls, cfg) -> Preprocessor:
        pre = cfg["preprocessing"]
        return cls(
            baseline=pre.get("baseline", "none"),
            snip_iterations=pre.get("snip_iterations", 40),
            normalization=pre.get("normalization", "l2"),
            per_channel=pre.get("per_channel", True),
            smooth_window=pre.get("smooth_window", 0),
            repair_outlier_shots=pre.get("repair_outlier_shots", True),
            outlier_z=pre.get("outlier_z", 4.0),
            outlier_window=pre.get("outlier_window", 11),
            bulk_fraction=pre.get("bulk_fraction", 0.3),
            normalization_reference=pre.get("normalization_reference", "bulk"),
            channel_bounds=tuple(cfg["data"].get("channel_bounds", DEFAULT_CHANNEL_BOUNDS)),
        )
