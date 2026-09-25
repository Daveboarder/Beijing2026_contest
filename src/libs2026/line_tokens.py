"""Data-driven spectral-line tokens: find the lines, describe them, rank them.

The theory-driven tokens of :mod:`.tokens` start from a line dictionary ranked
by Saha-Boltzmann intensity and Voigt-fit every entry. On this dataset that
loses most of the class information (``pca_mlp`` on their amplitudes: 0.42).
This module works the other way round:

1. **Detect** the peaks the spectrometer actually records, on the training
   mean spectrum. A [1, 2, 1] / 4 kernel is applied first: the CMOS detectors
   carry a fixed even/odd pixel pattern of +-10-15 % (strongest on channel 3)
   that would otherwise turn every other pixel into a "peak", and that kernel
   has an exact zero at the Nyquist frequency. Where two channels overlap, a
   wavelength is taken from one channel only (split at the overlap midpoint),
   which also drops the noisy first pixels of channels 2 and 3.
2. **Describe** every line in every depth row with robust, fit-free profile
   descriptors against a valley-to-valley linear background. Measured with
   ``pca_mlp`` on the 10-block rows (scripts/31), the 274 detected lines keep
   most of the full-spectrum accuracy (0.708 against 0.735), and the line
   *shape* (width, shift, asymmetry: 0.684) carries far more of it than the
   line intensity (area, height: 0.634). The descriptors therefore keep shape,
   intensity and the local continuum.
3. **Identify** each peak against the air line database, allowing for a
   per-channel wavelength-calibration offset. The assignment is tentative: it
   ranks candidates by Saha-Boltzmann intensity with rough abundance priors.
4. **Rank** lines by how much a classifier relies on them: out-of-fold
   occlusion of one line's descriptors, and recursive elimination on top of it
   so that redundant lines do not keep each other alive.

Descriptor layout mirrors :mod:`.tokens` (five line-shape channels, a validity
flag, then continuum channels), so :class:`~libs2026.cnn.TokenCNN` can consume
these tokens through :meth:`LineTokens.cnn_kwargs`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, peak_widths
from sklearn.model_selection import StratifiedGroupKFold

from .config import PROJECT_ROOT, Config
from .evaluation import predict_scores
from .lines_db import N_STATIC, atomic_number, ion_binary, line_intensities, list_elements
from .models import get_model
from .preprocessing import DEFAULT_CHANNEL_BOUNDS, Preprocessor

# Line shape (zeroed by TokenCNN when the line is not detected), validity flag,
# then the local continuum, which stays informative even without a line.
DESCRIPTOR_NAMES = (
    "area",             # net integral over the line core (a.u. * nm)
    "height",           # net intensity at the peak pixel
    "width",            # second moment of the net profile (nm)
    "shift",            # net-intensity centroid relative to the peak pixel (nm)
    "asymmetry",        # (left - right) / total net intensity of the core
    "detected",         # 1 if height > 3 row-noise sigmas
    "continuum",        # background level at the peak pixel
    "continuum_slope",  # background slope (a.u. / nm)
)
N_DESCRIPTORS = len(DESCRIPTOR_NAMES)
D_AREA, D_HEIGHT, D_WIDTH, D_SHIFT, D_ASYM, D_DETECTED, D_CONT, D_SLOPE = range(N_DESCRIPTORS)

NYQUIST_KERNEL = np.array([0.25, 0.5, 0.25])
# For white detector noise, the robust sigma of (smoothed - 5-pixel box average)
# is 0.623 of the sigma of the smoothed spectrum itself. Dividing by it expresses
# the noise floor in units of what a peak height is actually compared against.
HIGHPASS_TO_SMOOTHED_SIGMA = 0.623

# Rough abundances used only to rank candidate assignments. Steel matrix and
# alloying elements, ambient air, and typical surface contaminants.
DEFAULT_PRIORS: dict[str, float] = {
    "Fe": 1.0, "Cr": 2e-2, "Mn": 1e-2, "Ni": 1e-2, "Si": 5e-3, "C": 5e-3,
    "Mo": 2e-3, "Cu": 2e-3, "V": 1e-3, "Co": 1e-3, "W": 1e-3, "Ti": 5e-4,
    "Al": 5e-4, "N": 0.5, "O": 0.2, "H": 2e-2, "Ar": 5e-3, "Ca": 2e-4,
    "P": 2e-4, "S": 2e-4, "Na": 1e-4, "K": 1e-4, "Mg": 1e-4, "Li": 1e-5,
    "Sr": 1e-5, "Ba": 1e-5, "B": 1e-5, "Zn": 1e-5, "Sn": 1e-5, "Pb": 1e-5,
}


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------


def smooth_nyquist(spectra: np.ndarray, bounds=DEFAULT_CHANNEL_BOUNDS) -> np.ndarray:
    """[1, 2, 1] / 4 smoothing per channel; removes the even/odd pixel pattern exactly."""
    x = np.asarray(spectra, dtype=np.float64)
    out = np.empty_like(x)
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[..., a:b] = convolve1d(x[..., a:b], NYQUIST_KERNEL, axis=-1, mode="nearest")
    return out


def usable_ranges(wavelength: np.ndarray, bounds=DEFAULT_CHANNEL_BOUNDS,
                  edge_nm: float = 0.5) -> list[tuple[float, float]]:
    """Wavelength span each channel is trusted for.

    Overlapping channels are split at the midpoint of their overlap, so every
    wavelength is described once and the unreliable ends of each channel
    (e.g. channel 2 below ~382 nm, channel 3 below ~591 nm) are skipped.
    """
    spans = [(float(wavelength[a]), float(wavelength[b - 1]))
             for a, b in zip(bounds[:-1], bounds[1:])]
    out = []
    for i, (lo, hi) in enumerate(spans):
        lo = lo + edge_nm if i == 0 else max(lo, 0.5 * (spans[i - 1][1] + lo))
        hi = hi - edge_nm if i == len(spans) - 1 else min(hi, 0.5 * (hi + spans[i + 1][0]))
        out.append((lo, hi))
    return out


def detect_lines(
    mean_spectrum: np.ndarray,
    wavelength: np.ndarray,
    bounds=DEFAULT_CHANNEL_BOUNDS,
    k_noise: float = 5.0,
    noise_window: int = 401,
    max_extent: int = 4,
    prominence_window: int = 21,
) -> pd.DataFrame:
    """Emission lines of a (mean) spectrum with their core and background pixels.

    A peak is kept when its prominence, measured within ``prominence_window``
    pixels, exceeds ``k_noise`` sigmas of the local noise of the smoothed
    spectrum (from the robust spread of smoothed minus a 5-pixel box average,
    as a running median over ``noise_window`` pixels). Both restrictions
    matter: on pure white noise an unbounded prominence at 5 high-pass sigmas
    passes ~28 peaks per 1000 pixels, these settings ~0.1. The core spans
    +-FWHM around the peak; the background is anchored at the lowest point
    between this peak and each neighbouring detected peak, at most
    ``max_extent`` core half-widths away. All pixel indices are global.
    """
    smoothed = smooth_nyquist(mean_spectrum[None, :], bounds)[0]
    records = []
    for channel, ((a, b), (lo_nm, hi_nm)) in enumerate(
            zip(zip(bounds[:-1], bounds[1:]), usable_ranges(wavelength, bounds))):
        s, w = smoothed[a:b], np.asarray(wavelength[a:b], dtype=np.float64)
        n = s.size
        hp = s - uniform_filter1d(s, 5, mode="nearest")
        noise = 1.4826 * median_filter(np.abs(hp), size=min(noise_window, n), mode="nearest") \
            / HIGHPASS_TO_SMOOTHED_SIGMA
        peaks, props = find_peaks(s, prominence=0, wlen=prominence_window)
        inside = (w[peaks] >= lo_nm) & (w[peaks] < hi_nm)
        keep = inside & (props["prominences"] > k_noise * noise[peaks])
        peaks, prominence = peaks[keep], props["prominences"][keep]
        if not peaks.size:
            continue
        fwhm = peak_widths(s, peaks, rel_height=0.5)[0]
        step = float(np.median(np.diff(w)))
        for i, p in enumerate(peaks):
            half = max(2, int(round(fwhm[i])))
            reach = max_extent * half
            left_lim = max(peaks[i - 1] + 1 if i > 0 else 0, p - reach)
            right_lim = min(peaks[i + 1] - 1 if i + 1 < peaks.size else n - 1, p + reach)
            lv = left_lim + int(np.argmin(s[left_lim:p])) if p > left_lim else max(p - 1, 0)
            rv = (p + 1 + int(np.argmin(s[p + 1:right_lim + 1])) if right_lim > p
                  else min(p + 1, n - 1))
            records.append({
                "channel": channel + 1,
                "wavelength_nm": float(w[p]),
                "peak_px": a + int(p),
                "core_lo": a + max(int(p) - half, lv + 1),
                "core_hi": a + min(int(p) + half, rv - 1),
                "bg_left": a + int(lv),
                "bg_right": a + int(rv),
                "step_nm": step,
                "fwhm_nm": float(fwhm[i] * step),
                "prominence": float(prominence[i]),
                "snr": float(prominence[i] / max(noise[p], 1e-12)),
            })
    columns = ["channel", "wavelength_nm", "peak_px", "core_lo", "core_hi", "bg_left", "bg_right",
               "step_nm", "fwhm_nm", "prominence", "snr"]
    lines = pd.DataFrame.from_records(records, columns=columns)
    lines.insert(0, "line_id", np.arange(len(lines)))
    return lines


# ----------------------------------------------------------------------------
# Descriptors
# ----------------------------------------------------------------------------


def _row_noise(smoothed: np.ndarray, wavelength: np.ndarray, bounds) -> np.ndarray:
    """Robust noise sigma of every smoothed row, per channel: ``(n_rows, n_channels)``."""
    out = np.zeros((smoothed.shape[0], len(bounds) - 1))
    ranges = usable_ranges(wavelength, bounds)
    for c, ((a, b), (lo, hi)) in enumerate(zip(zip(bounds[:-1], bounds[1:]), ranges)):
        seg = smoothed[:, a:b]
        hp = seg - uniform_filter1d(seg, 5, axis=1, mode="nearest")
        inside = (wavelength[a:b] >= lo) & (wavelength[a:b] < hi)
        out[:, c] = 1.4826 * np.median(np.abs(hp[:, inside]), axis=1) / HIGHPASS_TO_SMOOTHED_SIGMA
    return out


def extract_line_tokens(spectra: np.ndarray, wavelength: np.ndarray, lines: pd.DataFrame,
                        bounds=DEFAULT_CHANNEL_BOUNDS) -> np.ndarray:
    """Descriptors of every line in every row: ``(n_rows, n_lines, N_DESCRIPTORS)``.

    Everything is measured on the Nyquist-smoothed rows. The background under
    a line is the straight line through the two valley anchors (each averaged
    over three pixels); area, height, width, shift and asymmetry are taken
    from the background-subtracted core.
    """
    s = smooth_nyquist(np.atleast_2d(spectra), bounds)
    n_rows, n_px = s.shape
    noise = _row_noise(s, np.asarray(wavelength, dtype=np.float64), bounds)
    out = np.zeros((n_rows, len(lines), N_DESCRIPTORS), dtype=np.float32)
    rows = np.arange(n_rows)
    for j, line in enumerate(lines.itertuples(index=False)):
        p, lv, rv = int(line.peak_px), int(line.bg_left), int(line.bg_right)
        step = float(line.step_nm)
        a, b = bounds[line.channel - 1], bounds[line.channel]
        bl = s[:, max(lv - 1, a):min(lv + 2, b)].mean(axis=1)
        br = s[:, max(rv - 1, a):min(rv + 2, b)].mean(axis=1)
        slope = (br - bl) / max(rv - lv, 1)
        idx = np.arange(int(line.core_lo), int(line.core_hi) + 1)
        offset = idx - p
        net = s[:, idx] - (bl[:, None] + slope[:, None] * (idx - lv))
        height = net[rows, np.searchsorted(idx, p)]
        positive = np.clip(net, 0.0, None)
        weight = positive.sum(axis=1) + 1e-12
        centroid = (positive * offset).sum(axis=1) / weight
        spread = (positive * (offset - centroid[:, None]) ** 2).sum(axis=1) / weight
        out[:, j, D_AREA] = net.sum(axis=1) * step
        out[:, j, D_HEIGHT] = height
        out[:, j, D_WIDTH] = np.sqrt(spread) * step
        out[:, j, D_SHIFT] = centroid * step
        out[:, j, D_ASYM] = (net[:, offset < 0].sum(axis=1) - net[:, offset > 0].sum(axis=1)) \
            / (np.abs(net).sum(axis=1) + 1e-12)
        out[:, j, D_DETECTED] = height > 3.0 * noise[:, line.channel - 1]
        out[:, j, D_CONT] = bl + slope * (p - lv)
        out[:, j, D_SLOPE] = slope / step
    return out


def inflection_bounds(mean_spectrum: np.ndarray, lines: pd.DataFrame,
                      bounds=DEFAULT_CHANNEL_BOUNDS, max_px: int = 25) -> pd.DataFrame:
    """Nearest inflection point left and right of every line centre (``b1_px``, ``b2_px``).

    Inflection points are sign changes of the second difference of the
    Nyquist-smoothed spectrum (without the smoothing the even/odd pixel
    pattern puts one next to every pixel). Walking out from the centre, where
    the second difference is negative, the bound is the pixel nearest the
    zero crossing, never the centre itself. For a Gaussian line they sit at
    +-sigma, i.e. ~0.85 FWHM apart and at ~60 % of the peak height.
    """
    smoothed = smooth_nyquist(mean_spectrum[None, :], bounds)[0]
    d2 = np.zeros_like(smoothed)
    for a, b in zip(bounds[:-1], bounds[1:]):
        d2[a + 1:b - 1] = smoothed[a:b - 2] - 2 * smoothed[a + 1:b - 1] + smoothed[a + 2:b]
    b1, b2 = [], []
    for line in lines.itertuples(index=False):
        p, a, b = int(line.peak_px), bounds[line.channel - 1], bounds[line.channel]
        i = p - 1
        while i > max(a + 1, p - max_px) and d2[i] < 0:
            i -= 1
        b1.append(i if i + 1 == p or abs(d2[i]) <= abs(d2[i + 1]) else i + 1)
        j = p + 1
        while j < min(b - 2, p + max_px) and d2[j] < 0:
            j += 1
        b2.append(j if j - 1 == p or abs(d2[j]) <= abs(d2[j - 1]) else j - 1)
    return pd.DataFrame({"b1_px": b1, "b2_px": b2}, index=lines.index)


def line_areas(spectra: np.ndarray, wavelength: np.ndarray, b1, b2,
               baseline: str = "linear") -> np.ndarray:
    """Signal area of every line in every row: ``(n_rows, n_lines)``.

    ``linear`` integrates ``b1..b2`` (trapezoid, in nm) and subtracts the
    trapezoid under the straight line joining the two bound values, so a
    constant or sloped continuum contributes nothing.

    ``libsmethods`` reproduces ``calculate_signal_area`` of
    ``context/LIBSmethods.py`` as written: pixel units, ``b2`` excluded, and a
    baseline term one pixel wide rather than ``b2 - b1`` pixels, so most of the
    continuum under the line stays in the "area".
    """
    x = np.asarray(spectra, dtype=np.float64)
    wl = np.asarray(wavelength, dtype=np.float64)
    out = np.zeros((x.shape[0], len(b1)), dtype=np.float32)
    for j, (lo, hi) in enumerate(zip(np.asarray(b1, int), np.asarray(b2, int))):
        if baseline == "linear":
            chord = 0.5 * (x[:, lo] + x[:, hi]) * (wl[hi] - wl[lo])
            out[:, j] = np.trapezoid(x[:, lo:hi + 1], wl[lo:hi + 1], axis=1) - chord
        elif baseline == "libsmethods":
            out[:, j] = np.trapezoid(x[:, lo:hi], axis=1) - 0.5 * (x[:, lo] + x[:, hi])
        else:
            raise ValueError(f"Unknown baseline '{baseline}' (linear | libsmethods)")
    return out


# ----------------------------------------------------------------------------
# Identification
# ----------------------------------------------------------------------------


def line_candidates(db_path: str | Path, priors: dict[str, float],
                    te_grid=(7000.0, 10000.0, 13000.0), ne: float = 1e17,
                    wl_min: float | None = None, wl_max: float | None = None) -> pd.DataFrame:
    """Every database line of the prior elements with a relative Saha-Boltzmann intensity.

    ``log_intensity`` is log10 of the modelled line intensity, maximised over
    ``te_grid``, with the prior abundance entering the optical depth. A single
    temperature (the Fe I Boltzmann value of ~8000 K) would suppress the
    high-excitation N I / O I lines of the air plasma, which are among the
    strongest peaks of channel 3.
    """
    db_path = str(Path(db_path).resolve())
    available = set(list_elements(db_path))
    frames = []
    for element, prior in priors.items():
        if element not in available:
            continue
        try:
            per_te = [line_intensities(element, float(te), ne, 1e-4, prior, 1.4e-4, db_path)
                      for te in te_grid]
        except (ValueError, ZeroDivisionError):
            continue
        wl, ion, ei, ek, gi, gk, ak, _ = per_te[0]
        inten = np.max([p[-1] for p in per_te], axis=0) if wl.size else wl
        keep = inten > 0
        if wl_min is not None:
            keep &= wl >= wl_min
        if wl_max is not None:
            keep &= wl <= wl_max
        if not keep.any():
            continue
        frames.append(pd.DataFrame({
            "element": element, "ion_state": np.asarray(ion)[keep].astype(str),
            "db_wavelength_nm": wl[keep], "Ei": ei[keep], "Ek": ek[keep], "gi": gi[keep],
            "gk": gk[keep], "Ak": ak[keep], "log_intensity": np.log10(inten[keep]),
        }))
    if not frames:
        raise RuntimeError(f"No candidate lines for {sorted(priors)} in {db_path}")
    return pd.concat(frames, ignore_index=True).sort_values("db_wavelength_nm", ignore_index=True)


def estimate_channel_offsets(lines: pd.DataFrame, candidates: pd.DataFrame,
                             max_offset_nm: float = 0.2, n_strong: int = 60,
                             top_fraction: float = 0.1) -> dict[int, float]:
    """Wavelength-calibration offset (observed - tabulated) of each channel.

    The ``n_strong`` most prominent peaks of a channel are matched against its
    theoretically strongest candidate lines, and the offset maximising the sum
    of Gaussian proximity scores (sigma = half a pixel) wins. Na D2, for
    example, appears ~0.07 nm below its tabulated air wavelength on channel 2.
    """
    offsets = {}
    grid = np.arange(-max_offset_nm, max_offset_nm + 1e-9, 0.002)
    for channel, group in lines.groupby("channel"):
        strong = group.nlargest(n_strong, "prominence")
        lo, hi = group["wavelength_nm"].min() - 1, group["wavelength_nm"].max() + 1
        cand = candidates[candidates["db_wavelength_nm"].between(lo, hi)]
        if cand.empty or strong.empty:
            offsets[int(channel)] = 0.0
            continue
        cand = cand[cand["log_intensity"] >= cand["log_intensity"].quantile(1 - top_fraction)]
        sigma = 0.5 * float(group["step_nm"].iloc[0])
        diff = (strong["wavelength_nm"].to_numpy()[:, None]
                - cand["db_wavelength_nm"].to_numpy()[None, :])
        scores = [np.exp(-0.5 * ((diff - d) / sigma) ** 2).max(axis=1).sum() for d in grid]
        # Several offsets can tie on a flat plateau; take the middle of the best ones.
        best = np.flatnonzero(np.isclose(scores, np.max(scores)))
        offsets[int(channel)] = float(grid[best[best.size // 2]])
    return offsets


def identify_lines(lines: pd.DataFrame, candidates: pd.DataFrame, offsets: dict[int, float],
                   n_alternatives: int = 3) -> pd.DataFrame:
    """Tentative element / ion assignment of every detected line.

    Candidates within ``max(2, min(FWHM / 2, 4))`` pixels of the offset-corrected
    peak are scored as ``log_intensity - 0.5 * (distance / 1.5 pixels)^2``. The
    distance penalty is in pixels, not in line widths, so the centre of a
    Stark-broadened line (H-alpha, ~1.5 nm FWHM) still decides its assignment.
    ``n_competing`` counts candidates within one decade of the winner's score,
    a direct measure of how ambiguous (blended) the assignment is.
    """
    cand_wl = candidates["db_wavelength_nm"].to_numpy()
    records = []
    for line in lines.itertuples(index=False):
        corrected = line.wavelength_nm - offsets.get(int(line.channel), 0.0)
        tol = line.step_nm * max(2.0, min(0.5 * line.fwhm_nm / line.step_nm, 4.0))
        sigma = 1.5 * line.step_nm
        lo, hi = np.searchsorted(cand_wl, [corrected - tol, corrected + tol])
        rec = {"corrected_wavelength_nm": corrected, "assignment": "?", "element": "",
               "ion_state": "", "db_wavelength_nm": np.nan, "delta_nm": np.nan,
               "Ei": np.nan, "Ek": np.nan, "gi": np.nan, "gk": np.nan, "Ak": np.nan,
               "log_intensity": np.nan, "n_competing": 0, "alternatives": ""}
        if hi > lo:
            near = candidates.iloc[lo:hi].copy()
            near["delta_nm"] = corrected - near["db_wavelength_nm"]
            near["score"] = near["log_intensity"] - 0.5 * (near["delta_nm"] / sigma) ** 2
            near = near.sort_values("score", ascending=False)
            top = near.iloc[0]
            rec.update({k: top[k] for k in ("element", "ion_state", "db_wavelength_nm", "delta_nm",
                                            "Ei", "Ek", "gi", "gk", "Ak", "log_intensity")})
            rec["assignment"] = f"{top['element']} {top['ion_state']} {top['db_wavelength_nm']:.3f}"
            rec["n_competing"] = int((near["score"] >= top["score"] - 1.0).sum() - 1)
            rec["alternatives"] = "; ".join(
                f"{r.element} {r.ion_state} {r.db_wavelength_nm:.3f} ({r.delta_nm:+.3f})"
                for r in near.iloc[1:1 + n_alternatives].itertuples())
        records.append(rec)
    return pd.concat([lines.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1)


def static_channels(lines: pd.DataFrame) -> np.ndarray:
    """``(n_lines, N_STATIC)`` physics channels in ``lines_db.STATIC_FEATURE_NAMES`` order.

    Unassigned lines keep their corrected wavelength and zeros elsewhere.
    """
    static = np.zeros((len(lines), N_STATIC), dtype=np.float32)
    wl = lines.get("db_wavelength_nm", pd.Series(np.nan, index=lines.index))
    fallback = lines.get("corrected_wavelength_nm", lines["wavelength_nm"])
    static[:, 0] = np.where(wl.notna(), wl, fallback)
    if "element" not in lines:
        return static
    known = lines["element"].astype(str).str.len().to_numpy() > 0
    for col, name in ((1, "Ei"), (2, "Ek")):
        static[known, col] = lines.loc[known, name]
    for col, name in ((3, "gi"), (4, "gk"), (5, "Ak")):
        static[known, col] = np.log10(np.maximum(lines.loc[known, name].to_numpy(float), 1e-30))
    static[known, 6] = lines.loc[known, "log_intensity"]
    static[known, 7] = [atomic_number(e) for e in lines.loc[known, "element"]]
    static[known, 8] = [ion_binary(s) for s in lines.loc[known, "ion_state"]]
    return static


# ----------------------------------------------------------------------------
# Token container
# ----------------------------------------------------------------------------


@dataclass
class LineTokens:
    """Line tokens with sample bookkeeping.

    ``X`` is ``(n_samples, n_rows, n_lines, N_DESCRIPTORS)``: one row per depth
    block, in shot order. ``lines`` describes every column (detection geometry,
    tentative assignment and, once ranked, importance).
    """

    X: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray
    split: np.ndarray
    lines: pd.DataFrame
    static: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    descriptor_names: tuple = field(default=DESCRIPTOR_NAMES)

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_lines(self) -> int:
        return int(self.X.shape[2])

    def subset(self, split: str) -> LineTokens:
        mask = self.split == split
        return LineTokens(self.X[mask], self.y[mask], self.sample_ids[mask], self.split[mask],
                          self.lines, self.static, self.feature_mean, self.feature_std,
                          self.descriptor_names)

    def select(self, line_idx) -> LineTokens:
        """Restrict to the given line columns (in the given order)."""
        idx = np.asarray(line_idx, dtype=int)
        return LineTokens(self.X[:, :, idx], self.y, self.sample_ids, self.split,
                          self.lines.iloc[idx].reset_index(drop=True), self.static[idx],
                          self.feature_mean[idx], self.feature_std[idx], self.descriptor_names)

    def select_descriptors(self, names) -> LineTokens:
        idx = [self.descriptor_names.index(n) for n in names]
        return LineTokens(self.X[..., idx], self.y, self.sample_ids, self.split, self.lines,
                          self.static, self.feature_mean[:, idx], self.feature_std[:, idx],
                          tuple(names))

    def rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(X_rows, y_rows, groups, sample_ids_rows)``; X_rows is ``(n, n_lines * n_desc)``."""
        n, r = self.X.shape[:2]
        return (self.X.reshape(n * r, -1), np.repeat(self.y, r), np.repeat(np.arange(n), r),
                np.repeat(self.sample_ids, r))

    def cnn_kwargs(self) -> dict:
        """Arguments that let :class:`~libs2026.cnn.TokenCNN` read these tokens."""
        return {"static": self.static, "feature_mean": self.feature_mean,
                "feature_std": self.feature_std, "n_rows": self.n_rows, "n_lines": self.n_lines,
                "n_features": len(self.descriptor_names),
                "valid_index": self.descriptor_names.index("detected"),
                "n_dynamic": self.descriptor_names.index("detected")}

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = {f"line__{c}": self.lines[c].to_numpy() for c in self.lines.columns}
        columns = {k: (v.astype(str) if v.dtype == object else v) for k, v in columns.items()}
        np.savez_compressed(
            path, X=self.X, y=np.where(np.isnan(self.y), -1, self.y),
            sample_ids=self.sample_ids.astype(str), split=self.split.astype(str),
            static=self.static, feature_mean=self.feature_mean, feature_std=self.feature_std,
            descriptor_names=np.array(self.descriptor_names), **columns)
        return path

    @classmethod
    def load(cls, path: Path) -> LineTokens:
        blob = np.load(path, allow_pickle=False)
        prefix = "line__"
        lines = pd.DataFrame({k[len(prefix):]: blob[k] for k in blob.files if k.startswith(prefix)})
        y = blob["y"].astype(float)
        return cls(blob["X"], np.where(y < 0, np.nan, y), blob["sample_ids"], blob["split"], lines,
                   blob["static"], blob["feature_mean"], blob["feature_std"],
                   tuple(str(n) for n in blob["descriptor_names"]))

    def __repr__(self) -> str:
        return (f"LineTokens(n_samples={len(self.sample_ids)}, rows={self.n_rows}, "
                f"lines={self.n_lines}, descriptors={len(self.descriptor_names)})")


@dataclass
class LineTokenConfig:
    """Detection and identification settings (part of the cache key)."""

    n_groups: int = 10
    k_noise: float = 5.0
    noise_window: int = 401
    max_extent: int = 4
    prominence_window: int = 21
    te_grid: tuple = (7000.0, 10000.0, 13000.0)
    ne: float = 1e17
    priors: dict = field(default_factory=lambda: dict(DEFAULT_PRIORS))

    @classmethod
    def from_config(cls, cfg: Config) -> LineTokenConfig:
        section = dict(cfg.get("line_tokens", {}) or {})
        kw = {k: v for k, v in section.items() if k in cls.__dataclass_fields__}
        # YAML reads 1.0e17 (no sign) as a string, so coerce.
        for key in ("k_noise", "ne"):
            if key in kw:
                kw[key] = float(kw[key])
        if "te_grid" in kw:
            kw["te_grid"] = tuple(float(t) for t in kw["te_grid"])
        if "priors" in kw:
            kw["priors"] = {str(k): float(v) for k, v in kw["priors"].items()}
        return cls(**kw)


def _feature_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-line, per-descriptor mean / std over every training row."""
    flat = X.reshape(-1, X.shape[2], X.shape[3])
    std = flat.std(axis=0)
    return flat.mean(axis=0).astype(np.float32), np.where(std > 1e-8, std, 1.0).astype(np.float32)


def build_line_tokens(cfg: Config, settings: LineTokenConfig | None = None,
                      pre: Preprocessor | None = None, n_jobs: int = 8,
                      use_cache: bool = True, verbose: bool = True) -> LineTokens:
    """Detect, identify and describe the lines of every sample (cached).

    Rows are the ``n_groups`` consecutive shot blocks of the best classical
    setting (shot-referenced L2, no baseline), taken from
    :func:`~libs2026.features.build_features`. Detection uses the training
    mean only; it needs no labels, so nothing leaks into later CV folds.
    """
    from .features import build_features

    settings = settings or LineTokenConfig.from_config(cfg)
    if pre is None:
        pre = Preprocessor.from_config(cfg)
        pre.normalization_reference = "shot"
    bounds = tuple(cfg["data"].get("channel_bounds", DEFAULT_CHANNEL_BOUNDS))
    db_path = Path(cfg.get("tokens", {}).get("db_path", ""))
    if not db_path.is_absolute():
        db_path = (PROJECT_ROOT / db_path).resolve()

    key = json.dumps({"pre": asdict(pre), "settings": asdict(settings), "db": db_path.name,
                      "db_found": db_path.is_file(), "bounds": list(bounds), "kind": "line_tokens",
                      "version": 2}, sort_keys=True, default=str)
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_file = cfg.cache_dir / "line_tokens" / f"{digest}.npz"
    if use_cache and cache_file.exists():
        if verbose:
            print(f"line token cache hit: {cache_file}")
        return LineTokens.load(cache_file)

    features = build_features(cfg, pre, n_groups=settings.n_groups, bin_factor=1,
                              encoding="mean", augment="blocks", n_jobs=n_jobs)
    wavelength = np.asarray(features.wavelength, dtype=np.float64)
    train = features.split == "train"
    lines = detect_lines(features.X[train].mean(axis=0), wavelength, bounds, settings.k_noise,
                         settings.noise_window, settings.max_extent, settings.prominence_window)
    if verbose:
        per_channel = lines["channel"].value_counts().sort_index().to_dict()
        print(f"detected {len(lines)} lines: {per_channel} per channel")

    if db_path.is_file():
        candidates = line_candidates(db_path, settings.priors, settings.te_grid, settings.ne,
                                     float(wavelength.min()) - 1, float(wavelength.max()) + 1)
        offsets = estimate_channel_offsets(lines, candidates)
        lines = identify_lines(lines, candidates, offsets)
        lines["channel_offset_nm"] = lines["channel"].map(offsets)
        if verbose:
            print("channel calibration offsets (observed - tabulated, nm): "
                  + ", ".join(f"ch{c} {o:+.3f}" for c, o in offsets.items()))
            print(f"assigned {int((lines['assignment'] != '?').sum())} / {len(lines)} lines")
    elif verbose:
        print(f"line database not found at {db_path}; lines stay unassigned")

    tokens = extract_line_tokens(features.X, wavelength, lines, bounds)
    n_samples = len(np.unique(features.groups))
    X = tokens.reshape(n_samples, settings.n_groups, len(lines), N_DESCRIPTORS)
    first = np.arange(n_samples) * settings.n_groups
    split = features.split[first]
    mean, std = _feature_stats(X[split == "train"])
    result = LineTokens(X, np.asarray(features.y, dtype=float)[first], features.sample_ids[first],
                        split, lines, static_channels(lines), mean, std)
    result.save(cache_file)
    cache_file.with_suffix(".json").write_text(key, encoding="utf-8")
    if verbose:
        print(f"{result} -> {cache_file}")
    return result


# ----------------------------------------------------------------------------
# Importance and selection
# ----------------------------------------------------------------------------


def line_token_model(n_lines: int, n_descriptors: int, n_pca: int = 30):
    """``pca_mlp`` sized for a line subset (PCA width capped by the feature count)."""
    return get_model("pca_mlp", n_pca=max(1, min(n_pca, n_lines * n_descriptors - 1)))


def line_occlusion(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_factory=line_token_model,
                   n_splits: int = 4, seed: int = 0,
                   classes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold drop of the true-class probability when one line is removed.

    ``X`` is ``(n_rows, n_lines, n_descriptors)``. Inside every fold a model is
    fitted on the training rows; in the held-out rows all descriptors of one
    line are replaced by their training mean (i.e. zero after scaling) and the
    fall of the true-class probability is recorded. Returns the mean drop per
    line and the full ``(n_rows, n_lines)`` matrix for depth breakdowns.
    """
    X = np.asarray(X, dtype=np.float32)
    n, n_lines, n_desc = X.shape
    classes = np.unique(y) if classes is None else np.asarray(classes)
    drop = np.zeros((n, n_lines), dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in splitter.split(X, y, groups):
        model = model_factory(n_lines, n_desc).fit(X[tr].reshape(len(tr), -1), y[tr])
        held = X[te].copy()
        col, r = np.searchsorted(classes, y[te]), np.arange(len(te))
        base = predict_scores(model, held.reshape(len(te), -1), classes)[r, col]
        mean = X[tr].mean(axis=0)
        for j in range(n_lines):
            saved = held[:, j].copy()
            held[:, j] = mean[j]
            drop[te, j] = base - predict_scores(model, held.reshape(len(te), -1), classes)[r, col]
            held[:, j] = saved
    return drop.mean(axis=0), drop


def eliminate_lines(X: np.ndarray, y: np.ndarray, groups: np.ndarray, targets,
                    model_factory=line_token_model, keep_fraction: float = 0.7,
                    n_splits: int = 4, seed: int = 0) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Recursive line elimination driven by inner-CV occlusion.

    Occlusion of a single line underrates lines whose information is also
    carried by others. Removing the least important ``1 - keep_fraction`` of
    lines and re-ranking the rest lets those redundancies resolve. Returns the
    surviving line indices for every size in ``targets`` and a full ranking
    (1 = most valuable) of all lines.
    """
    n_lines = X.shape[1]
    remaining = np.arange(n_lines)
    discarded: list[int] = []   # least valuable first
    subsets = {}
    for target in sorted({int(t) for t in targets if 0 < t <= n_lines}, reverse=True):
        while remaining.size > target:
            importance, _ = line_occlusion(X[:, remaining], y, groups, model_factory, n_splits,
                                           seed)
            n_keep = max(target, int(remaining.size * keep_fraction))
            order = np.argsort(importance, kind="stable")
            discarded.extend(remaining[order[:remaining.size - n_keep]].tolist())
            remaining = np.sort(remaining[order[remaining.size - n_keep:]])
        subsets[target] = remaining.copy()
    if remaining.size > 1:
        importance, _ = line_occlusion(X[:, remaining], y, groups, model_factory, n_splits, seed)
        remaining = remaining[np.argsort(importance, kind="stable")]
    discarded.extend(remaining.tolist())
    rank = np.empty(n_lines, dtype=int)
    rank[np.asarray(discarded[::-1], dtype=int)] = np.arange(1, n_lines + 1)
    return subsets, rank
