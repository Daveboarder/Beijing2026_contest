"""Plasma temperature from Fe Boltzmann and Saha-Boltzmann plots.

The aging grade of a heat-resistant steel is a bulk microstructure property
(carbide precipitation, spheroidisation). Such changes alter how the laser
couples to the material, so they should show up in the plasma state -- its
excitation temperature and electron density -- rather than only in element
line intensities. This module turns raw spectra into those physical
descriptors:

1. Fe transitions are read from the air line database (``LIBS_data.db``, the
   same file ``lines_db.py`` uses).
2. Candidate lines are screened for blends with every steel/air element via
   theoretical optically-thin emissivities at a typical plasma state, and
   ground-multiplet lines prone to self-absorption are dropped.
3. Net peak heights (or areas) are measured above a local linear background.
4. A Boltzmann plot ``ln(I lambda / (g_k A_k))`` vs ``E_k`` is fitted per
   spectrum. Lines that sit systematically off the line in all spectra
   (bad ``A_k``, blends, self-absorption) are removed one by one until the
   combination is straight within a tolerance, keeping a wide energy span.
5. The spectrometer is not radiometrically calibrated. Lines are taken from
   one channel and a smooth ln-response in wavelength is fitted jointly with
   the Boltzmann lines of all spectra.
6. Fe II lines can be added on the Saha-Boltzmann axis ``E_k + E_ion`` with
   the electron density from the Stark width of H-alpha.

Any residual response bias is identical for every sample, so temperature
differences between aging levels remain meaningful even where the absolute
value is uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.constants as const
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths
from scipy.special import voigt_profile

from .lines_db import _connect, _get_eion, partition_function

KB_EV = const.k / const.e                 # eV/K
_KB_CGS = const.k * 1e7                   # erg/K
_H_CGS = const.h * 1e7                    # erg*s
_ME_CGS = const.electron_mass * 1e3       # g
HALPHA_NM = 656.279

# Rough steel + ambient-air composition (mass fraction), used only to judge
# whether a neighbouring transition can contaminate a candidate line.
DEFAULT_CONCENTRATIONS = {
    "Fe": 0.95, "Cr": 0.02, "Mn": 0.006, "Ni": 0.003, "Si": 0.003,
    "Mo": 0.005, "V": 0.003, "Cu": 0.001, "Ti": 0.0005, "Al": 0.0003,
    "C": 0.002, "H": 0.001, "O": 0.01, "N": 0.03,
}


# ----------------------------------------------------------------------------
# Line database
# ----------------------------------------------------------------------------


def load_transitions(db_path: str, elements) -> pd.DataFrame:
    """All neutral and singly ionised transitions of ``elements``."""
    con = _connect(str(db_path))
    marks = ",".join("?" * len(elements))
    frame = pd.read_sql(
        "SELECT Elem_name AS element, ion_state, Wavelength AS wavelength, "
        f"Ei, Ek, gi, gk, Ak FROM QuantParam WHERE Elem_name IN ({marks}) "
        "AND ion_state IN ('I', 'II')",
        con, params=list(elements),
    )
    return frame.dropna().sort_values("wavelength").reset_index(drop=True)


def emissivity(frame: pd.DataFrame, te: float, ne: float, db_path: str,
               concentrations: dict[str, float] | None = None) -> np.ndarray:
    """Relative optically-thin emissivity ``C f_ion g_k A_k exp(-E_k/kT) / (U lambda)``."""
    conc = concentrations or DEFAULT_CONCENTRATIONS
    out = np.zeros(len(frame))
    for elem, rows in frame.groupby("element").groups.items():
        sub = frame.loc[rows]
        u_i, u_ii = partition_function(elem, te, db_path)
        if u_i <= 0:
            continue
        s10 = saha_ratio(te, ne, _get_eion(elem, db_path), u_i, u_ii)
        neutral = (sub["ion_state"] == "I").to_numpy()
        frac = np.where(neutral, 1 / (1 + s10), s10 / (1 + s10))
        u = np.where(neutral, u_i, u_ii if u_ii > 0 else 1.0)
        out[rows] = (
            conc.get(elem, 0.0) * frac * sub["gk"] * sub["Ak"]
            * np.exp(-sub["Ek"] / (KB_EV * te)) / (u * sub["wavelength"])
        )
    return out


def saha_ratio(te: float, ne: float, e_ion: float, u_i: float, u_ii: float) -> float:
    """``n_II / n_I`` from the Saha equation (``ne`` in cm^-3)."""
    return (
        (2 * u_ii / (ne * u_i))
        * ((2 * np.pi * _ME_CGS * _KB_CGS * te) / _H_CGS ** 2) ** 1.5
        * np.exp(-e_ion / (KB_EV * te))
    )


def saha_offset(te, ne) -> np.ndarray:
    """``ln(2 (2 pi m_e k T / h^2)^1.5 / n_e)``: shift applied to ionic lines."""
    te = np.asarray(te, dtype=np.float64)
    return np.log(2 * ((2 * np.pi * _ME_CGS * _KB_CGS * te) / _H_CGS ** 2) ** 1.5 / ne)


# ----------------------------------------------------------------------------
# Instrument and line measurement
# ----------------------------------------------------------------------------


def instrument_fwhm(wavelength: np.ndarray, spectrum: np.ndarray, bounds,
                    n_peaks: int = 60, quantile: float = 0.2) -> list[float]:
    """Instrumental FWHM (nm) per channel from the narrowest strong peaks.

    Most strong peaks are instrument-limited; blends and Stark-broadened lines
    only ever make a peak wider, so a low quantile of the widths is used.
    """
    widths = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = spectrum[a:b]
        peaks, props = find_peaks(seg, prominence=np.percentile(seg, 90) - np.median(seg))
        top = peaks[np.argsort(props["prominences"])[::-1][:n_peaks]]
        w_px = peak_widths(seg, top, rel_height=0.5)[0]
        step = float(np.median(np.diff(wavelength[a:b])))
        widths.append(float(np.quantile(w_px, quantile)) * step)
    return widths


@dataclass
class MeasureConfig:
    """Pixel geometry of the integration window and background flanks."""

    # "height" (net peak maximum) or "area". Lines in one channel share the
    # instrument profile, so height is proportional to area but picks up far
    # less of a partially resolved neighbour.
    intensity: str = "height"
    window_fwhm: float = 1.0      # half-width of the integration window, in FWHM
    bg_gap_px: int = 1            # pixels between window and background flank
    bg_width_px: int = 5          # flank width
    bg_quantile: float = 0.3      # robust flank level (ignores neighbouring peaks)
    max_shift_px: float = 1.5     # allowed centroid walk from the database value


def assign_pixels(lines: pd.DataFrame, wavelength: np.ndarray, bounds, fwhm: list[float],
                  measure: MeasureConfig, channels=None) -> pd.DataFrame:
    """Channel, centre pixel and window half-width for every line.

    Channels overlap; the finest-sampled channel that holds the full window and
    both background flanks wins. Lines outside every allowed channel are dropped.
    """
    allowed = set(channels) if channels else set(range(1, len(bounds)))
    rows = []
    for rec in lines.itertuples():
        best = None
        for ch, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
            if ch not in allowed:
                continue
            seg = wavelength[a:b]
            step = float(np.median(np.diff(seg)))
            half = max(1, int(round(measure.window_fwhm * fwhm[ch - 1] / step)))
            reach = half + measure.bg_gap_px + measure.bg_width_px + 2
            pix = int(np.argmin(np.abs(seg - rec.wavelength)))
            if abs(seg[pix] - rec.wavelength) > step or pix < reach or pix >= seg.size - reach:
                continue
            if best is None or step < best[3]:
                best = (ch, a + pix, half, step)
        if best is not None:
            rows.append((rec.Index, *best))
    geo = pd.DataFrame(rows, columns=["idx", "channel", "pixel", "half_px", "step_nm"]).set_index("idx")
    return lines.join(geo, how="inner")


def refine_centres(lines: pd.DataFrame, reference: np.ndarray, max_shift_px: float) -> pd.DataFrame:
    """Snap each line to the local maximum of ``reference``; drop lines with no peak nearby."""
    reach = int(np.ceil(max_shift_px))
    keep, pixels = [], []
    for rec in lines.itertuples():
        lo, hi = rec.pixel - reach, rec.pixel + reach + 1
        peak = lo + int(np.argmax(reference[lo:hi]))
        is_max = reference[peak] >= reference[peak - 1] and reference[peak] >= reference[peak + 1]
        ok = is_max and abs(peak - rec.pixel) <= max_shift_px
        keep.append(ok)
        pixels.append(peak)
    out = lines.assign(pixel=pixels)[keep]
    return out


def integrate_lines(spectra: np.ndarray, lines: pd.DataFrame, measure: MeasureConfig):
    """Background-corrected integrated area and peak height per line.

    ``spectra`` is ``(n_spectra, n_wavelengths)``; returns two ``(n_spectra, n_lines)``
    arrays. Areas are in counts*nm so channels with different dispersion compare.
    """
    x = np.asarray(spectra, dtype=np.float64)
    n_lines = len(lines)
    area = np.full((x.shape[0], n_lines), np.nan)
    height = np.full((x.shape[0], n_lines), np.nan)
    g, w = measure.bg_gap_px, measure.bg_width_px
    for j, rec in enumerate(lines.itertuples()):
        p, h = int(rec.pixel), int(rec.half_px)
        left = x[:, p - h - g - w:p - h - g]
        right = x[:, p + h + g + 1:p + h + g + w + 1]
        bl = np.quantile(left, measure.bg_quantile, axis=1)
        br = np.quantile(right, measure.bg_quantile, axis=1)
        xl, xr = p - h - g - (w + 1) / 2, p + h + g + (w + 1) / 2
        idx = np.arange(p - h, p + h + 1)
        bg = bl[:, None] + (br - bl)[:, None] * (idx - xl) / (xr - xl)
        net = x[:, idx] - bg
        area[:, j] = net.sum(axis=1) * rec.step_nm
        height[:, j] = net[:, h]
    return area, height


def line_intensity(spectra: np.ndarray, lines: pd.DataFrame, measure: MeasureConfig) -> np.ndarray:
    """Net line intensity as configured by ``measure.intensity``."""
    area, height = integrate_lines(spectra, lines, measure)
    if measure.intensity == "height":
        return height
    if measure.intensity == "area":
        return area
    raise ValueError(f"Unknown intensity measure '{measure.intensity}'")


def channel_noise(spectra: np.ndarray, bounds) -> np.ndarray:
    """Robust per-spectrum, per-channel noise from first differences, ``(n, n_channels)``."""
    x = np.asarray(spectra, dtype=np.float64)
    out = np.zeros((x.shape[0], len(bounds) - 1))
    for c, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        d = np.diff(x[:, a:b], axis=1)
        out[:, c] = 1.4826 * np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1) / np.sqrt(2)
    return out


# ----------------------------------------------------------------------------
# Candidate screening
# ----------------------------------------------------------------------------


@dataclass
class ScreenConfig:
    te: float = 9000.0
    ne: float = 1e17
    top_n: int = 100000            # strongest theoretical lines considered
    blend_fwhm: float = 0.6        # neighbours closer than this many FWHM interfere
    max_interference: float = 0.5  # tolerated neighbour/own emissivity ratio
    min_ei: float = 0.12           # drop the a5D ground multiplet (self-absorption)
    min_snr: float = 10.0          # median peak SNR over samples
    # A clear peak at a line predicted this much weaker than the element's
    # strongest line in the same channels belongs to some other transition.
    min_rel_emissivity: float = 2e-3
    channels: tuple | None = None  # 1-based channels allowed; None = all
    concentrations: dict = field(default_factory=lambda: dict(DEFAULT_CONCENTRATIONS))


def screen_candidates(transitions: pd.DataFrame, element: str, ion_state: str,
                      wavelength: np.ndarray, bounds, fwhm: list[float],
                      reference: np.ndarray, sample_spectra: np.ndarray,
                      db_path: str, screen: ScreenConfig, measure: MeasureConfig) -> pd.DataFrame:
    """Isolated, measurable lines of one species, with the reason each other line failed.

    Returns every considered line with a ``status`` column; ``status == 'ok'``
    marks the usable candidates.
    """
    all_lines = transitions.copy()
    all_lines["emissivity"] = emissivity(all_lines, screen.te, screen.ne, db_path,
                                         screen.concentrations)
    species = all_lines[(all_lines["element"] == element) & (all_lines["ion_state"] == ion_state)]
    species = species[(species["wavelength"] >= wavelength.min()) & (species["wavelength"] <= wavelength.max())]
    species = species.nlargest(screen.top_n, "emissivity")
    species = assign_pixels(species, wavelength, bounds, fwhm, measure, screen.channels)

    wl_all = all_lines["wavelength"].to_numpy()
    em_all = all_lines["emissivity"].to_numpy()
    interference = []
    for idx, rec in zip(species.index, species.itertuples()):
        reach = screen.blend_fwhm * fwhm[rec.channel - 1]
        near = np.abs(wl_all - rec.wavelength) <= reach
        near[idx] = False
        interference.append(em_all[near].sum() / max(rec.emissivity, 1e-300))
    species = species.assign(interference=interference)

    in_span = np.zeros(len(all_lines), dtype=bool)
    for ch in species["channel"].unique():
        a, b = bounds[ch - 1], bounds[ch]
        in_span |= all_lines["wavelength"].between(wavelength[a:b].min(), wavelength[a:b].max()).to_numpy()
    strongest = all_lines.loc[in_span & (all_lines["element"] == element).to_numpy(), "emissivity"].max()
    species = species.assign(rel_emissivity=species["emissivity"] / strongest)

    status = np.full(len(species), "ok", dtype=object)
    status[species["rel_emissivity"].to_numpy() < screen.min_rel_emissivity] = "implausible"
    status[(status == "ok") & (species["Ei"].to_numpy() < screen.min_ei)] = "ground_multiplet"
    status[(status == "ok") & (species["interference"].to_numpy() > screen.max_interference)] = "blended"
    species = species.assign(status=status)

    # Centre refinement and SNR are only meaningful for the survivors.
    ok = species[species["status"] == "ok"]
    refined = refine_centres(ok, reference, measure.max_shift_px)
    species.loc[ok.index.difference(refined.index), "status"] = "no_peak"
    species.loc[refined.index, "pixel"] = refined["pixel"]

    ok = species[species["status"] == "ok"]
    if len(ok):
        _, height = integrate_lines(sample_spectra, ok, measure)
        noise = channel_noise(sample_spectra, bounds)[:, ok["channel"].to_numpy() - 1]
        snr = np.median(height / np.maximum(noise, 1e-9), axis=0)
        species.loc[ok.index, "snr"] = snr
        species.loc[ok.index[snr < screen.min_snr], "status"] = "weak"
    return species.sort_values("wavelength")


# ----------------------------------------------------------------------------
# Boltzmann / Saha-Boltzmann fitting
# ----------------------------------------------------------------------------


def boltzmann_y(lines: pd.DataFrame, area: np.ndarray) -> np.ndarray:
    """``ln(I lambda / (g_k A_k))``; NaN where the net area is not positive."""
    factor = (lines["wavelength"] / (lines["gk"] * lines["Ak"])).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area > 0, np.log(area * factor), np.nan)


@dataclass
class LineFit:
    slope: np.ndarray
    intercept: np.ndarray
    slope_se: np.ndarray
    r2: np.ndarray
    n: np.ndarray

    @property
    def temperature(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.slope < 0, -1.0 / (KB_EV * self.slope), np.nan)

    @property
    def temperature_se(self) -> np.ndarray:
        return self.temperature ** 2 * KB_EV * self.slope_se


def fit_rows(x: np.ndarray, y: np.ndarray) -> LineFit:
    """Ordinary least squares of every row of ``y`` against ``x`` (NaN-aware).

    ``x`` may be shared ``(n_lines,)`` or per row ``(n_rows, n_lines)``.
    """
    y = np.atleast_2d(y)
    x = np.broadcast_to(x, y.shape)
    m = np.isfinite(y) & np.isfinite(x)
    n = m.sum(axis=1).astype(float)
    xv, yv = np.where(m, x, 0.0), np.where(m, y, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xm, ym = xv.sum(1) / n, yv.sum(1) / n
        sxx = (xv * xv).sum(1) - n * xm ** 2
        sxy = (xv * yv).sum(1) - n * xm * ym
        syy = (yv * yv).sum(1) - n * ym ** 2
        slope = sxy / sxx
        intercept = ym - slope * xm
        resid = np.where(m, y - intercept[:, None] - slope[:, None] * x, 0.0)
        sse = (resid ** 2).sum(1)
        r2 = 1 - sse / syy
        se = np.sqrt(sse / (n - 2) / sxx)
    return LineFit(slope, intercept, se, r2, n.astype(int))


def select_lines(x: np.ndarray, y: np.ndarray, tol: float = 0.25, min_lines: int = 6,
                 min_span: float = 1.5, removable: np.ndarray | None = None,
                 basis: np.ndarray | None = None):
    """Backward elimination of lines that do not sit on the Boltzmann line.

    The spectra here average 200 shots, so random noise is tiny and what bends
    a Boltzmann plot is *systematic*: inaccurate ``A_k``, unresolved blends,
    self-absorption of strong low-lying lines. Such a line is off the fit by
    about the same amount in every spectrum. Each step therefore fits every
    spectrum on the active set (refitting the shared response when ``basis``
    is given), takes the median residual of each line over all spectra, and
    drops the worst line -- until every remaining line lies within ``tol``
    (ln units; 0.25 is about the 25 % ``A_k`` accuracy of a NIST grade C line).

    Criteria that reward a small slope uncertainty keep improving as lines are
    removed and collapse onto a handful of lines; the tolerance stops on
    physical grounds instead. ``min_span`` (eV) protects the energy lever arm.

    Returns ``(mask, path, coef)``: ``path`` has one row per step and ``coef``
    are the response coefficients of the final set (empty without ``basis``).
    """
    n_lines = y.shape[1]
    active = np.ones(n_lines, dtype=bool)
    removable = np.ones(n_lines, dtype=bool) if removable is None else np.asarray(removable, bool)
    basis = np.zeros((n_lines, 0)) if basis is None else basis
    xs = x if x.ndim == 1 else np.nanmedian(x, axis=0)
    path = []
    removed = -1
    while True:
        coef = fit_response(xs[active], y[:, active], basis[active])
        yc = y - basis @ coef
        xm = x[..., active] if x.ndim == 2 else x[active]
        fit = fit_rows(xm, yc[:, active])
        resid = yc - fit.intercept[:, None] - fit.slope[:, None] * np.broadcast_to(x, y.shape)
        med = np.nanmedian(resid, axis=0)
        path.append({"n_lines": int(active.sum()), "removed": removed,
                     "max_abs_residual": float(np.nanmax(np.abs(med[active]))),
                     "median_r2": float(np.nanmedian(fit.r2)),
                     "median_T": float(np.nanmedian(fit.temperature)),
                     "median_slope_se": float(np.nanmedian(fit.slope_se))})
        if active.sum() <= min_lines:
            break
        order = np.argsort(-np.abs(np.where(active & removable, med, 0.0)))
        worst = None
        for j in order:
            if not (active[j] and removable[j]) or abs(med[j]) <= tol:
                break
            trial = active.copy()
            trial[j] = False
            if np.ptp(xs[trial]) >= min_span:
                worst = j
                break
        if worst is None:
            break
        active[worst] = False
        removed = int(worst)
    return active, pd.DataFrame(path), coef


def response_basis(wavelength_nm, lo: float, hi: float, degree: int) -> np.ndarray:
    """Legendre polynomials 1..degree of wavelength scaled to [-1, 1] on ``[lo, hi]``.

    The constant term is left out on purpose: it is absorbed by each
    spectrum's intercept.
    """
    t = 2 * (np.asarray(wavelength_nm, dtype=np.float64) - lo) / (hi - lo) - 1
    return np.stack([np.polynomial.legendre.Legendre.basis(d)(t) for d in range(1, degree + 1)], axis=1) \
        if degree > 0 else np.zeros((t.size, 0))


def fit_response(x: np.ndarray, y: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Shared ln-response coefficients from ``y[s, l] = a_s + b_s x_l + basis_l . c``.

    The spectrometer is not radiometrically calibrated, so every line carries
    an unknown ``ln R(lambda)``. It is identifiable from the data because
    Boltzmann plots of many spectra share it while their slopes and intercepts
    differ, and because lines sharing an upper level differ only by
    ``R(lambda)`` and atomic data. One joint least-squares problem is solved
    with per-spectrum intercepts and slopes as nuisance parameters.
    """
    n_s, n_l = y.shape
    d = basis.shape[1]
    if d == 0:
        return np.zeros(0)
    s_idx, l_idx = np.nonzero(np.isfinite(y))
    a = np.zeros((s_idx.size, 2 * n_s + d))
    rows = np.arange(s_idx.size)
    a[rows, s_idx] = 1.0
    a[rows, n_s + s_idx] = x[l_idx]
    a[:, 2 * n_s:] = basis[l_idx]
    coef, *_ = np.linalg.lstsq(a, y[s_idx, l_idx], rcond=None)
    return coef[2 * n_s:]


def saha_boltzmann_fit(x_i, y_i, x_ii, y_ii, e_ion: float, ne, te0,
                       n_iter: int = 30, tol: float = 1.0):
    """Joint Fe I + Fe II fit, iterating the T-dependent Saha shift of ionic lines.

    ``ne`` and ``te0`` are per-row arrays. Returns ``(LineFit, x, y)`` with the
    final, shifted coordinates.
    """
    ne = np.asarray(ne, dtype=np.float64)
    te = np.asarray(te0, dtype=np.float64).copy()
    x = np.concatenate([np.broadcast_to(x_i, y_i.shape), np.broadcast_to(x_ii + e_ion, y_ii.shape)], axis=1)
    for _ in range(n_iter):
        te_safe = np.where(np.isfinite(te) & (te > 0), te, 9000.0)
        y = np.concatenate([y_i, y_ii - saha_offset(te_safe, ne)[:, None]], axis=1)
        fit = fit_rows(x, y)
        new = fit.temperature
        done = np.nanmax(np.abs(new - te)) < tol if np.isfinite(new).any() else True
        te = new
        if done:
            break
    return fit, x, y


# ----------------------------------------------------------------------------
# Electron density
# ----------------------------------------------------------------------------


def electron_density_halpha(wavelength: np.ndarray, spectrum: np.ndarray, instrument_fwhm_nm: float,
                            span_nm: float = 4.0) -> dict:
    """``n_e`` (cm^-3) from the Lorentzian (Stark) FWHM of H-alpha.

    A Voigt profile with the Gaussian part fixed to the instrument function and
    a linear background is fitted robustly. Density follows Gigosos et al.
    (2003): ``FWHM[nm] = 0.549 (n_e / 1e17 cm^-3)^0.67954``. The detector gate
    is 1 ms long, so this is a time-integrated, late-plasma estimate.
    """
    sel = np.abs(wavelength - HALPHA_NM) <= span_nm
    wl = wavelength[sel].astype(np.float64)
    s = spectrum[sel].astype(np.float64)
    order = np.argsort(wl)
    wl, s = wl[order], s[order]
    sigma = instrument_fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    edge = np.r_[s[:5], s[-5:]].mean()

    def model(p):
        amp, centre, gamma, b0, b1 = p
        return amp * voigt_profile(wl - centre, sigma, gamma) + b0 + b1 * (wl - HALPHA_NM)

    p0 = [max(s.max() - edge, 1.0) * 0.5, HALPHA_NM, 0.3, edge, 0.0]
    lower = [0.0, HALPHA_NM - 0.3, 1e-3, -np.inf, -np.inf]
    upper = [np.inf, HALPHA_NM + 0.3, 3.0, np.inf, np.inf]
    try:
        res = least_squares(lambda p: model(p) - s, p0, bounds=(lower, upper),
                            loss="soft_l1", f_scale=max(np.std(s) * 0.1, 1e-6))
        amp, centre, gamma, _, _ = res.x
        fwhm_l = 2 * gamma
        ne = 1e17 * (fwhm_l / 0.549) ** (1 / 0.67954)
        ok = res.success and amp > 0 and 2e-3 < gamma < 2.99
    except (ValueError, RuntimeError):
        fwhm_l, ne, centre, ok = np.nan, np.nan, np.nan, False
    return {"ne": float(ne) if ok else np.nan, "halpha_fwhm_l": float(fwhm_l), "halpha_centre": float(centre),
            "halpha_ok": bool(ok)}
