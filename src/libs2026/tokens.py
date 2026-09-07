"""Turn each sample into a stack of spectral-line tokens.

Ported from the LIBS foundation-model project (``data/line_features.py`` and
``data/line_tokenization.py``). Instead of 12282 wavelength bins, a spectrum
becomes ``n_lines`` tokens, each describing one theoretical transition:

    static (9, from the dictionary, identical for every spectrum)
        0 central_wavelength   1 Ei   2 Ek   3 log10_gi   4 log10_gk
        5 log10_Ak   6 log10_theoretical_intensity   7 atomic_number
        8 ion_binary
    dynamic (5, Voigt fit of this spectrum at the line centre)
        9 max_intensity   10 fwhm   11 r2   12 delta_lambda   13 rmse
    mask and continuum (2, added here)
        14 fit_valid   15 local_continuum

Two deliberate departures from upstream:

* The Voigt model has no constant term, so a continuum pedestal would bias the
  amplitude and inflate the RMSE. A local linear baseline is removed inside
  each fit window and its level at the line centre is kept as its own channel,
  so the continuum information stays available to the model rather than being
  discarded (the preprocessing sweep showed the continuum is worth ~5 accuracy
  points on this dataset, which is why ``baseline: none`` is the default).
* Window bounds are resolved once per line against the *spectrometer channel*
  that covers it. The recorded axis is three concatenated overlapping ranges
  and is therefore not monotonic, so a global nearest-index search could span a
  channel boundary and would also be far too slow for millions of fits.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.special import wofz

from .config import PROJECT_ROOT, Config
from .data import load_index, load_shots, load_wavelength
from .lines_db import (
    STATIC_FEATURE_NAMES,
    STEEL_ELEMENTS,
    LineDictionary,
    build_line_dictionary,
)
from .preprocessing import DEFAULT_CHANNEL_BOUNDS, Preprocessor

DYNAMIC_FEATURE_NAMES = ("max_intensity", "fwhm", "r2", "delta_lambda", "rmse")
EXTRA_FEATURE_NAMES = ("fit_valid", "local_continuum")
# Channels that vary per sample and depth; these are what the CNN receives.
SAMPLE_FEATURE_NAMES = DYNAMIC_FEATURE_NAMES + EXTRA_FEATURE_NAMES

N_STATIC = len(STATIC_FEATURE_NAMES)
N_DYNAMIC = len(DYNAMIC_FEATURE_NAMES)
N_SAMPLE = len(SAMPLE_FEATURE_NAMES)
N_TOKEN_FEATURES = N_STATIC + N_SAMPLE

F_MAX_INT, F_FWHM, F_R2, F_DELTA, F_RMSE, F_VALID, F_CONTINUUM = range(N_SAMPLE)


# ----------------------------------------------------------------------------
# Voigt fitting
# ----------------------------------------------------------------------------


def voigt(x, x0, amplitude, gamma, sigma):
    """Voigt profile, guarded against overflow while curve_fit explores."""
    sigma = max(float(sigma), 1e-6)
    gamma = max(float(gamma), 0.0)
    z = (x - x0 + 1j * gamma) / (sigma * np.sqrt(2))
    with np.errstate(over="ignore", invalid="ignore"):
        profile = wofz(z).real
    out = amplitude * profile / (sigma * np.sqrt(2 * np.pi))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fwhm_voigt(gamma: float, sigma: float) -> float:
    """Olivero & Longbothum (1977) approximation, in nm."""
    return 0.5346 * (2 * gamma) + np.sqrt(
        0.2166 * (2 * gamma) ** 2 + (2 * sigma * np.sqrt(2 * np.log(2))) ** 2
    )


def window_bounds(
    wavelength: np.ndarray,
    centre_nm: float,
    window_nm: float,
    channel_bounds=DEFAULT_CHANNEL_BOUNDS,
    min_points: int = 5,
) -> tuple[int, int] | None:
    """Index range covering ``centre_nm +/- window_nm`` inside one channel.

    When several spectrometer channels overlap the line, the one where the line
    sits furthest from an edge is used, since edge pixels are the noisiest and
    a window clipped by a channel boundary would only see half the profile.
    """
    best: tuple[int, int] | None = None
    best_margin = -np.inf
    for start, stop in zip(channel_bounds[:-1], channel_bounds[1:]):
        axis = wavelength[start:stop]
        lo_wl, hi_wl = float(axis[0]), float(axis[-1])
        if not (lo_wl <= centre_nm <= hi_wl):
            continue
        margin = min(centre_nm - lo_wl, hi_wl - centre_nm)
        lo = start + int(np.searchsorted(axis, centre_nm - window_nm, side="left"))
        hi = start + int(np.searchsorted(axis, centre_nm + window_nm, side="right"))
        lo, hi = max(lo, start), min(hi, stop)
        if hi - lo < min_points:
            continue
        if margin > best_margin:
            best, best_margin = (lo, hi), margin
    return best


def fit_line(
    spectrum: np.ndarray,
    wavelength: np.ndarray,
    lo: int,
    hi: int,
    centre_nm: float,
    gamma_init: float,
    sigma_init: float,
    r2_min: float,
    out: np.ndarray,
    min_snr: float = 3.0,
    maxfev: int = 600,
) -> None:
    """Fit one Voigt profile into ``out`` (length ``N_SAMPLE``)."""
    out[:] = 0.0
    x = wavelength[lo:hi]
    y = spectrum[lo:hi].astype(np.float64)
    if y.size < 4 or not np.all(np.isfinite(y)):
        return

    # Local linear baseline from the window edges; the Voigt has no offset term.
    edge = max(1, y.size // 6)
    x_edge = np.concatenate([x[:edge], x[-edge:]])
    y_edge = np.concatenate([y[:edge], y[-edge:]])
    slope, intercept = np.polyfit(x_edge, y_edge, 1)
    baseline = slope * x + intercept
    out[F_CONTINUUM] = float(slope * centre_nm + intercept)
    y = y - baseline

    y_max = float(np.max(y))
    if y_max <= 0:
        return
    # Noise from the window edges, which hold continuum rather than the line.
    noise = float(np.std(y_edge - (slope * x_edge + intercept)))
    if noise > 0 and y_max < min_snr * noise:
        return

    lb = [float(x[0]), 0.0, 1e-4, 1e-4]
    ub = [float(x[-1]), y_max * 100.0, 0.5, 0.05]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                voigt, x, y,
                p0=[float(x[int(np.argmax(y))]), y_max, gamma_init, sigma_init],
                bounds=(lb, ub),
                maxfev=maxfev, ftol=1e-6, xtol=1e-6,
            )
    except (RuntimeError, ValueError, TypeError):
        return

    if not np.all(np.isfinite(popt)):
        return
    x0_fit, amp, gamma, sigma = popt
    if amp <= 0 or sigma <= 0:
        return
    fit_y = voigt(x, *popt)
    if not np.all(np.isfinite(fit_y)):
        return

    ss_res = float(np.sum((y - fit_y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 0 else -np.inf
    if not np.isfinite(r2) or r2 < r2_min:
        return

    out[F_MAX_INT] = float(amp)
    out[F_FWHM] = float(fwhm_voigt(gamma, sigma))
    out[F_R2] = float(r2)
    out[F_DELTA] = float(x0_fit - centre_nm)
    out[F_RMSE] = float(np.sqrt(ss_res / y.size))
    out[F_VALID] = 1.0


def fit_spectra(
    spectra: np.ndarray,
    wavelength: np.ndarray,
    bounds: list[tuple[int, int]],
    centres: np.ndarray,
    gamma_init: float,
    sigma_init: float,
    r2_min: float,
    min_snr: float = 3.0,
    maxfev: int = 600,
) -> np.ndarray:
    """Fit every line in every row; returns ``(n_rows, n_lines, N_SAMPLE)``."""
    n_rows, n_lines = spectra.shape[0], len(bounds)
    out = np.zeros((n_rows, n_lines, N_SAMPLE), dtype=np.float32)
    scratch = np.zeros(N_SAMPLE, dtype=np.float64)
    for r in range(n_rows):
        spec = spectra[r]
        for j, (lo, hi) in enumerate(bounds):
            fit_line(spec, wavelength, lo, hi, float(centres[j]),
                     gamma_init, sigma_init, r2_min, scratch, min_snr, maxfev)
            out[r, j] = scratch
    return out


# ----------------------------------------------------------------------------
# Token set
# ----------------------------------------------------------------------------


@dataclass
class FitConfig:
    """Voigt-fit settings (part of the cache key)."""

    window_nm: float = 0.3
    gamma_init: float = 0.1
    sigma_init: float = 0.006
    # Upstream rejects fits below R^2 0.85 because masked-feature pretraining
    # needs clean regression targets. Here the fits are inputs, not targets, so
    # a mediocre fit is kept and its R^2 channel tells the network how far to
    # trust it. Discarding them instead zeroes the amplitude of ~80% of tokens
    # (measured: 20% valid at 0.85 against 74% at 0.0).
    r2_min: float = 0.0
    # Lines buried under this many noise sigmas are not fitted at all: the
    # transition is simply absent from the sample, and skipping is both the
    # physically honest answer and the bulk of the speed-up.
    min_snr: float = 3.0
    maxfev: int = 600


def fit_config_from_config(cfg: Config) -> FitConfig:
    """Voigt-fit settings from the ``tokens.fit`` config block."""
    fit = dict(cfg.get("tokens", {}).get("fit", {}))
    return FitConfig(**{k: v for k, v in fit.items() if k in FitConfig.__dataclass_fields__})


def line_dictionary_from_config(cfg: Config, verbose: bool = True) -> LineDictionary:
    """Build the line dictionary described by the ``tokens`` config block.

    The dictionary is clipped to the wavelength span the spectrometer actually
    recorded, so no token can refer to a region that was never measured.
    """
    tok = dict(cfg.get("tokens", {}))
    db_path = Path(tok.get("db_path", ""))
    if not db_path.is_absolute():
        db_path = (PROJECT_ROOT / db_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Line database not found at {db_path}. Set tokens.db_path in the config."
        )
    wavelength = load_wavelength(cfg)
    return build_line_dictionary(
        db_path,
        elements=tuple(tok.get("elements", STEEL_ELEMENTS)),
        wl_min=float(wavelength.min()),
        wl_max=float(wavelength.max()),
        # YAML only reads scientific notation as a float when it carries a sign
        # (1.0e+17, not 1.0e17), so coerce rather than trust the parsed type.
        te_range=tuple(float(v) for v in tok.get("te_range", (6500.0, 11000.0))),
        ne_range=tuple(float(v) for v in tok.get("ne_range", (1e17, 5e17))),
        n_te=int(tok.get("n_te", 10)),
        n_ne=int(tok.get("n_ne", 10)),
        percent=float(tok.get("percent", 10.0)),
        min_keep=int(tok.get("min_keep", 10)),
        cache_dir=cfg.cache_dir,
        verbose=verbose,
    )


class TokenSet:
    """Per-sample line tokens with sample bookkeeping.

    ``X`` holds only the channels that vary per sample and depth, shaped
    ``(n_samples, n_rows, n_lines, N_SAMPLE)``. The 9 static physics channels
    are stored once in ``static`` and broadcast by the model, which keeps the
    tensor roughly 2.3x smaller than materialising all 16 channels.
    """

    def __init__(self, X, y, sample_ids, split, static, line_wavelength,
                 line_element, line_ion_state, feature_mean=None, feature_std=None):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = y
        self.sample_ids = np.asarray(sample_ids)
        self.split = np.asarray(split)
        self.static = np.asarray(static, dtype=np.float32)
        self.line_wavelength = np.asarray(line_wavelength, dtype=np.float32)
        self.line_element = np.asarray(line_element).astype(str)
        self.line_ion_state = np.asarray(line_ion_state).astype(str)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        # One token stack per physical sample, so group id equals row index.
        self.groups = np.arange(len(self.sample_ids))

    def subset(self, split: str) -> TokenSet:
        mask = self.split == split
        y = self.y[mask] if self.y is not None else None
        return TokenSet(
            self.X[mask], y, self.sample_ids[mask], self.split[mask],
            self.static, self.line_wavelength, self.line_element,
            self.line_ion_state, self.feature_mean, self.feature_std,
        )

    def select_lines(self, keep: np.ndarray) -> TokenSet:
        """Restrict to a subset of lines (used for the reduced-line ablation)."""
        keep = np.asarray(keep)
        mean = self.feature_mean[keep] if self.feature_mean is not None else None
        std = self.feature_std[keep] if self.feature_std is not None else None
        return TokenSet(
            self.X[:, :, keep], self.y, self.sample_ids, self.split,
            self.static[keep], self.line_wavelength[keep],
            self.line_element[keep], self.line_ion_state[keep], mean, std,
        )

    def as_flat(self) -> np.ndarray:
        """Flatten to ``(n_samples, n_rows * n_lines * N_SAMPLE)`` for sklearn APIs."""
        return self.X.reshape(self.X.shape[0], -1)

    @property
    def token_shape(self) -> tuple[int, int, int]:
        return int(self.X.shape[1]), int(self.X.shape[2]), int(self.X.shape[3])

    @property
    def n_lines(self) -> int:
        return int(self.X.shape[2])

    def valid_fraction(self) -> float:
        return float(self.X[..., F_VALID].mean())

    def __repr__(self) -> str:
        rows, lines, feats = self.token_shape
        return (f"TokenSet(n_samples={len(self.sample_ids)}, rows={rows}, "
                f"lines={lines}, features={feats})")


def _tokenize_sample(cfg: Config, sample_id: str, pre: Preprocessor, shot_bin: int,
                     wavelength: np.ndarray, bounds: list[tuple[int, int]],
                     centres: np.ndarray, fit_cfg: FitConfig) -> np.ndarray:
    shots = pre(load_shots(cfg, sample_id, mmap=False))
    if shot_bin > 1:
        n = (shots.shape[0] // shot_bin) * shot_bin
        shots = shots[:n].reshape(n // shot_bin, shot_bin, shots.shape[1]).mean(axis=1)
    return fit_spectra(
        np.asarray(shots, dtype=np.float32), wavelength, bounds, centres,
        fit_cfg.gamma_init, fit_cfg.sigma_init, fit_cfg.r2_min,
        fit_cfg.min_snr, fit_cfg.maxfev,
    )


def build_tokens(
    cfg: Config,
    dictionary: LineDictionary,
    pre: Preprocessor | None = None,
    shot_bin: int = 4,
    fit_cfg: FitConfig | None = None,
    n_jobs: int = 12,
    use_cache: bool = True,
    verbose: bool = True,
) -> TokenSet:
    """Voigt-fit every line of every (binned) shot and cache the result."""
    pre = pre or Preprocessor.from_config(cfg)
    fit_cfg = fit_cfg or FitConfig()
    channel_bounds = tuple(cfg["data"].get("channel_bounds", DEFAULT_CHANNEL_BOUNDS))
    wavelength = load_wavelength(cfg)

    key = json.dumps(
        {
            "pre": asdict(pre),
            "shot_bin": shot_bin,
            "fit": asdict(fit_cfg),
            "dict": dictionary.config_hash,
            "kind": "tokens",
            "version": 1,
        },
        sort_keys=True, default=str,
    )
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_file = cfg.cache_dir / "tokens" / f"{digest}.npz"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_file.exists():
        if verbose:
            print(f"token cache hit: {cache_file}")
        blob = np.load(cache_file, allow_pickle=False)
        y = blob["y"]
        return TokenSet(
            blob["X"], np.where(y < 0, np.nan, y), blob["sample_ids"], blob["split"],
            blob["static"], blob["line_wavelength"], blob["line_element"],
            blob["line_ion_state"], blob["feature_mean"], blob["feature_std"],
        )

    # Resolve each line's fit window once; lines outside every channel are dropped.
    resolved, keep = [], []
    for j, centre in enumerate(dictionary.wavelength):
        wb = window_bounds(wavelength, float(centre), fit_cfg.window_nm, channel_bounds)
        if wb is not None:
            resolved.append(wb)
            keep.append(j)
    keep = np.asarray(keep, dtype=int)
    if keep.size == 0:
        raise RuntimeError("No dictionary line falls inside the recorded wavelength range.")
    dropped = dictionary.n_lines - keep.size
    if verbose and dropped:
        print(f"  {dropped} lines outside the recorded channels were dropped")

    centres = dictionary.wavelength[keep]
    index = load_index(cfg)
    if verbose:
        n_rows = 200 // max(shot_bin, 1)
        print(f"tokenizing {len(index)} samples x ~{n_rows} rows x {keep.size} lines "
              f"({len(index) * n_rows * keep.size / 1e6:.1f}M Voigt fits)")

    arrays = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_tokenize_sample)(
            cfg, sid, pre, shot_bin, wavelength, resolved, centres, fit_cfg,
        )
        for sid in index["sample_id"]
    )
    X = np.stack(arrays, axis=0).astype(np.float32)
    y = index["label"].to_numpy(dtype=float)
    split = index["split"].to_numpy()

    # Normalisation stats from the training split only. Dynamic channels are
    # averaged over successful fits alone, as upstream does; the mask and the
    # continuum use every entry.
    train_mask = split == "train"
    feature_mean, feature_std = _feature_stats(X[train_mask])

    np.savez_compressed(
        cache_file,
        X=X, y=np.where(np.isnan(y), -1, y),
        sample_ids=index["sample_id"].to_numpy().astype(str),
        split=split.astype(str),
        static=dictionary.static[keep],
        line_wavelength=centres,
        line_element=dictionary.element[keep].astype(str),
        line_ion_state=dictionary.ion_state[keep].astype(str),
        feature_mean=feature_mean, feature_std=feature_std,
    )
    with open(cache_file.with_suffix(".json"), "w", encoding="utf-8") as fh:
        fh.write(key)

    tokens = TokenSet(
        X, y, index["sample_id"].to_numpy(), split, dictionary.static[keep],
        centres, dictionary.element[keep], dictionary.ion_state[keep],
        feature_mean, feature_std,
    )
    if verbose:
        print(f"{tokens}  fit_valid={tokens.valid_fraction():.1%}")
    return tokens


def _feature_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-line, per-channel mean/std, shaped ``(n_lines, N_SAMPLE)``.

    Statistics are per line because line amplitudes span orders of magnitude:
    one global scale would leave weak lines numerically invisible.
    """
    n_lines = X.shape[2]
    mean = np.zeros((n_lines, N_SAMPLE), dtype=np.float32)
    std = np.ones((n_lines, N_SAMPLE), dtype=np.float32)
    valid = X[..., F_VALID] > 0.5
    for c in range(N_SAMPLE):
        channel = X[..., c]
        for j in range(n_lines):
            col = channel[:, :, j]
            if c < N_DYNAMIC:
                col = col[valid[:, :, j]]
            if col.size == 0:
                continue
            mu = float(col.mean())
            sigma = float(col.std())
            mean[j, c] = mu
            std[j, c] = sigma if sigma > 1e-8 else 1.0
    return mean, std
