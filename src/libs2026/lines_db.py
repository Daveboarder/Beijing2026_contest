"""Theoretical spectral-line dictionary for the steel matrix.

Ported from the LIBS foundation-model project (``data/line_dictionary.py`` and
the physics helpers of ``data/libs_pipeline.py``) so this project stays
self-contained: only ``sqlite3``, ``numpy``, ``pandas`` and ``scipy.constants``
are needed, no HDF5 and no torch.

For every element the line intensities are evaluated with a Saha-Boltzmann
model over a Te x Ne grid, the grid maximum is kept per line, and the strongest
``percent`` of an element's lines survive. Only the wavelength span actually
recorded by the spectrometer is considered.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.constants as const

# Physical constants in CGS, matching the upstream spectrum generator.
_KB = const.k * 1e7              # erg/K
_H = const.h * 1e7               # erg*s
_C_SPEED = const.c               # m/s
_ME = const.electron_mass * 1e3  # g
_EV_TO_ERG = 1.60217e-12

# Steel matrix plus the ambient species that always show up in a LIBS plasma.
STEEL_ELEMENTS = (
    "Fe", "Cr", "Ni", "Mn", "Si", "Mo", "V", "Cu", "Ti", "Al", "C",
    "H", "O", "N",
)

# Z 1-103, used to turn an element symbol into the ``atomic_number`` channel.
_ATOMIC_NUMBERS: dict[str, int] = {
    sym: z + 1
    for z, sym in enumerate(
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
        "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
        "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf "
        "Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am "
        "Cm Bk Cf Es Fm Md No Lr".split()
    )
}


def atomic_number(symbol: str) -> int:
    sym = str(symbol).strip()
    if sym not in _ATOMIC_NUMBERS:
        raise KeyError(f"Unknown element symbol '{symbol}'")
    return _ATOMIC_NUMBERS[sym]


def ion_binary(ion_state: str) -> int:
    """0 for neutral ('I'), 1 for any ionised stage."""
    return 0 if str(ion_state).strip().upper() == "I" else 1


# ----------------------------------------------------------------------------
# SQLite access (cached per process; connections are not shared across workers)
# ----------------------------------------------------------------------------

_db_conn: sqlite3.Connection | None = None
_db_path_open: str | None = None
_quant_cache: dict[str, pd.DataFrame] = {}
_eion_cache: dict[str, float] = {}
_partf_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _connect(db_path: str) -> sqlite3.Connection:
    global _db_conn, _db_path_open
    if _db_conn is None or _db_path_open != db_path:
        if _db_conn is not None:
            _db_conn.close()
        _db_conn = sqlite3.connect(db_path)
        _db_path_open = db_path
        _quant_cache.clear()
        _eion_cache.clear()
        _partf_cache.clear()
    return _db_conn


def _get_quant_param(element: str, db_path: str) -> pd.DataFrame:
    if element not in _quant_cache:
        cur = _connect(db_path).cursor()
        cur.execute(
            "SELECT Elem_name, ion_state, Wavelength, Ei, Ek, gi, gk, Ak "
            "FROM QuantParam WHERE Elem_name = ?",
            (element,),
        )
        _quant_cache[element] = pd.DataFrame(
            cur.fetchall(),
            columns=["Elem_name", "ion_state", "Wavelength", "Ei", "Ek", "gi", "gk", "Ak"],
        )
    return _quant_cache[element]


def _get_eion(element: str, db_path: str) -> float:
    if element not in _eion_cache:
        cur = _connect(db_path).cursor()
        cur.execute("SELECT Eion FROM E_ion WHERE Elem_name = ?", (element + "+I",))
        row = cur.fetchall()
        if not row:
            raise ValueError(f"E_ion missing for '{element}+I'")
        _eion_cache[element] = row[0][0]
    return _eion_cache[element]


def _load_partf(element: str, db_path: str):
    if element not in _partf_cache:
        cur = _connect(db_path).cursor()
        cur.execute(
            "SELECT ion_state, Ei, gi FROM PartF_var WHERE Elem_name = ?",
            (element,),
        )
        gi_i, ei_i, gi_ii, ei_ii = [], [], [], []
        for ion_state, ei, gi in cur.fetchall():
            if ion_state == "I":
                gi_i.append(gi)
                ei_i.append(ei)
            elif ion_state == "II":
                gi_ii.append(gi)
                ei_ii.append(ei)
        _partf_cache[element] = (
            np.array(gi_i, dtype=np.float64),
            np.array(ei_i, dtype=np.float64),
            np.array(gi_ii, dtype=np.float64),
            np.array(ei_ii, dtype=np.float64),
        )
    return _partf_cache[element]


def partition_function(element: str, temperature: float, db_path: str) -> tuple[float, float]:
    """Boltzmann partition functions for the neutral (I) and singly ionised (II) species."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature} K")
    gi_i, ei_i, gi_ii, ei_ii = _load_partf(element, db_path)
    kb_ev = 8.617333262e-5  # eV/K
    u_i = float(np.sum(gi_i * np.exp(-ei_i / (kb_ev * temperature)))) if gi_i.size else 0.0
    u_ii = float(np.sum(gi_ii * np.exp(-ei_ii / (kb_ev * temperature)))) if gi_ii.size else 0.0
    return u_i, u_ii


def list_elements(db_path: str) -> list[str]:
    cur = _connect(db_path).cursor()
    cur.execute("SELECT DISTINCT Elem_name FROM QuantParam ORDER BY Elem_name")
    return [r[0] for r in cur.fetchall()]


# ----------------------------------------------------------------------------
# Saha-Boltzmann line intensities
# ----------------------------------------------------------------------------


def line_intensities(
    element: str,
    te: float,
    ne: float,
    number_density: float,
    concentration: float,
    optical_path: float,
    db_path: str,
):
    """Per-line theoretical intensities for one (Te, Ne) plasma point.

    Returns ``(wavelength, ion_state, Ei, Ek, gi, gk, Ak, intensity)``.
    """
    qp = _get_quant_param(element, db_path)
    if qp.empty:
        empty = np.array([], dtype=np.float64)
        return empty, np.array([], dtype=object), empty, empty, empty, empty, empty, empty

    e_ion = _get_eion(element, db_path)
    pf_i, pf_ii = partition_function(element, te, db_path)
    if pf_i <= 0.0:
        # Without a neutral partition function the Saha ratio is undefined.
        empty = np.array([], dtype=np.float64)
        return empty, np.array([], dtype=object), empty, empty, empty, empty, empty, empty

    s10 = (
        ((2 * pf_ii) / (ne * pf_i))
        * ((_ME * _KB * te) / ((_H ** 2) / (2 * np.pi))) ** 1.5
        * np.exp(-(e_ion * _EV_TO_ERG) / (_KB * te))
    )

    ion_is_neutral = (qp["ion_state"] == "I").values
    pf_per_line = np.where(ion_is_neutral, pf_i, pf_ii)
    # A missing PF_II would divide the ionic lines by zero; keep them finite.
    pf_per_line = np.where(pf_per_line > 0, pf_per_line, 1.0)
    ratio = np.where(ion_is_neutral, 1 / (1 + s10), s10 / (1 + s10))

    wl = qp["Wavelength"].values.astype(np.float64)
    ak = qp["Ak"].values.astype(np.float64)
    gk = qp["gk"].values.astype(np.float64)
    gi = qp["gi"].values.astype(np.float64)
    ei = qp["Ei"].values.astype(np.float64)
    ek = qp["Ek"].values.astype(np.float64)
    ion_state = qp["ion_state"].values

    kbt = _KB * te
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        kt = (
            (wl ** 4 / (8 * np.pi * _C_SPEED))
            * (ak * gk * np.exp(-ei * _EV_TO_ERG / kbt))
            * (1 - np.exp(-_EV_TO_ERG * (ek - ei) / kbt))
            / pf_per_line
        )
        lp = (
            (8 * np.pi * _H * _C_SPEED) / (10 * wl ** 3)
            * number_density * np.exp(-_EV_TO_ERG * (ek - ei) / kbt) * (gk / gi)
        )
        tau = concentration * number_density * ratio * optical_path * kt
        intensity = lp * (1 - np.exp(-tau))
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    return wl, ion_state, ei, ek, gi, gk, ak, intensity.astype(np.float64)


# ----------------------------------------------------------------------------
# Dictionary assembly
# ----------------------------------------------------------------------------

# Static token channels, in the order used by tokens.py (mirrors upstream).
STATIC_FEATURE_NAMES = (
    "central_wavelength",
    "Ei",
    "Ek",
    "log10_gi",
    "log10_gk",
    "log10_Ak",
    "log10_theoretical_intensity",
    "atomic_number",
    "ion_binary",
)
N_STATIC = len(STATIC_FEATURE_NAMES)


def _safe_log10(x: np.ndarray, floor: float = 1e-80) -> np.ndarray:
    """log10 with a floor that only catches non-positive values.

    LIBS intensities span ~1e-40 to 1e+8, so the real dynamic range is kept.
    """
    return np.log10(np.maximum(np.asarray(x, dtype=np.float64), floor)).astype(np.float32)


@dataclass
class LineDictionary:
    """Selected theoretical lines with their static token channels."""

    wavelength: np.ndarray      # (n_lines,) nm, ascending
    element: np.ndarray         # (n_lines,) str
    ion_state: np.ndarray       # (n_lines,) str
    static: np.ndarray          # (n_lines, N_STATIC) float32
    theoretical_intensity: np.ndarray
    db_path: str = ""
    config_hash: str = ""

    @property
    def n_lines(self) -> int:
        return int(self.wavelength.size)

    def counts_per_element(self) -> pd.Series:
        return pd.Series(self.element).value_counts().sort_values(ascending=False)

    def top_n(self, n: int) -> LineDictionary:
        """Keep the ``n`` strongest lines, still ordered by wavelength."""
        if n >= self.n_lines:
            return self
        return self._take(np.argsort(self.theoretical_intensity)[::-1][:n])

    def _take(self, keep: np.ndarray) -> LineDictionary:
        keep = np.sort(np.asarray(keep, dtype=int))
        return LineDictionary(
            wavelength=self.wavelength[keep],
            element=self.element[keep],
            ion_state=self.ion_state[keep],
            static=self.static[keep],
            theoretical_intensity=self.theoretical_intensity[keep],
            db_path=self.db_path,
            config_hash=self.config_hash,
        )

    def resolvable(
        self,
        wavelength_axis: np.ndarray,
        channel_bounds: tuple[int, ...],
        n_pixels: float = 2.0,
    ) -> LineDictionary:
        """Drop lines this spectrometer cannot separate from a stronger neighbour.

        The line list is far finer than the instrument: median spacing is about
        0.10 nm against a 0.08-0.18 nm resolution limit, so most theoretical
        lines arrive blended into a single observed peak. Fitting each one
        separately does not recover them, it just hands the same peak to several
        tokens under different labels, which makes the static physics channels
        describe the wrong transition.

        Within each blended group only the theoretically strongest line is kept,
        since that is the one dominating the peak the fit will actually find.
        """
        order = np.argsort(self.wavelength)
        wl = self.wavelength[order]
        intensity = self.theoretical_intensity[order]

        # Local dispersion per channel; a line counts as resolvable if any
        # channel covering it samples finely enough.
        steps = []
        for a, b in zip(channel_bounds[:-1], channel_bounds[1:]):
            seg = wavelength_axis[a:b]
            steps.append((seg.min(), seg.max(), float(np.median(np.diff(seg)))))

        def limit(value: float) -> float:
            covering = [s for lo, hi, s in steps if lo <= value <= hi]
            return n_pixels * (min(covering) if covering else max(s for _, _, s in steps))

        keep_sorted: list[int] = []
        group: list[int] = [0]
        for i in range(1, len(wl)):
            if wl[i] - wl[group[-1]] < limit(float(wl[i])):
                group.append(i)
            else:
                keep_sorted.append(max(group, key=lambda j: intensity[j]))
                group = [i]
        keep_sorted.append(max(group, key=lambda j: intensity[j]))
        return self._take(order[np.array(keep_sorted, dtype=int)])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            wavelength=self.wavelength,
            element=self.element.astype(str),
            ion_state=self.ion_state.astype(str),
            static=self.static,
            theoretical_intensity=self.theoretical_intensity,
            db_path=np.array(self.db_path),
            config_hash=np.array(self.config_hash),
        )

    @classmethod
    def load(cls, path: Path) -> LineDictionary:
        blob = np.load(path, allow_pickle=False)
        return cls(
            wavelength=blob["wavelength"],
            element=blob["element"].astype(str),
            ion_state=blob["ion_state"].astype(str),
            static=blob["static"],
            theoretical_intensity=blob["theoretical_intensity"],
            db_path=str(blob["db_path"]),
            config_hash=str(blob["config_hash"]),
        )

    def __repr__(self) -> str:
        span = f"{self.wavelength.min():.1f}-{self.wavelength.max():.1f} nm"
        return f"LineDictionary(n_lines={self.n_lines}, {span})"


def _select_for_element(records: list[dict], percent: float, min_keep: int) -> list[dict]:
    """Strongest ``percent`` of an element's lines, or all if it has very few."""
    n = len(records)
    if n == 0:
        return []
    k = n if n < min_keep else max(1, int(math.ceil((percent / 100.0) * n)))
    return sorted(records, key=lambda r: r["theoretical_intensity"], reverse=True)[:k]


def build_line_dictionary(
    db_path: str | Path,
    elements: tuple[str, ...] | list[str] = STEEL_ELEMENTS,
    wl_min: float | None = None,
    wl_max: float | None = None,
    te_range: tuple[float, float] = (6500.0, 11000.0),
    ne_range: tuple[float, float] = (1e17, 5e17),
    n_te: int = 10,
    n_ne: int = 10,
    number_density: float = 1e-4,
    concentration: float = 1.0,
    optical_path: float = 1.4e-4,
    percent: float = 10.0,
    min_keep: int = 10,
    cache_dir: Path | None = None,
    verbose: bool = True,
) -> LineDictionary:
    """Build (or load) the theoretical line dictionary for ``elements``."""
    db_path = str(Path(db_path).resolve())
    key = {
        "db": Path(db_path).name,
        "elements": sorted(elements),
        "wl": [wl_min, wl_max],
        "te": list(te_range),
        "ne": list(ne_range),
        "grid": [n_te, n_ne],
        "N": number_density,
        "C": concentration,
        "l": optical_path,
        "percent": percent,
        "min_keep": min_keep,
        "version": 1,
    }
    digest = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]

    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / "lines" / f"line_dict_{digest}.npz"
        if cache_file.exists():
            if verbose:
                print(f"line dictionary cache hit: {cache_file}")
            return LineDictionary.load(cache_file)

    te_grid = np.linspace(te_range[0], te_range[1], n_te)
    ne_grid = np.logspace(np.log10(ne_range[0]), np.log10(ne_range[1]), n_ne)

    available = set(list_elements(db_path))
    wanted = [e for e in elements if e in available]
    missing = [e for e in elements if e not in available]
    if verbose:
        print(f"line dictionary: {len(wanted)} elements, Te x Ne = {n_te} x {n_ne}")
        if missing:
            print(f"  not in database, skipped: {', '.join(missing)}")

    records: list[dict] = []
    for elem in wanted:
        try:
            _get_eion(elem, db_path)
        except ValueError:
            if verbose:
                print(f"  {elem}: no ionisation energy in database, skipped")
            continue

        # Keep the Te x Ne maximum per (wavelength, ion state).
        best: dict[tuple[float, str], dict] = {}
        for te in te_grid:
            for ne in ne_grid:
                wl, ion, ei, ek, gi, gk, ak, inten = line_intensities(
                    elem, float(te), float(ne), number_density,
                    concentration, optical_path, db_path,
                )
                if wl.size == 0:
                    continue
                keep = np.ones(wl.size, dtype=bool)
                if wl_min is not None:
                    keep &= wl >= wl_min
                if wl_max is not None:
                    keep &= wl <= wl_max
                for i in np.flatnonzero(keep):
                    line_key = (float(wl[i]), str(ion[i]))
                    prev = best.get(line_key)
                    if prev is None or inten[i] > prev["theoretical_intensity"]:
                        best[line_key] = {
                            "central_wavelength": float(wl[i]),
                            "element": elem,
                            "ion_state": str(ion[i]),
                            "Ei": float(ei[i]),
                            "Ek": float(ek[i]),
                            "gi": float(gi[i]),
                            "gk": float(gk[i]),
                            "Ak": float(ak[i]),
                            "theoretical_intensity": float(inten[i]),
                        }

        selected = _select_for_element(list(best.values()), percent, min_keep)
        records.extend(selected)
        if verbose:
            print(f"  {elem:>2s}: kept {len(selected):4d} / {len(best):5d}")

    if not records:
        raise RuntimeError(
            f"No lines selected from {db_path}. Check the element list and wavelength clip."
        )

    frame = pd.DataFrame(records).sort_values("central_wavelength").reset_index(drop=True)

    static = np.zeros((len(frame), N_STATIC), dtype=np.float32)
    static[:, 0] = frame["central_wavelength"].to_numpy(np.float32)
    static[:, 1] = frame["Ei"].to_numpy(np.float32)
    static[:, 2] = frame["Ek"].to_numpy(np.float32)
    static[:, 3] = _safe_log10(frame["gi"].to_numpy())
    static[:, 4] = _safe_log10(frame["gk"].to_numpy())
    static[:, 5] = _safe_log10(frame["Ak"].to_numpy())
    static[:, 6] = _safe_log10(frame["theoretical_intensity"].to_numpy())
    static[:, 7] = np.array([atomic_number(e) for e in frame["element"]], dtype=np.float32)
    static[:, 8] = np.array([ion_binary(s) for s in frame["ion_state"]], dtype=np.float32)

    dictionary = LineDictionary(
        wavelength=frame["central_wavelength"].to_numpy(np.float32),
        element=frame["element"].to_numpy(dtype=str),
        ion_state=frame["ion_state"].to_numpy(dtype=str),
        static=static,
        theoretical_intensity=frame["theoretical_intensity"].to_numpy(np.float64),
        db_path=db_path,
        config_hash=digest,
    )
    if verbose:
        print(f"line dictionary: {dictionary.n_lines} lines total")
    if cache_file is not None:
        dictionary.save(cache_file)
    return dictionary
