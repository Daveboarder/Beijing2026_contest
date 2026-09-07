"""
Theoretical spectral-line dictionary over a Te × Ne grid.

Computes per-line emission intensities (no Voigt broadening), keeps the maximum
over the grid per line, filters by absolute threshold, and caches to HDF5.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from data.libs_pipeline import (
    _EV_TO_ERG,
    _H,
    _KB,
    _ME,
    _C_SPEED,
    _get_eion,
    _get_quant_param,
    load_wavelength,
    partition_function_cached,
)

# Ion state vocabulary for categorical embedding
ION_STATES = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
ION_STATE_TO_ID = {s: i for i, s in enumerate(ION_STATES)}


def _config_hash(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _te_ne_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    te_lo, te_hi = float(cfg["te_range"][0]), float(cfg["te_range"][1])
    ne_lo, ne_hi = float(cfg["ne_range"][0]), float(cfg["ne_range"][1])
    n_te = int(cfg["n_grid"]["te"])
    n_ne = int(cfg["n_grid"]["ne"])
    te_grid = np.linspace(te_lo, te_hi, n_te)
    if cfg.get("ne_log_spaced", True):
        ne_grid = np.logspace(np.log10(ne_lo), np.log10(ne_hi), n_ne)
    else:
        ne_grid = np.linspace(ne_lo, ne_hi, n_ne)
    return te_grid, ne_grid


def compute_line_intensities_for_plasma(
    element: str,
    Te: float,
    Ne: float,
    N: float,
    C: float,
    l: float,
    db_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-line theoretical intensities (Ifin) for one (Te, Ne) point.

    Returns:
        wl, ion_state_str, Ei, Ek, gi, gk, Ak, intensity
    """
    QP = _get_quant_param(element, db_path)
    if QP.empty:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty, empty, empty, empty

    E_ion = _get_eion(element, db_path)
    PF_I, PF_II = partition_function_cached(element, Te, db_path)

    S10 = (
        ((2 * PF_II) / (Ne * PF_I))
        * ((_ME * _KB * Te) / ((_H ** 2) / (2 * np.pi))) ** 1.5
        * np.exp(-(E_ion * _EV_TO_ERG) / (_KB * Te))
    )

    ion_is_I = (QP["ion_state"] == "I").values
    pf_per_line = np.where(ion_is_I, PF_I, PF_II)
    ri = np.where(ion_is_I, 1 / (1 + S10), S10 / (1 + S10))

    wl = QP["Wavelength"].values.astype(np.float64)
    Ak = QP["Ak"].values.astype(np.float64)
    gk = QP["gk"].values.astype(np.float64)
    gi = QP["gi"].values.astype(np.float64)
    Ei = QP["Ei"].values.astype(np.float64)
    Ek = QP["Ek"].values.astype(np.float64)
    ion_state = QP["ion_state"].values

    kbT = _KB * Te
    kt = (
        (wl ** 4 / (8 * np.pi * _C_SPEED))
        * (Ak * gk * np.exp(-Ei * _EV_TO_ERG / kbT))
        * (1 - np.exp(-_EV_TO_ERG * (Ek - Ei) / kbT))
        / pf_per_line
    )
    Lp = (
        (8 * np.pi * _H * _C_SPEED) / (10 * wl ** 3)
        * N * np.exp(-_EV_TO_ERG * (Ek - Ei) / kbT) * (gk / gi)
    )
    tau = C * N * ri * l * kt
    Ifin = Lp * (1 - np.exp(-tau))
    return wl, ion_state, Ei, Ek, gi, gk, Ak, Ifin.astype(np.float64)


def _list_elements(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT Elem_name FROM QuantParam ORDER BY Elem_name")
    elems = [r[0] for r in cur.fetchall()]
    conn.close()
    return elems


def _wavelength_clip_bounds(cfg: dict, project_root: Path) -> tuple[float | None, float | None]:
    clip = cfg.get("wavelength_clip") or {}
    wmin, wmax = clip.get("min"), clip.get("max")
    if wmin is not None and wmax is not None:
        return float(wmin), float(wmax)
    if cfg.get("use_wavelength_json_clip") and cfg.get("wavelength_json"):
        wl_path = project_root / cfg["wavelength_json"]
        if wl_path.is_file():
            wl = load_wavelength(str(wl_path))
            return float(wl.min()), float(wl.max())
    return wmin, wmax


def _select_lines_for_element(
    element_records: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    """Select lines for one element based on config mode."""
    mode = str(selection_cfg.get("mode", "top_percent_per_element")).strip().lower()
    if mode == "threshold":
        return [r for r in element_records if r["theoretical_intensity"] >= threshold]
    if mode != "top_percent_per_element":
        raise ValueError(
            f"Unknown line_dictionary.selection.mode='{mode}'. "
            "Expected 'top_percent_per_element' or 'threshold'."
        )

    percent = float(selection_cfg.get("percent", 10.0))
    min_keep = int(selection_cfg.get("min_keep", 10))
    if percent <= 0.0:
        raise ValueError("line_dictionary.selection.percent must be > 0")
    if min_keep < 1:
        raise ValueError("line_dictionary.selection.min_keep must be >= 1")

    n = len(element_records)
    if n < min_keep:
        k = n
    else:
        k = max(1, int(math.ceil((percent / 100.0) * n)))
    return sorted(element_records, key=lambda r: r["theoretical_intensity"], reverse=True)[:k]


def build_line_dictionary(cfg: dict, project_root: Path | None = None, verbose: bool = True) -> str:
    """
    Build or load cached line dictionary HDF5.

    Returns:
        Path to the cache file.
    """
    project_root = project_root or Path(__file__).resolve().parents[1]
    ld_cfg = dict(cfg["line_dictionary"])
    db_path = str((project_root / ld_cfg["db_path"]).resolve())
    cache_dir = project_root / ld_cfg.get("cache_dir", "external_data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    hash_cfg = {k: v for k, v in ld_cfg.items() if k != "cache_dir"}
    key = _config_hash(hash_cfg)
    out_path = cache_dir / f"line_dict_{key}.h5"

    if out_path.is_file():
        if verbose:
            print(f"Line dictionary cache hit: {out_path}")
        return str(out_path)

    te_grid, ne_grid = _te_ne_grid(ld_cfg)
    N = float(ld_cfg["N"])
    C = float(ld_cfg["C"])
    l_path = float(ld_cfg["l"])
    threshold = float(ld_cfg["intensity_threshold"])
    selection_cfg = dict(ld_cfg.get("selection", {}))
    if "mode" not in selection_cfg:
        selection_cfg["mode"] = "top_percent_per_element"
    if "percent" not in selection_cfg:
        selection_cfg["percent"] = 10.0
    if "min_keep" not in selection_cfg:
        selection_cfg["min_keep"] = 10
    wmin, wmax = _wavelength_clip_bounds(ld_cfg, project_root)

    elements = _list_elements(db_path)
    if verbose:
        print(f"Building line dictionary: {len(elements)} elements, "
              f"Te×Ne = {len(te_grid)}×{len(ne_grid)}, threshold={threshold}")

    records: list[dict[str, Any]] = []
    kept_by_element: dict[str, int] = {}
    total_by_element: dict[str, int] = {}

    for elem in elements:
        # Accumulate max intensity per (wl, ion_state) — lines are unique in DB per row
        best: dict[tuple[float, str], dict[str, Any]] = {}

        for Te in te_grid:
            for Ne in ne_grid:
                wl, ion, Ei, Ek, gi, gk, Ak, I = compute_line_intensities_for_plasma(
                    elem, float(Te), float(Ne), N, C, l_path, db_path,
                )
                for i in range(wl.size):
                    if wmin is not None and wl[i] < wmin:
                        continue
                    if wmax is not None and wl[i] > wmax:
                        continue
                    key_line = (float(wl[i]), str(ion[i]))
                    prev = best.get(key_line)
                    if prev is None or I[i] > prev["theoretical_intensity"]:
                        best[key_line] = {
                            "central_wavelength": float(wl[i]),
                            "element": elem,
                            "ion_state": str(ion[i]),
                            "Ei": float(Ei[i]),
                            "Ek": float(Ek[i]),
                            "gi": float(gi[i]),
                            "gk": float(gk[i]),
                            "Ak": float(Ak[i]),
                            "theoretical_intensity": float(I[i]),
                            "Te_opt": float(Te),
                            "Ne_opt": float(Ne),
                        }

        elem_records = list(best.values())
        total_by_element[elem] = len(elem_records)
        selected = _select_lines_for_element(
            element_records=elem_records,
            selection_cfg=selection_cfg,
            threshold=threshold,
        )
        kept_by_element[elem] = len(selected)
        records.extend(selected)

    if not records:
        mode = selection_cfg.get("mode", "top_percent_per_element")
        raise RuntimeError(
            f"No lines selected for mode '{mode}'. "
            "Adjust line_dictionary.selection or intensity_threshold."
        )

    df = pd.DataFrame(records).sort_values("central_wavelength").reset_index(drop=True)
    max_lines = ld_cfg.get("max_lines")
    if max_lines is not None and len(df) > int(max_lines):
        df = (
            df.nlargest(int(max_lines), "theoretical_intensity")
            .sort_values("central_wavelength")
            .reset_index(drop=True)
        )
        if verbose:
            print(f"  Subsampled to max_lines={max_lines}")
    elem_to_id = {e: i for i, e in enumerate(sorted(df["element"].unique()))}
    df["element_id"] = df["element"].map(elem_to_id)
    df["ion_state_id"] = df["ion_state"].map(lambda s: ION_STATE_TO_ID.get(s, 0))

    if verbose:
        mode = str(selection_cfg.get("mode", "top_percent_per_element"))
        if mode == "threshold":
            print(f"  Kept {len(df)} lines (threshold={threshold})")
        else:
            pct = float(selection_cfg.get("percent", 10.0))
            min_keep = int(selection_cfg.get("min_keep", 10))
            print(
                f"  Kept {len(df)} lines (mode={mode}, percent={pct}, min_keep={min_keep})"
            )
            summary = ", ".join(
                f"{e}:{kept_by_element.get(e, 0)}/{total_by_element.get(e, 0)}"
                for e in sorted(total_by_element)
            )
            print(f"  Kept per element (kept/total): {summary}")

    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as f:
        f.attrs["config_hash"] = key
        f.attrs["n_lines"] = len(df)
        f.attrs["config_json"] = json.dumps(hash_cfg, sort_keys=True)
        f.create_dataset("central_wavelength", data=df["central_wavelength"].values)
        f.create_dataset("theoretical_intensity", data=df["theoretical_intensity"].values)
        f.create_dataset("Te_opt", data=df["Te_opt"].values)
        f.create_dataset("Ne_opt", data=df["Ne_opt"].values)
        f.create_dataset("Ei", data=df["Ei"].values)
        f.create_dataset("Ek", data=df["Ek"].values)
        f.create_dataset("gi", data=df["gi"].values)
        f.create_dataset("gk", data=df["gk"].values)
        f.create_dataset("Ak", data=df["Ak"].values)
        f.create_dataset("element_id", data=df["element_id"].values.astype(np.int32))
        f.create_dataset("ion_state_id", data=df["ion_state_id"].values.astype(np.int32))
        g = f.create_group("vocab")
        g.attrs["elements"] = json.dumps(elem_to_id)
        g.attrs["ion_states"] = json.dumps(ION_STATE_TO_ID)
        g.create_dataset("element", data=df["element"].astype(str).values, dtype=str_dt)
        g.create_dataset("ion_state", data=df["ion_state"].astype(str).values, dtype=str_dt)

    if verbose:
        print(f"Saved line dictionary: {out_path}")
    return str(out_path)


def load_line_dictionary_meta(path: str) -> dict[str, Any]:
    """Load dictionary arrays and vocab without keeping the file open."""
    with h5py.File(path, "r") as f:
        vocab = json.loads(f["vocab"].attrs["elements"])
        n_lines = int(f.attrs["n_lines"])
        meta = {
            "path": path,
            "n_lines": n_lines,
            "config_hash": f.attrs.get("config_hash", ""),
            "central_wavelength": f["central_wavelength"][:],
            "theoretical_intensity": f["theoretical_intensity"][:],
            "Ei": f["Ei"][:],
            "Ek": f["Ek"][:],
            "gi": f["gi"][:],
            "gk": f["gk"][:],
            "Ak": f["Ak"][:],
            "element_id": f["element_id"][:],
            "ion_state_id": f["ion_state_id"][:],
            "element_vocab": vocab,
            "n_elements": len(vocab),
        }
    return meta


if __name__ == "__main__":
    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(open(root / "config" / "line_embedding.yaml"))
    build_line_dictionary(cfg, project_root=root, verbose=True)
