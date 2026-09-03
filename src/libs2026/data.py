"""Reading the contest CSVs and caching them as binary arrays.

The released data are 180 CSV files of ~25 MB each (12282 wavelengths x 200
shots). Parsing them repeatedly is the main bottleneck of any experiment, so
they are converted once into float32 ``.npy`` files that can be memory-mapped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import Config

WAVELENGTH_FILE = "wavelength.npy"
SHOTS_SUBDIR = "shots"
INDEX_FILE = "index.csv"


def sample_files(cfg: Config, split: str) -> list[Path]:
    """Sorted CSV paths for ``split`` in {'train', 'test'}."""
    directory = cfg.train_dir if split == "train" else cfg.test_dir
    return sorted(directory.glob(f"{split}_*.csv"))


def load_labels(cfg: Config) -> pd.DataFrame:
    """Training labels as a frame with columns ``sample_id`` and ``label``."""
    # The file carries a UTF-8 BOM, which would otherwise corrupt the first name.
    labels = pd.read_csv(cfg.label_file, encoding="utf-8-sig")
    labels["sample_id"] = labels["filename"].str.replace(".csv", "", regex=False)
    return labels[["sample_id", "filename", "label"]]


def read_sample_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(wavelength, shots)`` for one sample CSV; shots is (n_shots, n_wl)."""
    frame = pd.read_csv(path, dtype=np.float32)
    wavelength = frame.iloc[:, 0].to_numpy(np.float32)
    shots = np.ascontiguousarray(frame.iloc[:, 1:].to_numpy(np.float32).T)
    return wavelength, shots


def _cache_one(path: Path, out_dir: Path, overwrite: bool) -> tuple[str, np.ndarray | None]:
    sample_id = path.stem
    out_file = out_dir / f"{sample_id}.npy"
    if out_file.exists() and not overwrite:
        return sample_id, None
    wavelength, shots = read_sample_csv(path)
    np.save(out_file, shots)
    return sample_id, wavelength


def build_cache(cfg: Config, overwrite: bool = False, n_jobs: int = 8) -> pd.DataFrame:
    """Convert every raw CSV into a cached ``.npy`` array and write an index."""
    shots_dir = cfg.cache_dir / SHOTS_SUBDIR
    shots_dir.mkdir(parents=True, exist_ok=True)

    paths = sample_files(cfg, "train") + sample_files(cfg, "test")
    if not paths:
        raise FileNotFoundError(f"No sample CSVs found under {cfg.raw_data_dir}")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_cache_one)(p, shots_dir, overwrite) for p in paths
    )

    wavelengths = [w for _, w in results if w is not None]
    wl_file = cfg.cache_dir / WAVELENGTH_FILE
    if wavelengths:
        reference = wavelengths[0]
        if not all(np.allclose(reference, w) for w in wavelengths[1:]):
            raise ValueError("Wavelength axis differs between samples.")
        np.save(wl_file, reference)
    elif not wl_file.exists():
        raise FileNotFoundError(
            "Every sample was already cached but the wavelength axis is missing; "
            "rerun with overwrite=True."
        )

    labels = load_labels(cfg).set_index("sample_id")["label"]
    index = pd.DataFrame({"sample_id": [p.stem for p in paths]})
    index["split"] = index["sample_id"].str.split("_").str[0]
    index["label"] = index["sample_id"].map(labels).astype("Int64")
    index.to_csv(cfg.cache_dir / INDEX_FILE, index=False)
    return index


def load_index(cfg: Config) -> pd.DataFrame:
    path = cfg.cache_dir / INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Cache index missing ({path}). Run scripts/01_prepare_data.py first."
        )
    return pd.read_csv(path)


def load_wavelength(cfg: Config) -> np.ndarray:
    return np.load(cfg.cache_dir / WAVELENGTH_FILE)


def load_shots(cfg: Config, sample_id: str, mmap: bool = True) -> np.ndarray:
    """Cached shot matrix of one sample, shaped (n_shots, n_wavelengths)."""
    path = cfg.cache_dir / SHOTS_SUBDIR / f"{sample_id}.npy"
    return np.load(path, mmap_mode="r" if mmap else None)


def iter_samples(cfg: Config, split: str | None = None):
    """Yield ``(sample_id, label, shots)`` for the requested split."""
    index = load_index(cfg)
    if split is not None:
        index = index[index["split"] == split]
    for row in index.itertuples():
        yield row.sample_id, row.label, load_shots(cfg, row.sample_id)
