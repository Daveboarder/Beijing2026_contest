"""Build each sample as a 2-D ``(shots x wavelengths)`` array.

The 200 shots are a depth profile through one spot, so stacking them as rows
of an image keeps both axes meaningful: height = depth, width = spectrum.
A 2-D CNN can then learn local patterns that couple wavelength and depth
(e.g. the aged-layer dip of a Cr line) the way a vision network learns texture.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from .config import Config
from .data import load_index, load_shots, load_wavelength
from .features import bin_spectrum, _bin_axis
from .preprocessing import Preprocessor


class ImageSet:
    """Stack of depth-spectrum images with sample bookkeeping.

    ``X`` has shape ``(n_samples, n_shots, n_wavelengths)``.
    """

    def __init__(self, X, y, sample_ids, split, wavelength):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = y
        self.sample_ids = np.asarray(sample_ids)
        self.split = np.asarray(split)
        self.wavelength = wavelength
        # One image per sample, so the group id equals the row index.
        self.groups = np.arange(len(self.sample_ids))

    def subset(self, split: str) -> "ImageSet":
        mask = self.split == split
        y = self.y[mask] if self.y is not None else None
        return ImageSet(
            self.X[mask], y, self.sample_ids[mask], self.split[mask], self.wavelength,
        )

    def as_flat(self) -> np.ndarray:
        """Flatten to ``(n_samples, n_shots * n_wavelengths)`` for sklearn APIs."""
        return self.X.reshape(self.X.shape[0], -1)

    @property
    def image_shape(self) -> tuple[int, int]:
        return int(self.X.shape[1]), int(self.X.shape[2])

    def __repr__(self) -> str:
        return f"ImageSet(X={self.X.shape}, n_samples={len(self.sample_ids)})"


def _one_image(cfg: Config, sample_id: str, pre: Preprocessor,
               bin_factor: int, bounds) -> np.ndarray:
    shots = pre(load_shots(cfg, sample_id, mmap=False))
    if bin_factor > 1:
        shots = bin_spectrum(shots, bin_factor, bounds)
    return np.asarray(shots, dtype=np.float32)


def build_images(
    cfg: Config,
    pre: Preprocessor | None = None,
    bin_factor: int = 8,
    n_jobs: int = 8,
    use_cache: bool = True,
) -> ImageSet:
    """Preprocess every sample into a ``(n_shots, n_wavelengths)`` image.

    Wavelengths are binned by ``bin_factor`` (default 8 → ~1535 columns): a raw
    200 x 12282 image is too wide for a small dataset, and neighbouring
    spectrometer pixels are highly correlated anyway.
    """
    pre = pre or Preprocessor.from_config(cfg)
    bounds = tuple(cfg["data"].get("channel_bounds", (0, 4094, 8188, 12282)))

    key = json.dumps(
        {"pre": asdict(pre), "bin": bin_factor, "kind": "image"},
        sort_keys=True, default=str,
    )
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_file: Path = cfg.cache_dir / "images" / f"{digest}.npz"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    wavelength = load_wavelength(cfg)
    if use_cache and cache_file.exists():
        blob = np.load(cache_file, allow_pickle=True)
        y = blob["y"]
        return ImageSet(
            blob["X"], np.where(y < 0, np.nan, y), blob["sample_ids"],
            blob["split"], blob["wavelength"],
        )

    index = load_index(cfg)
    arrays = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_image)(cfg, sid, pre, bin_factor, bounds)
        for sid in index["sample_id"]
    )
    X = np.stack(arrays, axis=0)
    y = index["label"].to_numpy(dtype=float)
    axis = _bin_axis(wavelength, bin_factor, bounds)

    np.savez_compressed(
        cache_file, X=X, y=np.where(np.isnan(y), -1, y),
        sample_ids=index["sample_id"].to_numpy(),
        split=index["split"].to_numpy(), wavelength=axis,
    )
    with open(cache_file.with_suffix(".json"), "w", encoding="utf-8") as fh:
        fh.write(key)
    return ImageSet(X, y, index["sample_id"].to_numpy(),
                    index["split"].to_numpy(), axis)
