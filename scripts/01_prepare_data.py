"""Convert the raw contest CSVs into cached float32 arrays.

Run once after checking out the project:

    python scripts/01_prepare_data.py --n-jobs 12
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from libs2026 import Config, build_cache, load_index, load_wavelength


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    print(f"raw data : {cfg.raw_data_dir}")
    print(f"cache    : {cfg.cache_dir}")

    index = build_cache(cfg, overwrite=args.overwrite, n_jobs=args.n_jobs)
    wavelength = load_wavelength(cfg)

    print(f"\ncached {len(index)} samples")
    print(index.groupby("split").size().to_string())
    print("\nlabel distribution (train):")
    print(load_index(cfg).dropna(subset=["label"])["label"].value_counts().sort_index().to_string())

    diffs = np.diff(wavelength)
    breaks = np.where(diffs <= 0)[0]
    print(f"\nwavelength axis: {wavelength.size} points, "
          f"{wavelength.min():.3f}-{wavelength.max():.3f} nm")
    print(f"channel starts at row indices: {[0] + (breaks + 1).tolist()}")


if __name__ == "__main__":
    main()
