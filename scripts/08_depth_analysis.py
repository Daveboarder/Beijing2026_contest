"""Quantify how the spectrum evolves with depth, and how that depends on aging.

Each of the 200 shots hits the same spot, so shot number is a depth coordinate.
This script measures, per sample, the depth profile of the total emission and of
a few diagnostic lines, normalised to the bulk (deep) shots, and reports how
strongly each profile descriptor tracks the aging level.
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from libs2026 import Config, Preprocessor, load_index, load_shots, load_wavelength, log_bin_edges
from libs2026 import plotting

# Lines that are informative for aged steel: carbon, the main Fe/Cr matrix
# lines, and the air/oxide indicators.
DIAGNOSTIC_LINES = {
    "C I 247.9": 247.86,
    "Fe I 404.6": 404.58,
    "Cr I 425.4": 425.43,
    "Mn I 403.1": 403.08,
    "H I 656.3": 656.28,
    "O I 777.4": 777.42,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-bins", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    pre = Preprocessor.from_config(cfg)
    wavelength = load_wavelength(cfg)
    index = load_index(cfg)
    train = index[index["split"] == "train"]

    line_idx = {name: int(np.argmin(np.abs(wavelength - wl)))
                for name, wl in DIAGNOSTIC_LINES.items()}

    total_profiles, line_profiles, labels, rows = [], {n: [] for n in line_idx}, [], []
    for record in train.itertuples():
        shots = np.asarray(load_shots(cfg, record.sample_id, mmap=False))
        bulk = pre.bulk_slice(shots.shape[0])

        total = shots.sum(axis=1)
        total_profiles.append(total / total[bulk].mean())
        labels.append(int(record.label))

        entry = {"sample_id": record.sample_id, "label": int(record.label)}
        for name, i in line_idx.items():
            profile = shots[:, i] / max(shots[bulk, i].mean(), 1e-6)
            line_profiles[name].append(profile)
            entry[f"{name} surface"] = profile[0]
            entry[f"{name} dip"] = profile[:20].min()
        rows.append(entry)

    labels = np.array(labels)
    profiles = {"total emission": np.vstack(total_profiles)}
    profiles.update({name: np.vstack(v) for name, v in list(line_profiles.items())[:2]})
    plotting.plot_depth_profiles(profiles, labels, cfg.figures_dir / "depth_profiles.png",
                                 bin_edges=log_bin_edges(len(total_profiles[0]), args.n_bins))

    frame = pd.DataFrame(rows)
    numeric = frame.drop(columns=["sample_id"])
    print("class means of the depth descriptors (intensity relative to bulk)")
    print(numeric.groupby("label").mean().round(3).to_string())

    print("\ncorrelation of each descriptor with aging level")
    correlations = (numeric.drop(columns=["label"])
                    .apply(lambda col: np.corrcoef(col, labels)[0, 1])
                    .sort_values(key=np.abs, ascending=False))
    print(correlations.round(3).to_string())

    out = cfg.metrics_dir / "depth_descriptors.csv"
    frame.to_csv(out, index=False)
    print(f"\ndescriptors -> {out}")
    print(f"figure      -> {cfg.figures_dir / 'depth_profiles.png'}")


if __name__ == "__main__":
    main()
