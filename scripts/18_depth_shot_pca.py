"""PCA across shots of one random sample — loadings and depth-coloured scores.

Shows where spectral variance lies during depth profiling (surface → bulk).

uv run python scripts/18_depth_shot_pca.py
uv run python scripts/18_depth_shot_pca.py --sample-id SAMPLE --seed 0
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from libs2026 import Config, Preprocessor, load_index, load_shots, load_wavelength
from libs2026 import plotting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample-id", default=None,
                        help="train sample id; default = random under --seed")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for random sample pick (default: nondeterministic)")
    parser.add_argument("--n-components", type=int, default=5)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    bounds = tuple(cfg["data"]["channel_bounds"])
    index = load_index(cfg)
    train = index[index["split"] == "train"].reset_index(drop=True)
    if train.empty:
        raise SystemExit("No training samples in the index")

    rng = np.random.default_rng(args.seed)
    if args.sample_id is None:
        row = train.iloc[int(rng.integers(0, len(train)))]
    else:
        match = train[train["sample_id"] == args.sample_id]
        if match.empty:
            raise SystemExit(f"Unknown train sample_id: {args.sample_id}")
        row = match.iloc[0]
    sample_id = str(row["sample_id"])
    label = row["label"]

    pre = Preprocessor.from_config(cfg)
    shots = pre(load_shots(cfg, sample_id, mmap=False))
    wavelength = load_wavelength(cfg)
    if shots.ndim != 2 or shots.shape[1] != len(wavelength):
        raise SystemExit(f"Unexpected shot matrix shape {shots.shape}")

    scaled = StandardScaler().fit_transform(shots)
    n_comp = min(args.n_components, shots.shape[0] - 1, shots.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(scaled)
    scores = pca.transform(scaled)
    # components_ is (n_comp, n_features); plot as loadings vs wavelength.
    loadings = pca.components_.T

    out = cfg.figures_dir / f"depth_shot_pca_{sample_id}.png"
    plotting.plot_depth_shot_pca(
        wavelength, loadings, scores, pca.explained_variance_ratio_,
        out, sample_id=sample_id, label=label, channel_bounds=bounds,
    )

    # Peak loading wavelengths for a short console summary.
    peaks = []
    for k in range(min(3, n_comp)):
        i = int(np.argmax(np.abs(loadings[:, k])))
        peaks.append(f"PC{k + 1}@{wavelength[i]:.1f}nm")

    print(f"sample_id   : {sample_id}")
    print(f"aging level : {int(label)}")
    print(f"shots x wl  : {shots.shape}")
    print("explained   : "
          + ", ".join(f"PC{i + 1}={v * 100:.1f}%"
                      for i, v in enumerate(pca.explained_variance_ratio_[:5])))
    print(f"peak |load| : {', '.join(peaks)}")
    print(f"figure      -> {out}")


if __name__ == "__main__":
    main()
