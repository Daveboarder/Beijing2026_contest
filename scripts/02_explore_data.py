"""Exploratory analysis: class-average spectra, PCA, shot-to-shot variability."""

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from libs2026 import Config, Preprocessor, build_features, load_shots
from libs2026 import plotting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    bounds = tuple(cfg["data"]["channel_bounds"])

    features = build_features(cfg, Preprocessor.from_config(cfg), n_jobs=args.n_jobs)
    train = features.subset("train")
    print(features, "->", train.X.shape, "training rows")

    plotting.plot_class_mean_spectra(
        features.wavelength, train.X, train.y,
        cfg.figures_dir / "class_mean_spectra.png", bounds,
    )

    scaled = StandardScaler().fit_transform(train.X)
    pca = PCA(n_components=10, random_state=0).fit(scaled)
    scores = pca.transform(scaled)
    plotting.plot_pca_scores(scores, train.y, pca.explained_variance_ratio_,
                             cfg.figures_dir / "pca_scores.png")
    print("explained variance (first 5 PCs): "
          + ", ".join(f"{v*100:.1f}%" for v in pca.explained_variance_ratio_[:5]))

    # Raw shot totals for a few samples of different aging levels.
    examples = (pd.DataFrame({"sample_id": train.sample_ids, "label": train.y})
                .drop_duplicates("sample_id").groupby("label").head(1))
    totals = {
        f"{row.sample_id} (level {int(row.label)})":
            np.asarray(load_shots(cfg, row.sample_id)).sum(axis=1)
        for row in examples.itertuples()
    }
    plotting.plot_shot_variability(totals, cfg.figures_dir / "shot_variability.png")

    rsd = {name: float(np.std(v) / np.mean(v) * 100) for name, v in totals.items()}
    print("\ntotal-intensity RSD per sample (%):")
    for name, value in rsd.items():
        print(f"  {name:32s} {value:5.1f}")
    print(f"\nfigures written to {cfg.figures_dir}")


if __name__ == "__main__":
    main()
