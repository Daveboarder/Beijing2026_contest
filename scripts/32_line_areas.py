"""One signal area per spectral line, classified with an MLP with and without PCA.

    python scripts/32_line_areas.py --n-repeats 10 --n-jobs 10

The lines are the 274 detected by scripts/31 (src/libs2026/line_tokens.py).
Every line becomes a single number per depth block: its signal area between
b1 and b2, the nearest inflection points left and right of the line centre,
found once on the smoothed training mean spectrum and then fixed for every
spectrum (as the b1_w / b2_w windows of context/LIBSmethods.py).

Area variants, all on the Nyquist-smoothed 20-shot block rows:
* inflection, linear baseline -- trapezoid over b1..b2 minus the trapezoid
  under the chord joining the two bound values;
* inflection, LIBSmethods formula -- calculate_signal_area as written, whose
  baseline term is one pixel wide, so the continuum largely stays in;
* valleys, linear baseline -- the whole line between the valley anchors of
  the detection, for comparison with the inflection window.

Models: MLP(128, 64) on the scaled areas (no PCA), and scale -> PCA(k) -> scale
-> the same MLP for every k in --pca. Grouped CV on the folds of scripts/31,
with the full-spectrum pca_mlp as the reference; row probabilities are
averaged per sample.

Outputs
* results/metrics/line_areas_benchmark.csv
* results/metrics/line_areas_windows.csv   b1/b2 of every line (pixels and nm)
"""

import argparse
import time
from importlib import import_module

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from libs2026 import Config, Preprocessor, build_features, get_model
from libs2026.line_tokens import (
    LineTokenConfig,
    build_line_tokens,
    inflection_bounds,
    line_areas,
    smooth_nyquist,
)
from libs2026.models import RANDOM_STATE

s31 = import_module("31_line_tokens")


def area_model(n_pca: int = 0) -> Pipeline:
    """The pca_mlp head on line areas; ``n_pca=0`` feeds the scaled areas straight in."""
    steps = [("scale", StandardScaler())]
    if n_pca:
        steps += [("pca", PCA(n_pca, random_state=RANDOM_STATE, svd_solver="randomized")),
                  ("scale2", StandardScaler())]
    return Pipeline(steps + [("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000,
                                                   alpha=1e-3, random_state=RANDOM_STATE))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--pca", default="30,120", help="PCA widths to score besides no PCA")
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv = cfg["cv"]
    started = time.perf_counter()
    settings = LineTokenConfig.from_config(cfg)
    pre = Preprocessor.from_config(cfg)
    pre.normalization_reference = "shot"
    lines = build_line_tokens(cfg, settings, pre=pre, n_jobs=8).lines
    features = build_features(cfg, pre, n_groups=settings.n_groups, bin_factor=1,
                              encoding="mean", augment="blocks", n_jobs=8)
    bounds = tuple(cfg["data"]["channel_bounds"])
    wavelength = np.asarray(features.wavelength, dtype=float)
    train = features.subset("train")
    X, y, groups = smooth_nyquist(train.X, bounds), train.y.astype(int), train.groups
    n_rows = settings.n_groups

    windows = inflection_bounds(train.X.mean(axis=0), lines, bounds)
    width = windows["b2_px"] - windows["b1_px"]
    fwhm_px = lines["fwhm_nm"] / lines["step_nm"]
    print(f"{len(lines)} lines; inflection window {width.median():.0f} px median "
          f"(5-95 %: {width.quantile(0.05):.0f}-{width.quantile(0.95):.0f}), "
          f"{(width / fwhm_px).median():.2f} x FWHM")
    table = lines[["line_id", "channel", "wavelength_nm", "assignment", "peak_px",
                   "bg_left", "bg_right"]].join(windows)
    table["b1_nm"], table["b2_nm"] = wavelength[table["b1_px"]], wavelength[table["b2_px"]]
    table.to_csv(cfg.metrics_dir / "line_areas_windows.csv", index=False)

    areas = {
        "inflection, linear baseline":
            line_areas(X, wavelength, windows["b1_px"], windows["b2_px"], "linear"),
        "inflection, LIBSmethods formula":
            line_areas(X, wavelength, windows["b1_px"], windows["b2_px"], "libsmethods"),
        "valleys, linear baseline":
            line_areas(X, wavelength, lines["bg_left"], lines["bg_right"], "linear"),
    }
    widths = [0] + [int(k) for k in args.pca.split(",") if k.strip()]
    runs = [("full spectrum (12282 px)", "PCA(30) + MLP", train.X, lambda: get_model("pca_mlp"))]
    for name, A in areas.items():
        for k in widths:
            label = "MLP (no PCA)" if k == 0 else f"PCA({k}) + MLP"
            runs.append((name, label, A, lambda k=k: area_model(k)))
    seeds = [cv["random_state"] + r for r in range(args.n_repeats)]
    probas = Parallel(n_jobs=args.n_jobs)(
        delayed(s31._cv_proba)(make(), A, y, groups, cv["n_splits"], s)
        for _, _, A, make in runs for s in seeds)

    rows = []
    for i, (name, label, A, _) in enumerate(runs):
        chunk = probas[i * len(seeds):(i + 1) * len(seeds)]
        per = pd.DataFrame([s31.sample_scores(p, y, n_rows) for p in chunk])
        rows.append({"features": name, "model": label, "n_features": A.shape[1],
                     "n_repeats": len(per), **per.mean().to_dict(),
                     "accuracy_std": per["accuracy"].std(ddof=0)})
    bench = pd.DataFrame(rows)
    bench.to_csv(cfg.metrics_dir / "line_areas_benchmark.csv", index=False)
    print("\nsample level, identical grouped folds")
    print(bench.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"[{time.perf_counter() - started:.0f}s]")


if __name__ == "__main__":
    main()
