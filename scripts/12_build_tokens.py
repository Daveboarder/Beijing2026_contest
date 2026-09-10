"""Tokenize every sample into spectral-line tokens and report fit diagnostics.

    uv run python scripts/12_build_tokens.py --n-jobs 23

Each spectrum becomes ``n_lines`` tokens instead of 12282 wavelength bins: a
theoretical line dictionary (Saha-Boltzmann over a Te x Ne grid) fixes the
columns, and a Voigt fit at each line centre fills in the per-spectrum
channels. Run this once; ``13_benchmark_token_cnn.py`` reuses the cache.

``--compare-db`` re-fits a subset against a second database, which is how the
air-vs-vacuum question was settled (air wins: the vacuum line list is offset by
~0.07-0.12 nm, more than one detector pixel).
"""

import argparse
from dataclasses import replace

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from libs2026 import Config, Preprocessor, plotting
from libs2026.data import load_shots, load_wavelength
from libs2026.lines_db import build_line_dictionary
from libs2026.tokens import (
    F_DELTA,
    F_MAX_INT,
    F_R2,
    F_VALID,
    fit_config_from_config,
    fit_spectra,
    build_tokens,
    line_dictionary_from_config,
    window_bounds,
)


def _sample_spectra(cfg, pre, shot_bin, sample_ids, rows_per_sample, n_shots=None):
    """A few binned rows from each of a handful of samples."""
    out = []
    for sid in sample_ids:
        shots = load_shots(cfg, sid, mmap=False)
        if n_shots is not None:
            shots = shots[: int(n_shots)]
        shots = pre(shots)
        n = (shots.shape[0] // shot_bin) * shot_bin
        binned = shots[:n].reshape(n // shot_bin, shot_bin, shots.shape[1]).mean(axis=1)
        out.append(binned[:rows_per_sample])
    return np.concatenate(out, axis=0).astype(np.float32)


def _fit_against(cfg, dictionary, spectra, fit_cfg):
    wavelength = load_wavelength(cfg)
    bounds_cfg = tuple(cfg["data"].get("channel_bounds", (0, 4094, 8188, 12282)))
    resolved, keep = [], []
    for j, centre in enumerate(dictionary.wavelength):
        wb = window_bounds(wavelength, float(centre), fit_cfg.window_nm, bounds_cfg)
        if wb is not None:
            resolved.append(wb)
            keep.append(j)
    keep = np.asarray(keep, dtype=int)
    features = fit_spectra(
        spectra, wavelength, resolved, dictionary.wavelength[keep],
        fit_cfg.gamma_init, fit_cfg.sigma_init, fit_cfg.r2_min,
        fit_cfg.min_snr, fit_cfg.maxfev, fit_cfg.max_shift_pixels,
    )
    return features, keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--shot-bin", type=int, default=None,
                        help="average consecutive shots (default from config)")
    parser.add_argument("--n-shots", type=int, default=None,
                        help="keep only the first N pulses (surface layer); "
                             "default from config, 0 = all shots")
    parser.add_argument("--norm-reference", default=None, choices=["shot", "bulk"],
                        help="intensity normalisation reference (default from config)")
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--compare-db", default=None,
                        help="second line database to benchmark the fits against")
    parser.add_argument("--compare-samples", type=int, default=4)
    parser.add_argument("--compare-rows", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    tok_cfg = cfg.get("tokens", {})
    shot_bin = args.shot_bin if args.shot_bin is not None else int(tok_cfg.get("shot_bin", 4))
    n_shots = args.n_shots if args.n_shots is not None else tok_cfg.get("n_shots")
    n_shots = int(n_shots) if n_shots else None
    fit_cfg = fit_config_from_config(cfg)

    pre = Preprocessor.from_config(cfg)
    if args.norm_reference is not None:
        pre = replace(pre, normalization_reference=args.norm_reference)

    dictionary = line_dictionary_from_config(cfg)
    counts = dictionary.counts_per_element()
    print("\nlines per element:")
    print("  " + "  ".join(f"{e}:{n}" for e, n in counts.items()))

    if args.compare_db:
        wavelength = load_wavelength(cfg)
        ids = [f"train_{i:03d}" for i in range(1, args.compare_samples + 1)]
        spectra = _sample_spectra(cfg, pre, shot_bin, ids, args.compare_rows, n_shots)
        print(f"\ndatabase comparison on {spectra.shape[0]} spectra")
        other = build_line_dictionary(
            args.compare_db,
            elements=tuple(tok_cfg.get("elements", dictionary.element.tolist())),
            wl_min=float(wavelength.min()), wl_max=float(wavelength.max()),
            cache_dir=cfg.cache_dir, verbose=False,
        )
        for name, dic in (("configured", dictionary), ("comparison", other)):
            feats, keep = _fit_against(cfg, dic, spectra, fit_cfg)
            valid = feats[..., F_VALID] > 0.5
            delta = np.abs(feats[..., F_DELTA][valid])
            median_delta = float(np.median(delta)) if delta.size else float("nan")
            print(f"  {name:<10s} {dic.n_lines:5d} lines  valid={valid.mean():6.1%}  "
                  f"median|dlambda|={median_delta:.4f} nm")

    tokens = build_tokens(
        cfg, dictionary, pre=pre, shot_bin=shot_bin, n_shots=n_shots,
        fit_cfg=fit_cfg, n_jobs=args.n_jobs, use_cache=not args.no_cache,
    )
    train = tokens.subset("train")
    valid = train.X[..., F_VALID] > 0.5

    print(f"\n{tokens}")
    print(f"fit_valid overall: {tokens.valid_fraction():.1%}  (train {valid.mean():.1%})")

    per_line_valid = valid.mean(axis=(0, 1))
    print(f"lines fitted in >50% of spectra: {(per_line_valid > 0.5).sum()} / {train.n_lines}")
    print(f"lines never fitted:              {(per_line_valid == 0).sum()}")

    if valid.any():
        delta = np.abs(train.X[..., F_DELTA][valid])
        r2 = train.X[..., F_R2][valid]
        print("\n|delta_lambda| (nm) percentiles: " + "  ".join(
            f"p{p}={np.percentile(delta, p):.4f}" for p in (10, 50, 90, 99)))
        print("r2 percentiles:                 " + "  ".join(
            f"p{p}={np.percentile(r2, p):.3f}" for p in (10, 50, 90, 99)))

    element_rows = []
    for element in pd.unique(train.line_element):
        mask = train.line_element == element
        element_rows.append({
            "element": element,
            "n_lines": int(mask.sum()),
            "valid_fraction": float(valid[:, :, mask].mean()),
            "mean_amplitude": float(train.X[..., F_MAX_INT][:, :, mask].mean()),
        })
    frame = pd.DataFrame(element_rows).sort_values("n_lines", ascending=False)
    print("\nper-element fit quality:")
    print(frame.to_string(index=False))

    out_csv = cfg.metrics_dir / "token_diagnostics.csv"
    frame.to_csv(out_csv, index=False)

    fig_path = cfg.figures_dir / "token_diagnostics.png"
    _plot_diagnostics(train, valid, fig_path)

    print(f"\ndiagnostics -> {out_csv}")
    print(f"figure      -> {fig_path}")


def _plot_diagnostics(train, valid, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    if valid.any():
        axes[0].hist(np.abs(train.X[..., F_DELTA][valid]), bins=60, color="steelblue")
        axes[0].set_xlabel("|delta_lambda| (nm)")
        axes[0].set_ylabel("tokens")
        axes[0].set_title("Fitted minus theoretical centre")

        axes[1].hist(train.X[..., F_R2][valid], bins=60, color="darkorange")
        axes[1].set_xlabel("Voigt fit R^2")
        axes[1].set_title("Fit quality")

    order = np.argsort(train.line_wavelength)
    axes[2].plot(train.line_wavelength[order],
                 valid.mean(axis=(0, 1))[order], lw=0.6, color="seagreen")
    axes[2].set_xlabel("line wavelength (nm)")
    axes[2].set_ylabel("fraction of spectra fitted")
    axes[2].set_title("Detection rate across the spectrum")

    plotting.save(fig, path)


if __name__ == "__main__":
    main()
