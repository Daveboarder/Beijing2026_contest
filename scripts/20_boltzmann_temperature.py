"""Fe plasma temperature per sample from Boltzmann (and Saha-Boltzmann) plots.

    uv run python scripts/20_boltzmann_temperature.py --n-jobs 12

Steps (see ``src/libs2026/boltzmann.py``):

1. Mean raw spectra per sample -- over all 200 shots and per log-spaced depth
   bin -- are cached once. No normalisation is applied: a Boltzmann plot needs
   the true intensity ratios between lines.
2. Fe I candidates from the air line database are screened for blends,
   self-absorption (ground multiplet) and weak signal.
3. The spectrometer is not radiometrically calibrated, and the three channels
   differ by orders of magnitude in response. Lines are therefore taken from
   one channel (default: channel 2, 380-632 nm, where Fe I is rich and clean),
   and a smooth ln-response tilt is fitted jointly with the Boltzmann lines.
4. Greedy backward elimination picks the Fe I combination giving the
   straightest Boltzmann plot (lowest median slope uncertainty).
5. Fe II lines are added on the Saha-Boltzmann axis only if enough of them
   survive screening in the same channel; with the 1 ms gate of this dataset
   the ionic emission is usually too weak outside the UV channel.
6. Temperatures are written per sample and depth bin, and class-mean
   Boltzmann plots are drawn. Labels are used only for figures and statistics;
   line selection is label-free and uses all 180 samples.
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from libs2026 import Config, load_index, load_shots, load_wavelength, log_bin_edges, plotting
from libs2026.boltzmann import (
    MeasureConfig,
    ScreenConfig,
    boltzmann_y,
    electron_density_halpha,
    fit_rows,
    instrument_fwhm,
    line_intensity,
    load_transitions,
    response_basis,
    saha_boltzmann_fit,
    screen_candidates,
    select_lines,
)
from libs2026.lines_db import STEEL_ELEMENTS, _get_eion
from libs2026.preprocessing import repair_outlier_shots, shot_outlier_mask


def _depth_spectra(cfg, sample_id, edges, outlier_z, outlier_window):
    shots = np.asarray(load_shots(cfg, sample_id, mmap=False), dtype=np.float32)
    shots = repair_outlier_shots(shots, shot_outlier_mask(shots, outlier_z, outlier_window))
    rows = [shots.mean(axis=0)] + [shots[sl].mean(axis=0) for sl in edges]
    return np.stack(rows)


def build_spectra(cfg, index, n_bins, n_jobs, overwrite=False):
    """``(n_samples, 1 + n_bins, n_wl)``: row 0 = all shots, then depth bins."""
    edges = log_bin_edges(cfg["data"]["n_shots"], n_bins)
    out = cfg.cache_dir / "boltzmann" / f"depth_spectra_b{n_bins}.npy"
    if out.exists() and not overwrite:
        return np.load(out), edges
    pre = cfg["preprocessing"]
    blocks = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_depth_spectra)(cfg, sid, edges, pre.get("outlier_z", 4.0), pre.get("outlier_window", 11))
        for sid in index["sample_id"]
    )
    spectra = np.stack(blocks).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, spectra)
    return spectra, edges


def corrected_y(spectra, lines, measure):
    """Response-corrected Boltzmann ordinate for every row of ``spectra``."""
    return boltzmann_y(lines, line_intensity(spectra, lines, measure)) - lines["response_ln"].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--db", default=None, help="air line database; default tokens.db_path")
    parser.add_argument("--n-bins", type=int, default=8, help="log-spaced depth bins")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--channel", type=int, default=2, help="spectrometer channel (1-based)")
    parser.add_argument("--response-degree", type=int, default=1,
                        help="Legendre degree of the fitted ln-response in wavelength; 0 = none")
    parser.add_argument("--intensity", choices=("height", "area"), default="height")
    parser.add_argument("--tol", type=float, default=0.25,
                        help="max |median residual| (ln units) a kept line may have")
    parser.add_argument("--min-lines", type=int, default=6)
    parser.add_argument("--min-span", type=float, default=1.5, help="min E_k span (eV)")
    parser.add_argument("--top-n", type=int, default=100000, help="strongest theoretical lines screened")
    parser.add_argument("--min-snr", type=float, default=10.0)
    parser.add_argument("--max-interference", type=float, default=0.5)
    parser.add_argument("--blend-fwhm", type=float, default=0.6,
                        help="neighbours closer than this many FWHM count as blends")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--tag", default="fe")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    db_path = args.db or cfg["tokens"]["db_path"]
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Line database not found: {db_path} (pass --db)")
    out_dir = cfg.results_dir / "boltzmann" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = tuple(cfg["data"]["channel_bounds"])

    wavelength = load_wavelength(cfg).astype(np.float64)
    index = load_index(cfg)
    spectra, edges = build_spectra(cfg, index, args.n_bins, args.n_jobs, args.overwrite_cache)
    whole = spectra[:, 0]                       # all-shot mean per sample
    reference = whole.mean(axis=0)              # grand mean, for centres/FWHM
    fwhm = instrument_fwhm(wavelength, reference, bounds)
    print("instrument FWHM per channel (nm):", ", ".join(f"{w:.3f}" for w in fwhm))
    a, b = bounds[args.channel - 1], bounds[args.channel]
    span = (float(wavelength[a:b].min()), float(wavelength[a:b].max()))

    measure = MeasureConfig(intensity=args.intensity)
    transitions = load_transitions(db_path, STEEL_ELEMENTS)

    # ---- Fe I: screening, response and line selection ---------------------
    screen_i = ScreenConfig(top_n=args.top_n, min_snr=args.min_snr, blend_fwhm=args.blend_fwhm,
                            max_interference=args.max_interference, channels=(args.channel,))
    cand_i = screen_candidates(transitions, "Fe", "I", wavelength, bounds, fwhm, reference,
                               whole, db_path, screen_i, measure)
    print("Fe I screening:", cand_i["status"].value_counts().to_dict())
    ok_i = cand_i[cand_i["status"] == "ok"].copy()
    y_raw = boltzmann_y(ok_i, line_intensity(whole, ok_i, measure))
    x_i = ok_i["Ek"].to_numpy()
    wl_i = ok_i["wavelength"].to_numpy()

    mask_raw, _, _ = select_lines(x_i, y_raw, args.tol, args.min_lines, args.min_span)
    fit_raw = fit_rows(x_i[mask_raw], y_raw[:, mask_raw])
    basis_i = response_basis(wl_i, *span, args.response_degree)
    mask_i, path_i, coef = select_lines(x_i, y_raw, args.tol, args.min_lines, args.min_span,
                                        basis=basis_i)
    ok_i["response_ln"] = basis_i @ coef
    y_i = y_raw - ok_i["response_ln"].to_numpy()
    path_i["removed_line"] = ["" if r < 0 else f"{wl_i[r]:.3f}" for r in path_i["removed"]]
    cand_i["selected"] = False
    cand_i.loc[ok_i.index[mask_i], "selected"] = True
    cand_i["response_ln"] = ok_i["response_ln"]
    sel_i = ok_i[mask_i]
    fit_sel = fit_rows(x_i[mask_i], y_i[:, mask_i])
    print(f"Fe I without response correction: {mask_raw.sum()} lines, median R2 "
          f"{np.nanmedian(fit_raw.r2):.3f}, median T {np.nanmedian(fit_raw.temperature):,.0f} K")
    print(f"Fe I with response (degree {args.response_degree}, coef {np.round(coef, 3)}): "
          f"{len(ok_i)} candidates -> {mask_i.sum()} lines, median R2 {np.nanmedian(fit_sel.r2):.3f}, "
          f"median T {np.nanmedian(fit_sel.temperature):,.0f} K, Ek span {np.ptp(x_i[mask_i]):.2f} eV")

    # ---- Electron density from H-alpha --------------------------------------
    ha_channel = next(c for c, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])) if
                      wavelength[lo:hi].min() + 5 < 656.28 < wavelength[lo:hi].max() - 5)
    lo, hi = bounds[ha_channel], bounds[ha_channel + 1]
    ne_frame = pd.DataFrame([electron_density_halpha(wavelength[lo:hi], s[lo:hi], fwhm[ha_channel])
                             for s in whole])
    ne_used = ne_frame["ne"].fillna(ne_frame["ne"].median()).to_numpy()
    print(f"n_e from H-alpha: median {ne_frame['ne'].median():.2e} cm^-3, "
          f"{ne_frame['halpha_ok'].mean():.0%} fits valid")

    # ---- Fe II + Saha-Boltzmann (same channel only) ----------------------------
    screen_ii = ScreenConfig(top_n=args.top_n, min_snr=args.min_snr / 2, ne=float(np.median(ne_used)),
                             max_interference=2 * args.max_interference, min_ei=-1.0,
                             blend_fwhm=args.blend_fwhm, channels=(args.channel,))
    cand_ii = screen_candidates(transitions, "Fe", "II", wavelength, bounds, fwhm, reference,
                                whole, db_path, screen_ii, measure)
    cand_ii["selected"] = False
    ok_ii = cand_ii[cand_ii["status"] == "ok"].copy()
    print("Fe II screening:", cand_ii["status"].value_counts().to_dict())
    e_ion = _get_eion("Fe", db_path)
    sb_lines = None
    if len(ok_ii) >= 2:
        ok_ii["response_ln"] = response_basis(ok_ii["wavelength"], *span, args.response_degree) @ coef
        y_ii = corrected_y(whole, ok_ii, measure)
        _, x_sb, y_sb = saha_boltzmann_fit(x_i[mask_i], y_i[:, mask_i], ok_ii["Ek"].to_numpy(), y_ii,
                                           e_ion, ne_used, fit_sel.temperature)
        removable = np.r_[np.zeros(mask_i.sum(), bool), np.ones(len(ok_ii), bool)]
        mask_sb, path_sb, _ = select_lines(x_sb[0], y_sb, args.tol, mask_i.sum() + 2, args.min_span,
                                           removable)
        mask_ii = mask_sb[mask_i.sum():]
        cand_ii.loc[ok_ii.index[mask_ii], "selected"] = True
        sb_lines = pd.concat([sel_i, ok_ii[mask_ii]])
        path_sb.to_csv(out_dir / "selection_path_saha.csv", index=False)
        print(f"Fe II: {len(ok_ii)} candidates -> {mask_ii.sum()} kept for Saha-Boltzmann")
    else:
        print(f"Saha-Boltzmann skipped: only {len(ok_ii)} measurable, unblended Fe II lines in "
              f"channel {args.channel}")

    pd.concat([cand_i, cand_ii]).to_csv(out_dir / "line_candidates.csv", index=False)
    path_i.to_csv(out_dir / "selection_path_boltzmann.csv", index=False)
    sel_i.to_csv(out_dir / "selected_lines_boltzmann.csv", index=False)
    if sb_lines is not None:
        sb_lines.to_csv(out_dir / "selected_lines_saha.csv", index=False)

    # ---- Temperatures per sample and depth bin -------------------------------
    n_i = len(sel_i)

    def temperatures(rows):
        yb = corrected_y(rows, sel_i, measure)
        fb = fit_rows(sel_i["Ek"].to_numpy(), yb)
        out = {"T_boltz": fb.temperature, "T_boltz_se": fb.temperature_se,
               "r2_boltz": fb.r2, "intercept_boltz": fb.intercept}
        if sb_lines is not None:
            ys = corrected_y(rows, sb_lines, measure)
            fs, _, _ = saha_boltzmann_fit(sel_i["Ek"].to_numpy(), ys[:, :n_i], sb_lines["Ek"].to_numpy()[n_i:],
                                          ys[:, n_i:], e_ion, ne_used, fb.temperature)
            out.update({"T_saha": fs.temperature, "T_saha_se": fs.temperature_se, "r2_saha": fs.r2})
        return out

    records = pd.concat([index[["sample_id", "split", "label"]], ne_frame], axis=1)
    for key, val in temperatures(whole).items():
        records[key] = val
    depth_rows = []
    for k, sl in enumerate(edges):
        frame = index[["sample_id", "split", "label"]].copy()
        frame["depth_bin"], frame["shot_start"], frame["shot_stop"] = k, sl.start + 1, sl.stop
        for key, val in temperatures(spectra[:, k + 1]).items():
            frame[key] = val
        depth_rows.append(frame)
    depth = pd.concat(depth_rows, ignore_index=True)
    records.to_csv(out_dir / "sample_temperatures.csv", index=False)
    depth.to_csv(out_dir / "depth_temperatures.csv", index=False)

    # ---- Statistics against the aging level -----------------------------------
    train = records[records["split"] == "train"].copy()
    train["label"] = train["label"].astype(int)
    summary = []
    for col in [c for c in ("T_boltz", "T_saha", "ne", "r2_boltz") if c in train]:
        valid = train[np.isfinite(train[col])]
        rho, p_rho = stats.spearmanr(valid["label"], valid[col])
        groups = [g[col].to_numpy() for _, g in valid.groupby("label")]
        h, p_kw = stats.kruskal(*groups)
        per_level = valid.groupby("label")[col]
        summary.append({"quantity": col, "spearman_rho": rho, "spearman_p": p_rho,
                        "kruskal_H": h, "kruskal_p": p_kw,
                        **{f"mean_L{lv}": v for lv, v in per_level.mean().items()},
                        **{f"sem_L{lv}": v for lv, v in per_level.sem().items()}})
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "class_statistics.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ---- Figures ----------------------------------------------------------------
    is_train = (index["split"] == "train").to_numpy()
    labels = index.loc[is_train, "label"].astype(int).to_numpy()
    title = "Boltzmann plot, Fe I" + (" (response-corrected)" if args.response_degree else "")
    panels = [(title, sel_i["Ek"].to_numpy(), corrected_y(whole[is_train], sel_i, measure),
               train["T_boltz"].to_numpy(), sel_i)]
    if sb_lines is not None:
        ys = corrected_y(whole[is_train], sb_lines, measure)
        fs, x_sb, y_sb = saha_boltzmann_fit(sel_i["Ek"].to_numpy(), ys[:, :n_i], sb_lines["Ek"].to_numpy()[n_i:],
                                            ys[:, n_i:], e_ion, ne_used[is_train], train["T_boltz"].to_numpy())
        panels.append(("Saha-Boltzmann plot, Fe I + Fe II", x_sb, y_sb, fs.temperature, sb_lines))
    figs = cfg.figures_dir
    plotting.plot_boltzmann_by_class(panels, labels, figs / f"boltzmann_by_class_{args.tag}.png")
    plotting.plot_temperature_by_class(train, depth[depth["split"] == "train"],
                                       figs / f"temperature_by_class_{args.tag}.png")
    plotting.plot_line_selection(cand_i, path_i, x_i, y_i, mask_i,
                                 figs / f"boltzmann_line_selection_{args.tag}.png",
                                 y_raw=y_raw, response_ln=ok_i["response_ln"].to_numpy())
    print(f"Results: {out_dir}\nFigures: {figs}")


if __name__ == "__main__":
    main()
