"""Depth profile of one emission line for a representative sample of each aging level.

    python scripts/22_line_depth_profile.py --wl 588.86 --name "Na I D2"

The net line intensity is measured in every one of the 200 shots (shot index =
depth) as the background-corrected integral over a window around the line, with
a linear background from both flanks. The representative of a level is its
medoid: the training sample whose log-intensity depth profile is closest (RMS)
to the level's median profile, so an unusual sample is never shown as typical.

The default is the Na D2 resonance line. In this dataset it appears at
588.87-588.92 nm on channel 2, about 0.1 nm below its tabulated air wavelength
(588.995 nm), so the peak is searched for near ``--wl`` rather than assumed.
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from libs2026 import Config, load_index, load_shots, load_wavelength, plotting  # noqa: E402
from libs2026.preprocessing import repair_outlier_shots, shot_outlier_mask  # noqa: E402


def line_window(wavelength, bounds, wl, search_nm, half_nm, bg_left, bg_right, channel, reference):
    """Pixel indices of the peak window and of both background ranges on one channel.

    Background ranges are given explicitly in nm because the obvious symmetric
    flanks often hit a neighbour -- for Na D2 the right flank would sit on D1.
    """
    a, b = bounds[channel - 1], bounds[channel]
    seg = wavelength[a:b]
    step = float(np.median(np.diff(seg)))
    near = np.flatnonzero(np.abs(seg - wl) <= search_nm)
    peak = a + near[np.argmax(reference[a + near])]
    half = max(1, int(round(half_nm / step)))
    window = np.arange(peak - half, peak + half + 1)
    left = a + np.flatnonzero((seg >= bg_left[0]) & (seg <= bg_left[1]))
    right = a + np.flatnonzero((seg >= bg_right[0]) & (seg <= bg_right[1]))
    if not left.size or not right.size:
        raise ValueError("Background range outside the channel")
    return peak, window, left, right, step


def net_intensity(shots, window, left, right, step):
    """Background-corrected integral per shot (counts*nm)."""
    bl = np.median(shots[:, left], axis=1)
    br = np.median(shots[:, right], axis=1)
    xl, xr = left.mean(), right.mean()
    bg = bl[:, None] + (br - bl)[:, None] * (window - xl) / (xr - xl)
    return (shots[:, window] - bg).sum(axis=1) * step


def _profile(cfg, sample_id, geometry, channel_slice, pre_cfg):
    shots = np.asarray(load_shots(cfg, sample_id, mmap=False), dtype=np.float64)
    shots = repair_outlier_shots(shots, shot_outlier_mask(shots, pre_cfg.get("outlier_z", 4.0),
                                                          pre_cfg.get("outlier_window", 11)))
    window, left, right, step = geometry
    line = net_intensity(shots, window, left, right, step)
    total = shots[:, channel_slice].sum(axis=1)
    return line, total, float(shots[:, window].max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--wl", type=float, default=588.86, help="approximate line position (nm)")
    parser.add_argument("--name", default="Na I D2")
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--search-nm", type=float, default=0.1, help="peak search radius")
    parser.add_argument("--half-nm", type=float, default=0.25, help="integration half-width")
    parser.add_argument("--bg-left", default="588.05,588.30", help="left background range (nm)")
    parser.add_argument("--bg-right", default="589.90,590.20",
                        help="right background range (nm); default clears the Na D1 line")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    bounds = tuple(cfg["data"]["channel_bounds"])
    wavelength = load_wavelength(cfg).astype(np.float64)
    index = load_index(cfg)
    train = index[index["split"] == "train"].reset_index(drop=True)

    # Locate the peak on the surface shots, where the line is strongest.
    surface = np.mean([np.asarray(load_shots(cfg, s)[:3]).mean(axis=0) for s in train["sample_id"][:20]],
                      axis=0)
    bg_left = tuple(float(v) for v in args.bg_left.split(","))
    bg_right = tuple(float(v) for v in args.bg_right.split(","))
    peak, window, left, right, step = line_window(wavelength, bounds, args.wl, args.search_nm, args.half_nm,
                                                  bg_left, bg_right, args.channel, surface)
    print(f"{args.name}: peak at {wavelength[peak]:.3f} nm (channel {args.channel}); window "
          f"{wavelength[window[0]]:.2f}-{wavelength[window[-1]]:.2f} nm, background "
          f"{wavelength[left[0]]:.2f}-{wavelength[left[-1]]:.2f} / "
          f"{wavelength[right[0]]:.2f}-{wavelength[right[-1]]:.2f} nm")

    ch = slice(bounds[args.channel - 1], bounds[args.channel])
    out = Parallel(n_jobs=args.n_jobs, verbose=2)(
        delayed(_profile)(cfg, s, (window, left, right, step), ch, cfg["preprocessing"])
        for s in train["sample_id"]
    )
    line = np.stack([o[0] for o in out])          # (n_samples, n_shots)
    total = np.stack([o[1] for o in out])
    max_counts = np.array([o[2] for o in out])
    labels = train["label"].astype(int).to_numpy()
    shots = np.arange(1, line.shape[1] + 1)

    # Medoid per level on the log profile (floored so near-zero bulk shots don't dominate).
    floor = np.percentile(line[:, -50:], 75)
    log_profile = np.log10(np.maximum(line, max(floor, 1e-3)))
    reps = {}
    for c in np.unique(labels):
        m = np.flatnonzero(labels == c)
        dist = np.sqrt(((log_profile[m] - np.median(log_profile[m], axis=0)) ** 2).mean(axis=1))
        reps[c] = m[np.argmin(dist)]

    tag = args.name.lower().replace(" ", "_")
    frame = pd.DataFrame(line, columns=[f"shot_{i}" for i in shots])
    frame.insert(0, "label", labels)
    frame.insert(0, "sample_id", train["sample_id"])
    frame.to_csv(cfg.metrics_dir / f"depth_profile_{tag}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for c, color in zip(sorted(reps), plotting.CLASS_COLORS):
        i = reps[c]
        sid = train.loc[i, "sample_id"]
        # Log axis: the few bulk shots whose net intensity dips to <= 0 are noise and are hidden.
        raw = np.where(line[i] > 0, line[i], np.nan)
        axes[0].plot(shots, raw, color=color, lw=1.2, marker="o", ms=2.5, label=f"level {c}: {sid}")
        axes[1].plot(shots, raw / total[i] * 1e3, color=color, lw=1.2, marker="o", ms=2.5,
                     label=f"level {c}: {sid}")
        print(f"level {c}: {sid}  shot-1 {line[i, 0]:.0f}, mean shots 101-200 {line[i, 100:].mean():.1f} "
              f"counts*nm, raw max {max_counts[i]:.0f} counts")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.3)
        ax.set_xlabel("shot number (increasing depth)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("net integrated intensity (counts·nm)")
    axes[0].set_title(f"{args.name} {wavelength[peak]:.2f} nm — raw")
    axes[1].set_ylabel(f"line / total channel-{args.channel} emission (×10⁻³)")
    axes[1].set_title("normalised to the shot's total emission")
    fig.suptitle(f"{args.name} depth profile — class medoid of each aging level")
    fig.tight_layout()
    path = plotting.save(fig, cfg.figures_dir / f"depth_profile_{tag}.png")
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
