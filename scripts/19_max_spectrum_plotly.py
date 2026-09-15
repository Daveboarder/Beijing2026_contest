"""Interactive Plotly max-over-shots spectra — one random sample per aging class.

For each selected file and wavelength, intensity = max over the 200 depth shots.

uv run python scripts/19_max_spectrum_plotly.py
uv run python scripts/19_max_spectrum_plotly.py --seed 42 --no-open
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from libs2026 import Config, Preprocessor, load_index, load_shots, load_wavelength


def _pick_one_per_class(train, rng):
    """Return one row per aging label, sampled without replacement within class."""
    rows = []
    for label in sorted(train["label"].dropna().unique()):
        subset = train[train["label"] == label]
        if subset.empty:
            continue
        rows.append(subset.iloc[int(rng.integers(0, len(subset)))])
    if not rows:
        raise SystemExit("No labelled training samples")
    return rows


def _envelope(cfg, sample_id: str, preprocessed: bool):
    shots = load_shots(cfg, sample_id, mmap=False)
    if preprocessed:
        shots = Preprocessor.from_config(cfg)(shots)
    if shots.ndim != 2:
        raise SystemExit(f"{sample_id}: unexpected shot matrix shape {shots.shape}")
    envelope = np.max(shots, axis=0)
    argmax_shot = np.argmax(shots, axis=0).astype(np.int32) + 1
    return envelope, argmax_shot, shots.shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for per-class sample picks (default: nondeterministic)")
    parser.add_argument("--preprocessed", action="store_true",
                        help="apply config Preprocessor before taking the max; "
                             "default uses raw shot intensities")
    parser.add_argument("--no-open", action="store_true",
                        help="write HTML only; do not open a browser")
    parser.add_argument("--output", default=None,
                        help="HTML path (default: results/figures/max_spectrum_by_class.html)")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    bounds = tuple(cfg["data"]["channel_bounds"])
    index = load_index(cfg)
    train = index[index["split"] == "train"].reset_index(drop=True)
    if train.empty:
        raise SystemExit("No training samples in the index")

    rng = np.random.default_rng(args.seed)
    picks = _pick_one_per_class(train, rng)
    wavelength = load_wavelength(cfg)
    prep_label = "preprocessed" if args.preprocessed else "raw"
    colors = qualitative.Plotly

    fig = go.Figure()
    chosen = []
    for i, row in enumerate(picks):
        sample_id = str(row["sample_id"])
        label = int(row["label"])
        envelope, argmax_shot, shape = _envelope(cfg, sample_id, args.preprocessed)
        if shape[1] != len(wavelength):
            raise SystemExit(f"{sample_id}: wavelength length mismatch")
        fig.add_trace(go.Scattergl(
            x=wavelength,
            y=envelope,
            mode="lines",
            name=f"level {label} ({sample_id})",
            line=dict(width=1.2, color=colors[i % len(colors)]),
            customdata=np.column_stack([argmax_shot]),
            hovertemplate=(
                f"level {label} · {sample_id}<br>"
                "λ = %{x:.2f} nm<br>"
                "max intensity = %{y:.4g}<br>"
                "max at shot %{customdata[0]}<extra></extra>"
            ),
        ))
        peak = int(np.argmax(envelope))
        chosen.append(dict(
            label=label, sample_id=sample_id, shape=shape,
            peak_nm=float(wavelength[peak]), peak_shot=int(argmax_shot[peak]),
            ymax=float(envelope.max()),
        ))
        print(f"level {label}: {sample_id}  peak={wavelength[peak]:.2f} nm "
              f"(shot {argmax_shot[peak]})  max={envelope.max():.4g}")

    for edge in bounds[1:-1]:
        fig.add_vline(x=float(wavelength[edge]), line_width=1, line_dash="dot",
                      line_color="grey", opacity=0.5)

    fig.update_layout(
        title=dict(text=f"Max-over-shots spectra — one train sample per class ({prep_label})"),
        xaxis_title="wavelength (nm)",
        yaxis_title="max intensity across 200 shots (a.u.)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(title="Aging level (sample)"),
        margin=dict(l=60, r=20, t=60, b=50),
    )

    out = Path(args.output) if args.output else (
        cfg.figures_dir / "max_spectrum_by_class.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"figure -> {out}")

    if not args.no_open:
        webbrowser.open_new_tab(out.resolve().as_uri())


if __name__ == "__main__":
    main()
