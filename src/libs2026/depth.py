"""Depth-resolved encodings of a shot sequence.

The 200 pulses of a sample are fired into the *same spot*, so each successive
shot ablates deeper material. Shot index is therefore a depth coordinate, and
because steel ages from the surface inwards, the way the spectrum evolves from
shot 1 to shot 200 is itself the quantity that distinguishes aging levels.

Measured on the training set, the profile has three regimes:

* shot 1 -- surface layer, 1.4-1.9x the bulk emission;
* shots ~3-10 -- a pronounced minimum, the aged/oxidised layer. The depth of
  this dip is what separates the classes most clearly (Cr I 425.4 nm drops to
  ~0.59 of bulk for level 1 but to ~0.46 for level 4);
* shots ~100-200 -- a plateau at the bulk composition.

Depth bins are therefore spaced logarithmically: the surface, where the classes
differ, gets fine resolution and the flat bulk gets one wide bin.

Full-spectrum depth contrasts (``depth_bins``, ``mean_depth``) add tens of
thousands of noisy channels and underperform plain averaging. Compact line
profiles (``mean_lines``) keep only a handful of diagnostic wavelengths whose
depth curves actually track aging, and append them to the mean spectrum.
"""

from __future__ import annotations

import numpy as np

# Wavelengths (nm) of lines whose depth curves track aging on this dataset.
# Indices are resolved against the recorded wavelength axis at feature-build time.
DIAGNOSTIC_LINES_NM: dict[str, float] = {
    "C I 247.9": 247.86,
    "Fe I 404.6": 404.58,
    "Cr I 425.4": 425.43,
    "Mn I 403.1": 403.08,
    "H I 656.3": 656.28,
    "O I 777.4": 777.42,
}

ENCODINGS = (
    "mean",
    "depth_bins",
    "depth_contrast",
    "depth_profile",
    "mean_depth",
    "mean_lines",
    "lines",
)


def line_indices(wavelength: np.ndarray,
                 lines_nm: dict[str, float] | None = None) -> np.ndarray:
    """Nearest detector pixel for each diagnostic line, in a fixed order."""
    lines_nm = lines_nm or DIAGNOSTIC_LINES_NM
    return np.array([int(np.argmin(np.abs(wavelength - wl))) for wl in lines_nm.values()])


def log_bin_edges(n_shots: int, n_bins: int = 8) -> list[slice]:
    """``n_bins`` contiguous shot ranges, finely spaced near the surface."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if n_bins == 1:
        return [slice(0, n_shots)]
    # Geometric edges, de-duplicated so that the first bins stay single shots.
    raw = np.unique(np.round(np.geomspace(1, n_shots, n_bins + 1)).astype(int))
    edges = [0] + [int(e) for e in raw if 0 < e < n_shots] + [n_shots]
    edges = sorted(set(edges))
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def bin_means(processed: np.ndarray, edges: list[slice]) -> np.ndarray:
    """Mean spectrum of each depth bin, shaped ``(n_bins, n_wavelengths)``."""
    return np.vstack([processed[sl].mean(axis=0) for sl in edges])


def line_depth_descriptors(block: np.ndarray, line_idx: np.ndarray,
                           edges: list[slice], bulk: slice) -> np.ndarray:
    """Compact depth curve of each diagnostic line, relative to the bulk.

    For every line the vector holds, in order:

    * surface (first-shot) / bulk,
    * minimum over the shallow half / bulk (the aged-layer dip),
    * and the mean of each log-spaced depth bin / bulk.

    That is the information that actually separates aging levels, without
    paying for 12k noisy per-wavelength contrasts.
    """
    bulk_mean = block[bulk].mean(axis=0)
    shallow_n = max(block.shape[0] // 10, 5)  # first ~10 % of this block
    parts = []
    for i in line_idx:
        scale = max(float(bulk_mean[i]), 1e-6)
        profile = block[:, i] / scale
        bin_vals = np.array([profile[sl].mean() for sl in edges], dtype=np.float32)
        parts.append(np.concatenate([
            [profile[0], profile[:shallow_n].min()],
            bin_vals,
        ]))
    return np.concatenate(parts).astype(np.float32)


def encode_bins(means: np.ndarray, bulk_spectrum: np.ndarray, overall: np.ndarray,
                encoding: str, line_desc: np.ndarray | None = None) -> np.ndarray:
    """Assemble one feature vector from per-bin mean spectra and line descriptors."""
    if encoding == "mean":
        return overall
    if encoding == "depth_bins":
        return means.ravel()
    if encoding == "depth_contrast":
        # Differences, not ratios: after bulk-referenced normalisation the
        # spectra share a scale, and ratios would explode wherever the bulk
        # signal is only noise.
        return np.concatenate([bulk_spectrum, (means - bulk_spectrum).ravel()])
    if encoding == "depth_profile":
        shallow = means[: max(len(means) // 2, 1)]
        return np.concatenate([
            bulk_spectrum,
            means[0] - bulk_spectrum,              # outermost layer
            shallow.min(axis=0) - bulk_spectrum,   # depth of the aged-layer dip
            shallow.mean(axis=0) - bulk_spectrum,  # its average strength
        ])
    if encoding == "mean_depth":
        # The full-average spectrum carries most of the signal on its own, so
        # this keeps it and only appends the aged-layer contrast. The surface
        # bins are averaged together first: a bin holding one or two shots is
        # far too noisy across 12282 channels to contrast wavelength by
        # wavelength.
        shallow = means[: max(len(means) // 2, 1)]
        return np.concatenate([overall, shallow.mean(axis=0) - bulk_spectrum])
    if encoding == "mean_lines":
        if line_desc is None:
            raise ValueError("mean_lines encoding requires diagnostic line indices")
        return np.concatenate([overall, line_desc])
    if encoding == "lines":
        if line_desc is None:
            raise ValueError("lines encoding requires diagnostic line indices")
        return line_desc
    raise ValueError(f"Unknown encoding '{encoding}'. Available: {ENCODINGS}")


AUGMENTATIONS = ("blocks", "interleaved", "surface")


def row_shot_sets(n_shots: int, n_groups: int, augment: str, n_bins: int) -> list[np.ndarray]:
    """Shot indices behind each feature row of one sample.

    ``blocks`` cuts the sequence into consecutive ranges, so each row describes a
    *different depth range* of the same sample. That helps as data augmentation,
    but deep blocks of an aged sample look like bulk of any sample yet still
    carry the aging label -- a form of label noise.

    ``surface`` avoids that: only the first half of the shots (surface + aged
    layer + early recovery) is used, split into consecutive blocks. Deep bulk
    shots never become training rows of their own.

    ``interleaved`` takes every ``n_groups``-th shot within each depth bin, so
    every row spans the full depth range but uses independent shots.
    """
    if n_groups <= 1:
        return [np.arange(n_shots)]
    if augment == "blocks":
        return [idx for idx in np.array_split(np.arange(n_shots), n_groups) if idx.size]
    if augment == "surface":
        # Keep the depth region where classes separate (surface -> recovery).
        keep = max(n_shots // 2, n_groups)
        return [idx for idx in np.array_split(np.arange(keep), n_groups) if idx.size]
    if augment != "interleaved":
        raise ValueError(f"Unknown augmentation '{augment}'. Available: {AUGMENTATIONS}")

    subsets = []
    for offset in range(n_groups):
        picked = []
        for sl in log_bin_edges(n_shots, n_bins):
            idx = np.arange(sl.start, sl.stop)[offset::n_groups]
            # A bin narrower than n_groups runs out of shots for some offsets;
            # reuse its nearest shot so every row keeps all depth bins.
            picked.append(idx if idx.size else np.array([min(sl.start + offset, sl.stop - 1)]))
        subsets.append(np.concatenate(picked))
    return subsets


def encode_sample(
    processed: np.ndarray,
    encoding: str = "mean",
    n_bins: int = 8,
    bulk_fraction: float = 0.3,
    n_groups: int = 1,
    augment: str = "blocks",
    line_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Encode one preprocessed shot sequence into ``n_groups`` feature rows."""
    processed = np.asarray(processed, dtype=np.float32)
    needs_lines = encoding in ("mean_lines", "lines")
    if needs_lines and line_idx is None:
        raise ValueError(f"encoding '{encoding}' needs line_idx from the wavelength axis")

    # Full-spectrum depth encodings need many bins; line encodings always do.
    use_bins = 1 if encoding == "mean" else n_bins
    rows = []
    for subset in row_shot_sets(processed.shape[0], max(n_groups, 1), augment, n_bins):
        block = processed[subset]
        n = block.shape[0]
        edges = log_bin_edges(n, use_bins)
        means = bin_means(block, edges)
        bulk = slice(min(int(round(n * (1.0 - bulk_fraction))), n - 1), n)
        line_desc = None
        if needs_lines:
            line_desc = line_depth_descriptors(block, line_idx, edges, bulk)
        rows.append(encode_bins(means, block[bulk].mean(axis=0), block.mean(axis=0),
                                encoding, line_desc))
    return np.vstack(rows).astype(np.float32)
