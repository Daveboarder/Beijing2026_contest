"""Unit tests for data-driven spectral-line tokens (no line database, no torch)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from libs2026.line_tokens import (
    D_AREA,
    D_CONT,
    D_DETECTED,
    D_HEIGHT,
    D_SHIFT,
    D_WIDTH,
    DESCRIPTOR_NAMES,
    N_DESCRIPTORS,
    LineTokens,
    detect_lines,
    eliminate_lines,
    estimate_channel_offsets,
    extract_line_tokens,
    identify_lines,
    inflection_bounds,
    line_areas,
    line_occlusion,
    smooth_nyquist,
    static_channels,
    usable_ranges,
)

BOUNDS = (0, 600, 1200, 1800)
# Three overlapping channels, like the Avantes spectrometer.
WAVELENGTH = np.concatenate([np.linspace(300, 360, 600), np.linspace(355, 415, 600),
                             np.linspace(410, 470, 600)])
STEP = 60 / 599
LINES_NM = [310.0, 322.0, 340.0, 371.0, 390.0, 425.0, 450.0]


def spectrum(amplitudes=None, shift_px=0.0, sigma_px=1.3, pattern=0.1, noise=0.0, seed=0,
             centres=LINES_NM):
    """Gaussian lines on a sloped continuum, times a fixed even/odd pixel pattern."""
    rng = np.random.default_rng(seed)
    amplitudes = amplitudes if amplitudes is not None else [4.0] * len(centres)
    x = np.zeros(WAVELENGTH.size)
    for a, b in zip(BOUNDS[:-1], BOUNDS[1:]):
        w = WAVELENGTH[a:b]
        seg = 1.0 + 0.002 * (w - w[0])
        for centre, amp in zip(centres, amplitudes):
            if w[0] + 1 < centre < w[-1] - 1:
                z = (w - centre - shift_px * STEP) / (sigma_px * STEP)
                seg = seg + amp * np.exp(-0.5 * z ** 2)
        x[a:b] = seg
    x *= 1.0 + pattern * (-1.0) ** np.arange(x.size)
    return x + rng.normal(scale=noise, size=x.size)


class DetectionTests(unittest.TestCase):
    def test_nyquist_smoothing_removes_even_odd_pattern(self):
        flat = 2.0 * (1.0 + 0.15 * (-1.0) ** np.arange(1800))
        smoothed = smooth_nyquist(flat[None, :], BOUNDS)[0]
        for a, b in zip(BOUNDS[:-1], BOUNDS[1:]):
            np.testing.assert_allclose(smoothed[a + 1:b - 1], 2.0, atol=1e-12)

    def test_usable_ranges_split_overlaps_at_midpoint(self):
        ranges = usable_ranges(WAVELENGTH, BOUNDS, edge_nm=0.5)
        self.assertAlmostEqual(ranges[0][0], 300.5)
        self.assertAlmostEqual(ranges[0][1], 357.5)
        self.assertAlmostEqual(ranges[1][0], 357.5)
        self.assertAlmostEqual(ranges[1][1], 412.5)
        self.assertAlmostEqual(ranges[2][1], 469.5)

    def test_detects_injected_lines_once_and_orders_pixels(self):
        lines = detect_lines(spectrum(noise=0.01, seed=1), WAVELENGTH, BOUNDS)
        found = lines["wavelength_nm"].to_numpy()
        distance = np.abs(found[:, None] - np.array(LINES_NM)[None, :])
        self.assertTrue(np.all(distance.min(axis=0) <= 1.01 * STEP), "every injected line is found")
        self.assertTrue(np.all((distance <= 1.01 * STEP).sum(axis=0) == 1), "and found only once")
        for row in lines.itertuples():
            self.assertTrue(row.bg_left < row.core_lo <= row.peak_px <= row.core_hi < row.bg_right)
        # 371 nm lies in the channel 1/2 overlap and must come from channel 2 only.
        overlap = (lines["wavelength_nm"] - 371).abs().idxmin()
        self.assertEqual(int(lines.loc[overlap, "channel"]), 2)

    def test_false_detections_on_patterned_noise_are_rare(self):
        # Pixel pattern + white noise, no lines. The detector passes ~0.1 noise peaks
        # per 1000 pixels (an unbounded prominence at the same threshold: ~28).
        n_false = sum(len(detect_lines(spectrum(amplitudes=[0.0] * len(LINES_NM), pattern=0.15,
                                                noise=0.002, seed=seed), WAVELENGTH, BOUNDS))
                      for seed in range(20))
        self.assertLessEqual(n_false, 3)            # 20 spectra x ~1700 usable pixels

    def test_no_lines_gives_an_empty_table_with_columns(self):
        lines = detect_lines(np.ones(WAVELENGTH.size), WAVELENGTH, BOUNDS)
        self.assertEqual(len(lines), 0)
        self.assertIn("wavelength_nm", lines.columns)


class DescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = detect_lines(spectrum(), WAVELENGTH, BOUNDS)

    def tokens(self, **kw):
        return extract_line_tokens(spectrum(**kw)[None, :], WAVELENGTH, self.lines, BOUNDS)

    def test_area_height_continuum_match_the_injected_line(self):
        amp, sigma_px = 4.0, 1.3
        tokens = self.tokens(sigma_px=sigma_px)
        self.assertEqual(tokens.shape, (1, len(LINES_NM), N_DESCRIPTORS))
        expected_area = amp * sigma_px * STEP * np.sqrt(2 * np.pi)
        # The +-FWHM core holds ~98 % of a Gaussian; [1,2,1] smoothing widens it slightly.
        np.testing.assert_allclose(tokens[0, :, D_AREA], expected_area, rtol=0.08)
        np.testing.assert_allclose(tokens[0, :, D_HEIGHT], amp, rtol=0.2)
        centre_cont = 1.0 + 0.002 * (self.lines["wavelength_nm"].to_numpy() - np.array(
            [WAVELENGTH[BOUNDS[c - 1]] for c in self.lines["channel"]]))
        np.testing.assert_allclose(tokens[0, :, D_CONT], centre_cont, atol=0.05)
        self.assertTrue(np.all(tokens[0, :, D_DETECTED] == 1))

    def test_shift_and_width_follow_the_profile(self):
        base = self.tokens()[0]
        moved, broad = self.tokens(shift_px=0.3)[0], self.tokens(sigma_px=1.8)[0]
        self.assertTrue(np.all(moved[:, D_SHIFT] > base[:, D_SHIFT]))
        self.assertTrue(np.all(broad[:, D_WIDTH] > base[:, D_WIDTH]))

    def test_absent_line_is_not_detected(self):
        amps = [4.0] * len(LINES_NM)
        amps[0] = 0.0
        tokens = self.tokens(amplitudes=amps, noise=0.02)
        first = int(np.argmin(np.abs(self.lines["wavelength_nm"].to_numpy() - LINES_NM[0])))
        self.assertEqual(tokens[0, first, D_DETECTED], 0)


def on_pixel(centres_nm):
    """Snap line centres to the nearest pixel of the channel that holds them."""
    snapped = []
    for c in centres_nm:
        a, b = next((a, b) for a, b in zip(BOUNDS[:-1], BOUNDS[1:])
                    if WAVELENGTH[a] < c < WAVELENGTH[b - 1])
        snapped.append(float(WAVELENGTH[a + np.argmin(np.abs(WAVELENGTH[a:b] - c))]))
    return snapped


class AreaTests(unittest.TestCase):
    SIGMA_PX = 3.0
    CENTRES = on_pixel(LINES_NM)

    @classmethod
    def setUpClass(cls):
        observed = spectrum(sigma_px=cls.SIGMA_PX, centres=cls.CENTRES)
        cls.lines = detect_lines(observed, WAVELENGTH, BOUNDS)
        cls.windows = inflection_bounds(observed, cls.lines, BOUNDS)

    def areas(self, baseline="linear", offset=0.0, amplitude=4.0):
        rows = spectrum(amplitudes=[amplitude] * len(self.CENTRES), sigma_px=self.SIGMA_PX,
                        pattern=0.0, centres=self.CENTRES)[None, :] + offset
        return line_areas(rows, WAVELENGTH, self.windows["b1_px"], self.windows["b2_px"],
                          baseline)[0]

    def test_inflection_bounds_sit_at_plus_minus_sigma(self):
        # The pixel pattern must not create extra inflections next to the centre;
        # [1,2,1]/4 smoothing only widens sigma = 3 px to sqrt(9.5) ~ 3.1 px.
        self.assertEqual(len(self.lines), len(LINES_NM))
        offsets = np.r_[self.lines["peak_px"] - self.windows["b1_px"],
                        self.windows["b2_px"] - self.lines["peak_px"]]
        self.assertTrue(np.all(offsets == 3), offsets)

    def test_linear_area_ignores_continuum_and_scales_with_amplitude(self):
        base = self.areas()
        # Gaussian cap above the chord between +-sigma: ~0.498 * sigma * amplitude.
        np.testing.assert_allclose(base, 0.498 * self.SIGMA_PX * STEP * 4.0, rtol=0.05)
        np.testing.assert_allclose(self.areas(offset=0.7), base, atol=1e-5)
        np.testing.assert_allclose(self.areas(amplitude=8.0), 2 * base, rtol=1e-5)

    def test_libsmethods_formula_keeps_most_of_the_continuum(self):
        n_px = (self.windows["b2_px"] - self.windows["b1_px"]).to_numpy()
        lifted = self.areas("libsmethods", offset=0.7) - self.areas("libsmethods")
        # trapezoid over b2 - b1 pixels (b2 excluded) minus a one-pixel-wide baseline
        np.testing.assert_allclose(lifted, 0.7 * (n_px - 2), atol=1e-4)


class IdentificationTests(unittest.TestCase):
    def test_offsets_are_recovered_and_lines_assigned(self):
        offset = -0.03
        lines = pd.DataFrame({
            "line_id": range(4), "channel": [1, 1, 1, 1],
            "wavelength_nm": np.array([310.0, 322.0, 340.0, 350.0]) + offset,
            "step_nm": STEP / 4, "fwhm_nm": STEP, "prominence": [5.0, 4.0, 3.0, 2.0]})
        candidates = pd.DataFrame({
            "element": ["Fe", "Cr", "Fe", "Mn", "Fe"], "ion_state": ["I", "I", "II", "I", "I"],
            "db_wavelength_nm": [310.0, 322.0, 340.0, 340.08, 360.0],
            "Ei": 0.0, "Ek": 3.0, "gi": 1.0, "gk": 3.0, "Ak": 1e7,
            "log_intensity": [2.0, 1.5, 1.8, 3.0, 2.0],
        }).sort_values("db_wavelength_nm", ignore_index=True)
        offsets = estimate_channel_offsets(lines, candidates, top_fraction=1.0)
        self.assertAlmostEqual(offsets[1], offset, delta=0.004)
        named = identify_lines(lines, candidates, offsets)
        self.assertEqual(named["assignment"].tolist()[:2], ["Fe I 310.000", "Cr I 322.000"])
        # Mn I is stronger but 0.08 nm (3 pixels) away: proximity wins.
        self.assertEqual(named["assignment"].iloc[2], "Fe II 340.000")
        self.assertEqual(named["assignment"].iloc[3], "?")
        static = static_channels(named)
        self.assertEqual(static.shape, (4, 9))
        self.assertAlmostEqual(float(static[0, 7]), 26.0)       # Fe
        self.assertAlmostEqual(float(static[2, 8]), 1.0)        # Fe II is ionised
        self.assertAlmostEqual(float(static[3, 0]), 350.0, places=3)


def token_problem(n_samples=40, n_rows=2, n_lines=12, informative=(3, 7), seed=0):
    """Row tokens ``(n_samples * n_rows, n_lines, 2)`` where only two lines carry the class."""
    rng = np.random.default_rng(seed)
    y_samples = np.tile([1, 2], n_samples // 2)
    X = rng.normal(size=(n_samples, n_rows, n_lines, 2)).astype(np.float32)
    for j in informative:
        X[y_samples == 2, :, j, 0] += 1.5
    return (X.reshape(-1, n_lines, 2), np.repeat(y_samples, n_rows),
            np.repeat(np.arange(n_samples), n_rows))


def fast_model(n_lines, n_desc):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))


class SelectionTests(unittest.TestCase):
    def test_occlusion_ranks_informative_lines_first(self):
        X, y, groups = token_problem()
        importance, drop = line_occlusion(X, y, groups, fast_model, n_splits=4, seed=1)
        self.assertEqual(drop.shape, (80, 12))
        self.assertEqual(set(np.argsort(importance)[-2:]), {3, 7})

    def test_elimination_keeps_informative_lines_and_ranks_everything(self):
        X, y, groups = token_problem()
        subsets, rank = eliminate_lines(X, y, groups, targets=[6, 2], model_factory=fast_model,
                                        seed=1)
        self.assertEqual(set(subsets[2]), {3, 7})
        self.assertTrue(set(subsets[2]) <= set(subsets[6]))
        self.assertEqual(sorted(rank), list(range(1, 13)))
        self.assertEqual(set(np.flatnonzero(rank <= 2)), {3, 7})


class ContainerTests(unittest.TestCase):
    def make(self):
        rng = np.random.default_rng(0)
        n, rows, n_lines = 4, 3, 5
        lines = pd.DataFrame({
            "line_id": range(n_lines), "wavelength_nm": np.linspace(300, 400, n_lines),
            "assignment": ["Fe I 300.000", "?", "Cr I 350.000", "?", "H I 400.000"]})
        stats = np.zeros((n_lines, N_DESCRIPTORS), np.float32)
        return LineTokens(rng.normal(size=(n, rows, n_lines, N_DESCRIPTORS)).astype(np.float32),
                          np.array([1.0, 2.0, np.nan, np.nan]), np.array(["a", "b", "c", "d"]),
                          np.array(["train", "train", "test", "test"]), lines,
                          np.zeros((n_lines, 9), np.float32), stats, stats + 1)

    def test_save_load_roundtrip(self):
        tokens = self.make()
        with tempfile.TemporaryDirectory() as tmp:
            loaded = LineTokens.load(tokens.save(Path(tmp) / "t.npz"))
        np.testing.assert_array_equal(loaded.X, tokens.X)
        np.testing.assert_array_equal(np.isnan(loaded.y), np.isnan(tokens.y))
        self.assertEqual(loaded.lines["assignment"].tolist(), tokens.lines["assignment"].tolist())
        self.assertEqual(loaded.descriptor_names, DESCRIPTOR_NAMES)

    def test_select_rows_and_cnn_layout(self):
        tokens = self.make().subset("train").select([4, 0]).select_descriptors(DESCRIPTOR_NAMES)
        X, y, groups, ids = tokens.rows()
        self.assertEqual(X.shape, (6, 2 * N_DESCRIPTORS))
        self.assertEqual(groups.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(tokens.lines["line_id"].tolist(), [4, 0])
        kw = tokens.cnn_kwargs()
        self.assertEqual((kw["n_rows"], kw["n_lines"], kw["n_features"]), (3, 2, N_DESCRIPTORS))
        # TokenCNN zeroes the shape channels (before the flag) of undetected lines, keeps continuum.
        self.assertEqual(kw["valid_index"], DESCRIPTOR_NAMES.index("detected"))
        self.assertEqual(kw["n_dynamic"], kw["valid_index"])
        self.assertGreater(DESCRIPTOR_NAMES.index("continuum"), kw["valid_index"])


if __name__ == "__main__":
    unittest.main()
