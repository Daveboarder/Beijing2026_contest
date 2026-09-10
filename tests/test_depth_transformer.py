import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

from libs2026.config import Config
from libs2026.depth_transformer import (
    DepthTransformerClassifier,
    depth_windows,
    prepare_sample,
)
from libs2026.preprocessing import Preprocessor


class DepthTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(20, 8, 51)).astype(np.float32)
        y = np.tile([1, 5], 10)
        X[y == 5, :4, :16] += 2
        return X, y

    def model(self, **kwargs):
        params = dict(channel_widths=(16, 16, 16), n_shots=12,
                      surface_shots=4, late_bin=2, bulk_start=8,
                      d_model=16, heads=2, ff_dim=32, epochs=2, patience=1,
                      batch_size=4, dropout=0, device="cpu")
        return DepthTransformerClassifier(**{**params, **kwargs})

    def test_depth_windows_preserve_surface_and_cover_all_shots(self):
        windows = depth_windows()
        self.assertEqual(len(windows), 65)
        self.assertEqual(windows[:20], [(i, i + 1) for i in range(20)])
        self.assertEqual([i for a, b in windows for i in range(a, b)], list(range(200)))
        self.assertEqual(depth_windows(23, 20, 4)[-1], (20, 23))

    def test_shape_normalization_and_intensity_are_separate(self):
        shots = np.ones((12, 96), dtype=np.float32)
        shots[:4] *= 2
        pre = Preprocessor(channel_bounds=(0, 32, 64, 96), repair_outlier_shots=False)
        packed = prepare_sample(shots, pre, 2, 4, 2)
        self.assertEqual(packed.shape, (8, 51))
        np.testing.assert_allclose(packed[:, :48], 1, atol=1e-6)
        np.testing.assert_allclose(packed[:4, -3:], np.log(3), atol=1e-6)
        np.testing.assert_allclose(packed[4:, -3:], np.log(2), atol=1e-6)
        np.testing.assert_allclose(packed, prepare_sample(shots * 5, pre, 2, 4, 2), atol=1e-6)

    def test_channel_boundaries_do_not_mix(self):
        shots = np.tile(np.arange(1, 97, dtype=np.float32), (12, 1))
        pre = Preprocessor(channel_bounds=(0, 32, 64, 96), repair_outlier_shots=False)
        first = prepare_sample(shots, pre, 2, 4, 2)
        shots[:, 32:64] = shots[:, 32:64][:, ::-1]
        second = prepare_sample(shots, pre, 2, 4, 2)
        np.testing.assert_allclose(first[:, :16], second[:, :16])
        np.testing.assert_allclose(first[:, 32:48], second[:, 32:48])
        self.assertFalse(np.allclose(first[:, 16:32], second[:, 16:32]))

    def test_all_encoders_train_and_probability_columns_match_labels(self):
        X, y = self.data()
        for encoder in ["pool", "conv", "transformer"]:
            with self.subTest(encoder=encoder):
                model = self.model(depth_encoder=encoder).fit(X, y)
                p = model.predict_proba(X)
                self.assertEqual(p.shape, (20, 2))
                np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
                np.testing.assert_array_equal(model.classes_, [1, 5])
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(set(model.predict(X)).issubset({1, 5}))
                self.assertGreaterEqual(model.best_epoch_, 1)

    def test_inner_validation_does_not_influence_selection_scaler(self):
        X, y = self.data()
        first = self.model().fit(X, y)
        changed = X.copy()
        changed[first.inner_val_indices_] += 100
        second = self.model().fit(changed, y)
        np.testing.assert_array_equal(first.selection_mean_, second.selection_mean_)
        np.testing.assert_array_equal(first.selection_std_, second.selection_std_)
        expected, _ = first._stats(X[first.inner_train_indices_])
        np.testing.assert_array_equal(first.selection_mean_, expected)
        # Final refit deliberately uses all samples passed to fit.
        self.assertFalse(np.allclose(first.mean_, second.mean_))
        before = first.mean_.copy()
        first.predict_proba(changed)
        np.testing.assert_array_equal(first.mean_, before)

    def test_seed_clone_serialization_and_end_to_end_gradients(self):
        X, y = self.data()
        model = self.model(val_fraction=0).fit(X, y)
        other = clone(model).fit(X, y)
        np.testing.assert_array_equal(model.predict_proba(X), other.predict_proba(X))
        gradients = dict(model.net_.named_parameters())
        for key in ["branches.0.0.weight", "encoder.layers.0.self_attn.in_proj_weight",
                    "head.3.weight"]:
            self.assertGreater(float(gradients[key].grad.abs().sum()), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(model.predict_proba(X), restored.predict_proba(X))

    def test_depth_order_affects_predictions_and_intensity_ablation(self):
        X, y = self.data()
        model = self.model(val_fraction=0, use_intensity=False).fit(X, y)
        altered = X.copy()
        altered[..., -3:] += 100
        np.testing.assert_array_equal(model.predict_proba(X), model.predict_proba(altered))
        self.assertFalse(np.allclose(model.predict_proba(X), model.predict_proba(X[:, ::-1])))

    def test_invalid_inputs_fail_clearly(self):
        X, y = self.data()
        with self.assertRaises(ValueError):
            self.model().fit(X[:, :-1], y)
        X[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.model().fit(X, y)
        with self.assertRaises(ValueError):
            depth_windows(late_bin=0)

    def test_production_shape_and_parameter_budget(self):
        model = DepthTransformerClassifier()
        model.classes_ = np.arange(1, 6)
        net = model._net().eval()
        with torch.no_grad():
            output = net(torch.zeros(2, 65, 3072))
        self.assertEqual(output.shape, (2, 5))
        self.assertLess(sum(p.numel() for p in net.parameters()), 150000)

    def test_model_can_learn_a_simple_depth_signal(self):
        rng = np.random.default_rng(9)
        X = rng.normal(0, 0.1, (20, 8, 51)).astype(np.float32)
        y = np.tile([1, 5], 10)
        X[y == 5, :4, :16] += 2
        model = self.model(epochs=40, val_fraction=0, lr=0.003).fit(X, y)
        self.assertGreaterEqual(float(np.mean(model.predict(X) == y)), 0.95)

    def test_benchmark_and_recipe_submission_round_trip(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location(
                "depth_cli", scripts / "15_depth_transformer.py")
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cache" / "shots").mkdir(parents=True)
            (root / "raw" / "sample_submission").mkdir(parents=True)
            records = []
            rng = np.random.default_rng(8)
            for split, count in [("train", 20), ("test", 4)]:
                for i in range(count):
                    sid = f"{split}_{i:03d}"
                    shots = rng.uniform(0.1, 2, (12, 96)).astype(np.float32)
                    np.save(root / "cache" / "shots" / f"{sid}.npy", shots)
                    records.append(dict(sample_id=sid, split=split,
                                        label=1 if i % 2 else 5))
            pd.DataFrame(records).to_csv(root / "cache" / "index.csv", index=False)
            filenames = [f"test_{i:03d}.csv" for i in reversed(range(4))]
            pd.DataFrame(dict(filename=filenames, predicted_label=1)).to_csv(
                root / "raw" / "sample_submission" / "predictions.csv", index=False)
            cfg = Config(raw=dict(
                paths=dict(raw_data=str(root / "raw"), cache=str(root / "cache"),
                           results=str(root / "results"), submissions=str(root / "submissions")),
                data=dict(n_shots=12, channel_bounds=[0, 32, 64, 96]),
                preprocessing=dict(repair_outlier_shots=False),
                cv=dict(n_splits=2, n_repeats=1, random_state=42),
                depth_transformer=dict(bin_factor=2, surface_shots=4, late_bin=2, bulk_start=8),
            ))
            args = argparse.Namespace(folds=2, repeats=1, seeds=[42, 7], epochs=1,
                                      no_intensity=False, tag="test", encoders=["transformer"],
                                      device="cpu")
            cli.benchmark(cfg, args)
            run = root / "results" / "depth_transformer" / "test"
            oof = pd.read_csv(run / "oof.csv")
            self.assertEqual(len(oof), 60)
            np.testing.assert_allclose(oof[["p1", "p5"]].sum(axis=1), 1, atol=1e-6)
            means = oof[oof.seed != "ensemble"].groupby("sample_id")[["p1", "p5"]].mean()
            ensemble = oof[oof.seed == "ensemble"].set_index("sample_id")[["p1", "p5"]]
            np.testing.assert_allclose(means.sort_index(), ensemble.sort_index())
            recipe_path = run / "recipe_transformer.json"
            recipe = json.loads(recipe_path.read_text())
            self.assertEqual(recipe["seeds"], [42, 7])
            out = root / "predictions.csv"
            cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None,
                                           device="cpu", output=str(out)))
            self.assertEqual(pd.read_csv(out).filename.tolist(), filenames)
            self.assertTrue(out.with_suffix(".joblib").exists())
            # The training fingerprint prevents using a recipe with different spectra.
            changed = np.ones((12, 96), np.float32)
            np.save(root / "cache" / "shots" / "train_000.npy", changed)
            with self.assertRaisesRegex(ValueError, "Training data/preprocessing changed"):
                cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None,
                                               device="cpu", output=str(root / "other.csv")))


if __name__ == "__main__":
    unittest.main()
