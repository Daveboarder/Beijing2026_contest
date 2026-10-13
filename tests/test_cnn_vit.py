"""Synthetic CPU tests for the 2-D CNN + vision-transformer depth-image classifier and its CLI."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from libs2026.cnn_vit import (
    HEADS,
    LOSSES,
    CnnVitClassifier,
    CnnVitNet,
    PrototypeClassifier,
    ViTLayer,
    crop_resize,
    depth_images,
    select_bands,
)
from libs2026.config import Config
from libs2026.images import build_images
from libs2026.preprocessing import Preprocessor

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_COLUMNS = [
    "model", "labels",
    "all120_accuracy", "all120_accuracy_std", "all120_balanced_accuracy", "all120_macro_f1",
    "unchanged110_accuracy", "unchanged110_accuracy_std", "unchanged110_balanced_accuracy",
    "unchanged110_macro_f1", "n_params", "fit_seconds_mean",
]
TINY_NET = dict(channels=(4, 8), d_model=16, pool=(2, 3), n_layers=1, n_heads=2, ff_dim=32,
                stride=4)


class CnnVitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self, noise=1.0, seed=5):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, noise, (20, 6, 48)).astype(np.float32)
        y = np.tile([1, 5], 10)
        X[y == 5, :3, :16] += 2
        return X, y

    def model(self, **kwargs):
        # (6, 48) images: the two stride-4 convolutions leave a (6, 3) feature map that the
        # adaptive pool turns into a 2 x 3 token grid (6 tokens + class token). Evaluations
        # happen at episodes 2 and 4 only, so best_episodes_ is 2 or 4.
        params = dict(**TINY_NET, episodes=4, eval_every=2, patience=1, n_support=2, n_query=2,
                      crop_min=1.0, noise_std=0.0, dropout=0.0, device="cpu")
        return CnnVitClassifier(**{**params, **kwargs})

    def net(self, **kwargs):
        model = self.model(**kwargs)
        model.classes_ = np.array([1, 5])
        return model._net()

    def test_depth_images_drops_shots_and_averages_groups(self):
        X = np.random.default_rng(0).normal(size=(3, 12, 5)).astype(np.float32)
        out = depth_images(X, drop_shots=4, shot_bin=2)
        self.assertEqual(out.shape, (3, 4, 5))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, X[:, 4:].reshape(3, 4, 2, 5).mean(axis=2), atol=1e-6)
        np.testing.assert_allclose(out[:, 0], (X[:, 4] + X[:, 5]) / 2, atol=1e-6)
        # An incomplete trailing group is dropped; shot_bin=1 only removes the first rows.
        self.assertEqual(depth_images(X, drop_shots=4, shot_bin=3).shape, (3, 2, 5))
        np.testing.assert_array_equal(depth_images(X, drop_shots=2, shot_bin=1), X[:, 2:])
        for kwargs in (dict(drop_shots=-1), dict(shot_bin=0), dict(drop_shots=11, shot_bin=2),
                       dict(drop_shots=12, shot_bin=1)):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                depth_images(X, **kwargs)
        with self.assertRaises(ValueError):
            depth_images(X[0], drop_shots=0, shot_bin=1)

    def test_select_bands_prefers_discriminative_columns(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(10, 12))
        y = np.repeat([1, 5], 5)
        X[y == 5, 7] += 50
        bands = select_bands(X, y, 3)
        self.assertEqual(bands.shape, (3,))
        self.assertIn(7, bands)
        np.testing.assert_array_equal(bands, np.sort(bands))
        self.assertEqual(len(np.unique(bands)), 3)
        # Depth-resolved images are averaged over rows before the class means are compared.
        X3 = np.repeat(X[:, None, :], 4, axis=1) + rng.normal(0, 0.01, (10, 4, 12))
        self.assertIn(7, select_bands(X3, y, 3))
        # Three classes with one marker column per pair: both markers are counted most often.
        y3 = np.repeat([0, 1, 2], 5)
        X3c = rng.normal(size=(15, 12))
        X3c[y3 == 1, 2] += 50
        X3c[y3 == 2, 9] += 50
        np.testing.assert_array_equal(select_bands(X3c, y3, 2, top_per_pair=1), [2, 9])
        for n_bands in (12, 20, 0):
            np.testing.assert_array_equal(select_bands(X, y, n_bands), np.arange(12))
        with self.assertRaises(ValueError):
            select_bands(X, np.ones(10), 3)

    def test_crop_resize_semantics(self):
        rng = np.random.default_rng(2)
        x = torch.randn(4, 1, 6, 48)
        self.assertIs(crop_resize(x, 1.0, rng), x)
        out = crop_resize(x, 0.5, rng)
        self.assertEqual(out.shape, (4, 1, 6, 48))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertFalse(torch.equal(out, x))
        # Resizing a crop of a constant image gives back the constant.
        flat = torch.full((2, 1, 6, 48), 3.0)
        np.testing.assert_allclose(crop_resize(flat, 0.5, rng).numpy(), 3.0, atol=1e-5)

    def test_network_outputs_attention_and_gradients(self):
        torch.manual_seed(0)
        layer = ViTLayer(16, 2, 32, 0.0).eval()
        out, weights = layer(torch.randn(2, 7, 16))
        self.assertEqual(out.shape, (2, 7, 16))
        self.assertEqual(weights.shape, (2, 7, 7))
        np.testing.assert_allclose(weights.sum(-1).detach(), 1, atol=1e-5)

        X, _ = self.data()
        x = torch.from_numpy(X[:4])[:, None]
        target = torch.tensor([0, 1, 0, 1])
        net = self.net().train()
        full = []
        net.layers[-1].attention.register_forward_hook(lambda m, i, o: full.append(o[1]))
        out = net(x)
        self.assertEqual(set(out), {"embedding", "logits", "tokens", "attention"})
        self.assertEqual(out["embedding"].shape, (4, 16))
        self.assertEqual(out["logits"].shape, (4, 2))
        self.assertEqual(out["tokens"].shape, (4, 6, 16))
        self.assertEqual(out["attention"].shape, (4, 2, 3))
        attention = out["attention"].detach()
        self.assertTrue(bool((attention >= 0).all()))
        # The map is the class token's attention over the grid; the class token's own weight
        # is left out, so every grid sum falls short of one by exactly that weight.
        grid_sum = attention.flatten(1).sum(1)
        self.assertTrue(bool((grid_sum > 0).all()))
        self.assertTrue(bool((grid_sum <= 1 + 1e-5).all()))
        np.testing.assert_allclose(grid_sum + full[-1][:, 0, 0].detach(), 1, atol=1e-5)
        np.testing.assert_allclose(full[-1][:, 0, 1:].reshape(4, 2, 3).detach(), attention,
                                   atol=1e-6)
        F.cross_entropy(out["logits"], target).backward()
        params = dict(net.named_parameters())
        for key in ["stem.0.weight", "block.0.weight", "project.0.weight", "cls", "positions",
                    "layers.0.attention.in_proj_weight", "head.weight"]:
            grad = params[key].grad
            self.assertIsNotNone(grad, key)
            self.assertGreater(float(grad.abs().sum()), 0, key)
        for bad in (torch.zeros(4, 6, 48), torch.zeros(4, 2, 6, 48)):
            with self.assertRaisesRegex(ValueError, "Expected input"):
                net(bad)
        for kwargs in (dict(n_heads=3), dict(channels=(4, 7)), dict(stride=0)):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                CnnVitNet(2, **{**TINY_NET, **kwargs})

    def test_prototype_classifier(self):
        X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])
        y = np.array([5, 5, 1, 1])
        head = PrototypeClassifier().fit(X, y)
        np.testing.assert_array_equal(head.classes_, [1, 5])
        np.testing.assert_allclose(head.prototypes_, [[5.0, 5.5], [0.0, 0.5]])
        np.testing.assert_array_equal(head.predict(X), y)
        self.assertEqual(head.decision_function(X).shape, (4, 2))
        p = head.predict_proba(np.array([[0.0, 0.0], [5.0, 5.0], [2.5, 3.0]]))
        self.assertEqual(p.shape, (3, 2))
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-12)
        self.assertGreater(p[0, 1], 0.99)
        self.assertGreater(p[1, 0], 0.99)
        np.testing.assert_allclose(p[2], [0.5, 0.5], atol=1e-12)   # equidistant point
        np.testing.assert_allclose(PrototypeClassifier(scale=0.0).fit(X, y).predict_proba(X), 0.5)

    def test_every_loss_and_head_fits_and_predicts(self):
        X, y = self.data()
        head_types = dict(svm=SVC, mlp=MLPClassifier, proto=PrototypeClassifier)
        settings = [dict(loss=loss, head=head) for loss in LOSSES for head in HEADS]
        # Query mixing needs more than one query per episode: 2 classes x n_query=2 -> 4.
        settings.append(dict(loss="proto", head="svm", mix_queries=True))
        for kwargs in settings:
            with self.subTest(**kwargs):
                model = self.model(**kwargs).fit(X, y)
                self.assertIsInstance(model.head_, head_types[kwargs["head"]])
                p = model.predict_proba(X)
                self.assertEqual(p.shape, (20, 2))
                np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
                self.assertTrue(np.isfinite(p).all())
                np.testing.assert_array_equal(model.classes_, [1, 5])
                self.assertEqual(model.image_shape_, (6, 48))
                self.assertEqual(model.n_features_in_, 6 * 48)
                self.assertTrue(set(model.predict(X)).issubset({1, 5}))
                self.assertIn(model.best_episodes_, (2, 4))
                embedding = model.transform(X)
                self.assertEqual(embedding.shape, (20, 16))
                self.assertTrue(np.isfinite(embedding).all())
                self.assertEqual(model.train_embedding_.shape, (20, 16))
                np.testing.assert_array_equal(model.train_labels_, (y == 5).astype(int))
                self.assertEqual(model.attention_maps(X).shape, (20, 2, 3))
                self.assertEqual(model.transform(X[:0]).shape, (0, 16))
                self.assertEqual(model.predict_proba(X[:0]).shape, (0, 2))
                self.assertEqual(model.attention_maps(X[:0]).shape, (0, 2, 3))
                other = model.refit_head("mlp")
                self.assertIs(other.net_, model.net_)
                self.assertEqual(other.head, "mlp")
                self.assertEqual(model.head, kwargs["head"])
                self.assertIsInstance(other.head_, MLPClassifier)
                q = other.predict_proba(X)
                self.assertEqual(q.shape, (20, 2))
                np.testing.assert_allclose(q.sum(1), 1, atol=1e-6)
                np.testing.assert_array_equal(other.transform(X), embedding)

    def test_inner_validation_does_not_influence_selection_statistics(self):
        X, y = self.data()
        first = self.model().fit(X, y)
        tr, va = first.inner_train_indices_, first.inner_val_indices_
        self.assertEqual(len(va), 4)
        np.testing.assert_array_equal(np.sort(np.concatenate([tr, va])), np.arange(20))
        changed = X.copy()
        changed[va] += 100
        second = self.model().fit(changed, y)
        np.testing.assert_array_equal(second.inner_val_indices_, va)
        np.testing.assert_array_equal(first.selection_bands_, second.selection_bands_)
        np.testing.assert_array_equal(first.selection_mean_, second.selection_mean_)
        np.testing.assert_array_equal(first.selection_std_, second.selection_std_)
        bands, mean, std = first._fit_preprocessing(X[tr], y[tr])
        np.testing.assert_array_equal(bands, np.arange(48))
        self.assertEqual(mean.shape, (48,))
        np.testing.assert_array_equal(first.selection_bands_, bands)
        np.testing.assert_array_equal(first.selection_mean_, mean)
        np.testing.assert_array_equal(first.selection_std_, std)
        # The final refit deliberately standardises with every sample passed to fit.
        self.assertFalse(np.allclose(first.mean_, second.mean_))
        before = first.mean_.copy()
        first.predict_proba(changed)
        np.testing.assert_array_equal(first.mean_, before)

    def test_seed_clone_determinism_and_joblib_round_trip(self):
        X, y = self.data()
        model = self.model(val_fraction=0).fit(X, y)
        self.assertEqual(model.best_episodes_, 4)
        self.assertFalse(hasattr(model, "inner_val_indices_"))
        other = clone(model).fit(X, y)
        np.testing.assert_array_equal(model.predict_proba(X), other.predict_proba(X))
        np.testing.assert_array_equal(model.transform(X), other.transform(X))
        reseeded = self.model(val_fraction=0, random_state=7).fit(X, y)
        self.assertFalse(np.allclose(model.transform(X), reseeded.transform(X)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(model.predict_proba(X), restored.predict_proba(X))
            np.testing.assert_array_equal(model.transform(X), restored.transform(X))
            np.testing.assert_array_equal(model.attention_maps(X), restored.attention_maps(X))

    def test_band_selection_inside_fit(self):
        X, y = self.data()
        model = self.model(n_bands=8).fit(X, y)
        for bands in (model.selection_bands_, model.bands_):
            self.assertEqual(bands.shape, (8,))
            np.testing.assert_array_equal(bands, np.sort(bands))
            self.assertEqual(len(np.unique(bands)), 8)
            self.assertTrue(bool((bands < 16).all()))      # the planted columns
        self.assertEqual(model.mean_.shape, (8,))
        self.assertEqual(model.image_shape_, (6, 48))
        self.assertEqual(model.transform(X).shape, (20, 16))
        self.assertEqual(model.predict_proba(X).shape, (20, 2))
        self.assertEqual(model.attention_maps(X).shape, (20, 2, 3))
        with self.assertRaises(ValueError):
            self.model(n_bands=-1).fit(X, y)

    def test_invalid_inputs_fail_clearly(self):
        X, y = self.data()
        model = self.model(val_fraction=0).fit(X, y)
        with self.assertRaises(ValueError):
            model.predict(X[:, :-1])
        with self.assertRaises(ValueError):
            model.predict_proba(X[:, :, :-1])
        with self.assertRaises(ValueError):
            model.transform(X[:, 0])
        with self.assertRaises(ValueError):
            model.refit_head("x")
        with self.assertRaises(ValueError):
            self.model().refit_head("mlp")           # not fitted
        for kwargs in (dict(loss="x"), dict(head="x"), dict(crop_min=0.0), dict(crop_min=1.5),
                       dict(episodes=0), dict(eval_every=0), dict(patience=0),
                       dict(n_support=0), dict(n_query=0), dict(noise_std=-1.0),
                       dict(mix_weight=-1.0), dict(val_fraction=1.0), dict(n_heads=3)):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.model(**kwargs).fit(X, y)
        with self.assertRaises(ValueError):
            self.model().fit(X[:, 0], y)
        with self.assertRaises(ValueError):
            self.model().fit(X, y[:-1])
        with self.assertRaises(ValueError):
            self.model().fit(X, np.ones(20))
        with self.assertRaises(ValueError):
            self.model().fit(X[:0], y[:0])
        X[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.model().fit(X, y)

    def test_production_shape_and_parameter_budget(self):
        model = CnnVitClassifier()
        model.classes_ = np.arange(1, 6)
        net = model._net().eval()
        self.assertEqual(net.n_tokens, 216)
        with torch.no_grad():
            out = net(torch.zeros(2, 1, 45, 1533))
        self.assertEqual(out["embedding"].shape, (2, 64))
        self.assertEqual(out["logits"].shape, (2, 5))
        self.assertEqual(out["tokens"].shape, (2, 216, 64))
        self.assertEqual(out["attention"].shape, (2, 9, 24))
        for key, value in out.items():
            self.assertTrue(bool(torch.isfinite(value).all()), key)
        count = sum(p.numel() for p in net.parameters())
        print(f"cnn_vit parameters: {count}")
        self.assertLess(count, 400000)

    def test_model_can_learn_a_planted_signal(self):
        X, y = self.data(noise=0.1, seed=9)
        model = self.model(episodes=60, eval_every=10, val_fraction=0, lr=0.003, head="svm")
        model.fit(X, y)
        self.assertEqual(model.best_episodes_, 60)
        self.assertGreaterEqual(float(np.mean(model.predict(X) == y)), 0.9)

    def test_benchmark_and_recipe_submission_round_trip(self):
        scripts = ROOT / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("cnn_vit_cli", scripts / "34_cnn_vit.py")
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
        finally:
            sys.path.pop(0)
        self.assertEqual(tuple(cli.VARIANTS), ("proto", "proto_mix", "ce"))
        self.assertTrue(callable(cli.main))
        scores = cli.metrics(np.array([1, 5, 5]), np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
                             np.array([1, 5]))
        self.assertAlmostEqual(scores["accuracy"], 2 / 3)
        self.assertAlmostEqual(scores["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(scores["macro_f1"], 2 / 3)

        corrections = {"train_000": (5, 1)}
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
                    records.append(dict(sample_id=sid, split=split, label=1 if i % 2 else 5))
            pd.DataFrame(records).to_csv(root / "cache" / "index.csv", index=False)
            np.save(root / "cache" / "wavelength.npy",
                    np.linspace(200, 300, 96, dtype=np.float32))
            filenames = [f"test_{i:03d}.csv" for i in reversed(range(4))]
            pd.DataFrame(dict(filename=filenames, predicted_label=1)).to_csv(
                root / "raw" / "sample_submission" / "predictions.csv", index=False)
            cfg = Config(raw=dict(
                paths=dict(raw_data=str(root / "raw"), cache=str(root / "cache"),
                           results=str(root / "results"), submissions=str(root / "submissions")),
                data=dict(n_shots=12, n_wavelengths=96, channel_bounds=[0, 32, 64, 96]),
                preprocessing=dict(repair_outlier_shots=False),
                cv=dict(n_splits=2, n_repeats=1, random_state=42),
                cnn_vit=dict(bin_factor=2, drop_shots=4, shot_bin=2, channels=[4, 8], d_model=16,
                             pool=[2, 3], n_layers=1, n_heads=2, ff_dim=32, stride=4, episodes=4,
                             eval_every=2, patience=1, n_support=2, n_query=2, crop_min=1.0,
                             dropout=0.0),
            ))
            args = argparse.Namespace(device="cpu", variants=["proto"], heads=["svm", "proto"],
                                      labels="both", seeds=[42, 7], folds=2, repeats=1,
                                      episodes=4, smoke=False, tag="test")
            cli.benchmark(cfg, args, corrections=corrections)
            run = root / "results" / "cnn_vit" / "test"
            train_ids = [f"train_{i:03d}" for i in range(20)]

            folds = pd.read_csv(run / "folds.csv")
            self.assertEqual(len(folds), 20)
            self.assertEqual(set(folds.columns), {"sample_id", "repeat", "fold"})
            self.assertEqual(sorted(folds["sample_id"]), train_ids)
            self.assertEqual(set(folds["repeat"]), {0})
            self.assertEqual(set(folds["fold"]), {0, 1})

            # One network per fold and seed feeds both heads, so n_params is the same
            # everywhere and equals the module's count for the configured architecture.
            summary = pd.read_csv(run / "summary.csv")
            self.assertEqual(len(summary), 4)
            self.assertEqual(list(summary.columns), SUMMARY_COLUMNS)
            self.assertEqual(set(zip(summary["model"], summary["labels"])),
                             {(m, s) for m in ("proto+svm", "proto+proto")
                              for s in ("original", "corrected")})
            expected_params = sum(p.numel() for p in CnnVitNet(2, dropout=0.0,
                                                               **TINY_NET).parameters())
            self.assertTrue((summary["n_params"] == expected_params).all())
            self.assertTrue((summary["fit_seconds_mean"] > 0).all())

            # 2 heads x 2 label sets x 1 repeat x (2 seeds + ensemble) x 20 samples.
            oof = pd.read_csv(run / "oof.csv")
            self.assertEqual(len(oof), 240)
            self.assertEqual(set(oof.columns), {"model", "labels", "repeat", "seed", "sample_id",
                                                "y_true", "y_given", "p1", "p5"})
            self.assertEqual(set(oof["model"]), {"proto+svm", "proto+proto"})
            self.assertEqual(set(oof["labels"]), {"original", "corrected"})
            self.assertEqual(set(oof["seed"].astype(str)), {"42", "7", "ensemble"})
            np.testing.assert_allclose(oof[["p1", "p5"]].sum(axis=1), 1, atol=1e-6)
            keys = ["model", "labels", "sample_id"]
            members = oof[oof["seed"].astype(str) != "ensemble"]
            ensemble = oof[oof["seed"].astype(str) == "ensemble"]
            self.assertEqual(len(ensemble), 80)
            means = members.groupby(keys)[["p1", "p5"]].mean().sort_index()
            np.testing.assert_allclose(means, ensemble.set_index(keys)[["p1", "p5"]].sort_index(),
                                       atol=1e-9)
            first = oof[oof["sample_id"] == "train_000"]
            self.assertEqual(len(first), 12)
            self.assertTrue((first["y_given"] == 5).all())
            self.assertTrue((first.loc[first["labels"] == "corrected", "y_true"] == 1).all())
            self.assertTrue((first.loc[first["labels"] == "original", "y_true"] == 5).all())
            rest = oof[oof["sample_id"] != "train_000"]
            self.assertTrue((rest["y_true"] == rest["y_given"]).all())
            # Summary metrics are those of the seed ensemble; unchanged110 drops train_000.
            for _, row in summary.iterrows():
                rows = ensemble[(ensemble["model"] == row["model"])
                                & (ensemble["labels"] == row["labels"])]
                pred = np.where(rows["p1"].to_numpy() >= rows["p5"].to_numpy(), 1, 5)
                correct = pred == rows["y_true"].to_numpy()
                kept = rows["sample_id"].to_numpy() != "train_000"
                self.assertAlmostEqual(row["all120_accuracy"], correct.mean(), places=6)
                self.assertAlmostEqual(row["unchanged110_accuracy"], correct[kept].mean(),
                                       places=6)
                self.assertAlmostEqual(row["all120_accuracy_std"], 0.0, places=9)

            recipe_path = run / "recipe_proto_corrected.json"
            recipe = json.loads(recipe_path.read_text())
            self.assertEqual(recipe["labels"], "corrected")
            self.assertEqual(recipe["heads"], ["svm", "proto"])
            self.assertEqual(recipe["seeds"], [42, 7])
            self.assertEqual(recipe["preparation"], dict(bin_factor=2, drop_shots=4, shot_bin=2))
            self.assertEqual(recipe["training_ids"], train_ids)
            self.assertEqual(recipe["training_labels"][0], 1)
            self.assertEqual(recipe["training_labels"][1:],
                             [1 if i % 2 else 5 for i in range(1, 20)])
            self.assertEqual(recipe["corrections"], [["train_000", 5, 1]])
            self.assertEqual(recipe["classes"], [1, 5])
            self.assertEqual(set(recipe["episodes_by_seed"]), {"42", "7"})
            for value in recipe["episodes_by_seed"].values():
                self.assertIn(value, range(1, 5))
            self.assertEqual(len(recipe["episode_selection"]), 4)   # 2 seeds x 2 folds
            for entry in recipe["episode_selection"]:
                self.assertEqual(set(entry), {"repeat", "seed", "fold", "episodes"})
                self.assertIn(entry["episodes"], (2, 4))
            self.assertEqual(len(recipe["repeat_metrics"]), 1)
            self.assertEqual(set(recipe["repeat_metrics"][0]), {"svm", "proto"})
            self.assertEqual(set(recipe["repeat_metrics"][0]["svm"]), {"all120", "unchanged110"})
            params = recipe["model_params"]
            self.assertEqual(params["loss"], "proto")
            self.assertIs(params["mix_queries"], False)
            self.assertEqual(params["episodes"], 4)
            self.assertEqual(params["channels"], [4, 8])
            self.assertEqual(params["pool"], [2, 3])
            for key in ("head", "device", "random_state"):
                self.assertNotIn(key, params)
            self.assertEqual(recipe["cv"], dict(n_splits=2, n_repeats=1, random_state=42))
            self.assertEqual(recipe["n_params"], expected_params)
            self.assertEqual(recipe["device"], "cpu")
            self.assertIn("torch", recipe["environment"])
            self.assertEqual(recipe["config"]["cnn_vit"]["drop_shots"], 4)
            source = ROOT / "src" / "libs2026"
            expected_hash = hashlib.sha256((source / "cnn_vit.py").read_bytes()
                                           + (source / "images.py").read_bytes()).hexdigest()
            self.assertEqual(recipe["source_sha256"], expected_hash)
            # The input fingerprint covers the prepared training images and original labels.
            images = build_images(cfg, Preprocessor.from_config(cfg), bin_factor=2, shot_bin=1)
            train = images.subset("train")
            self.assertEqual(train.X.shape, (20, 12, 48))
            prepared = depth_images(train.X, 4, 2)
            self.assertEqual(prepared.shape, (20, 4, 48))
            y_orig = train.y.astype(int)
            expected_input = hashlib.sha256(prepared.tobytes() + y_orig.tobytes()).hexdigest()
            self.assertEqual(recipe["input_sha256"], expected_input)
            original = json.loads((run / "recipe_proto_original.json").read_text())
            self.assertEqual(original["labels"], "original")
            self.assertEqual(original["corrections"], [])
            self.assertEqual(original["training_labels"][0], 5)
            self.assertEqual(original["heads"], ["svm", "proto"])
            self.assertEqual(original["model_params"]["loss"], "proto")

            out = root / "predictions.csv"
            cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None, device="cpu",
                                           output=str(out), head="svm"), corrections=corrections)
            submission = pd.read_csv(out)
            self.assertEqual(list(submission.columns), ["filename", "predicted_label"])
            self.assertEqual(submission["filename"].tolist(), filenames)
            self.assertTrue(submission["predicted_label"].isin([1, 5]).all())
            checkpoint = out.with_suffix(".joblib")
            self.assertTrue(checkpoint.exists())
            saved = joblib.load(checkpoint)
            self.assertEqual(set(saved), {"models", "recipe", "probabilities", "sample_ids"})
            self.assertEqual(len(saved["models"]), 2)
            self.assertEqual(saved["probabilities"].shape, (4, 2))
            np.testing.assert_allclose(saved["probabilities"].sum(1), 1, atol=1e-6)
            self.assertEqual(list(saved["sample_ids"]), [f"test_{i:03d}" for i in range(4)])
            self.assertEqual(saved["recipe"]["labels"], "corrected")
            for model, seed in zip(saved["models"], [42, 7]):
                self.assertEqual(model.head, "svm")
                self.assertEqual(model.random_state, seed)
                self.assertEqual(model.val_fraction, 0)
                self.assertEqual(model.loss, "proto")
                self.assertEqual(model.best_episodes_, recipe["episodes_by_seed"][str(seed)])
                np.testing.assert_array_equal(model.classes_, [1, 5])
            # The CSV carries the argmax of the saved seed average, in template order.
            labels = np.array([1, 5])[saved["probabilities"].argmax(1)]
            predicted = dict(zip(saved["sample_ids"], labels))
            self.assertEqual(submission["predicted_label"].tolist(),
                             [int(predicted[name[:-4]]) for name in filenames])
            with self.assertRaises((FileExistsError, ValueError)):
                cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None,
                                               device="cpu", output=str(out), head="svm"),
                            corrections=corrections)
            # A head that was not benchmarked with this recipe is refused.
            with self.assertRaises(ValueError):
                cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None,
                                               device="cpu", output=str(root / "mlp.csv"),
                                               head="mlp"), corrections=corrections)
            self.assertFalse((root / "mlp.csv").exists())
            # An original-label recipe needs no corrections at all.
            cli.predict(argparse.Namespace(recipe=str(run / "recipe_proto_original.json"),
                                           config=None, device="cpu",
                                           output=str(root / "original.csv"), head="proto"))
            self.assertEqual(pd.read_csv(root / "original.csv")["filename"].tolist(), filenames)
            self.assertEqual(joblib.load(root / "original.joblib")["models"][0].head, "proto")
            # Corrections that differ from the benchmarked label set are refused.
            with self.assertRaises(ValueError):
                cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None,
                                               device="cpu", output=str(root / "mismatch.csv"),
                                               head="svm"), corrections={"train_000": (5, 2)})
            self.assertFalse((root / "mismatch.csv").exists())
            # The training fingerprint prevents using a recipe with different spectra.
            # build_images caches the preprocessed stack under cache/images keyed on the
            # preprocessing alone, so that cache is cleared for the raw change to surface.
            np.save(root / "cache" / "shots" / "train_000.npy", np.ones((12, 96), np.float32))
            shutil.rmtree(root / "cache" / "images", ignore_errors=True)
            with self.assertRaisesRegex(ValueError, "Training data/preprocessing changed"):
                cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None,
                                               device="cpu", output=str(root / "other.csv"),
                                               head="svm"), corrections=corrections)
            self.assertFalse((root / "other.csv").exists())


if __name__ == "__main__":
    unittest.main()
