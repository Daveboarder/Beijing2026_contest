"""Synthetic CPU tests for the bidirectional Mamba dual-pathway classifier and its CLI."""

import argparse
import hashlib
import importlib.util
import json
import math
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

from libs2026.bimamba import (
    S6,
    VARIANTS,
    BiMambaBlock,
    BiMambaClassifier,
    SpectralPath,
    augment_batch,
    depth_geometry,
    selective_scan,
    selective_scan_reference,
    sinusoidal_positions,
)
from libs2026.config import Config

ROOT = Path(__file__).resolve().parents[1]


class BiMambaTests(unittest.TestCase):
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
        # 12 shots, 4 surface shots and pairs afterwards -> 8 depth tokens; bulk starts at
        # window (8, 10), so the three physical regions hold 4, 2 and 2 tokens.
        params = dict(channel_widths=(16, 16, 16), n_shots=12, surface_shots=4, late_bin=2,
                      bulk_start=8, d_model=16, d_state=4, patch=4, chunk=4, n_queries=2,
                      stem_bins=2, epochs=2, patience=1, batch_size=4, dropout=0,
                      noise_std=0, token_dropout=0, device="cpu")
        return BiMambaClassifier(**{**params, **kwargs})

    def net(self, **kwargs):
        model = self.model(**kwargs)
        model.classes_ = np.array([1, 5])
        return model._net()

    def test_chunked_scan_matches_reference_values_and_gradients(self):
        # Hand-traced: constant decay 1/2 and unit inputs give h = 1, 1.5, 1.75 (L=3, chunk=2
        # pads one step whose zero input and unit decay must not change the kept outputs).
        u = torch.ones(1, 3, 1, 1, dtype=torch.float64)
        log_decay = torch.full_like(u, math.log(0.5))
        c = torch.ones(1, 3, 1, dtype=torch.float64)
        np.testing.assert_allclose(selective_scan(u, log_decay, c, chunk=2)[0, :, 0],
                                   [1, 1.5, 1.75])
        torch.manual_seed(0)
        batch, length, d_inner, d_state = 2, 13, 3, 4
        u0 = torch.randn(batch, length, d_inner, d_state, dtype=torch.float64)
        log0 = -2 * torch.rand(batch, length, d_inner, d_state, dtype=torch.float64)
        c = torch.randn(batch, length, d_state, dtype=torch.float64)
        weight = torch.randn(batch, length, d_inner, dtype=torch.float64)
        u_ref = u0.clone().requires_grad_(True)
        log_ref = log0.clone().requires_grad_(True)
        y_ref = selective_scan_reference(u_ref, torch.exp(log_ref), c)
        self.assertEqual(y_ref.shape, (batch, length, d_inner))
        (y_ref * weight).sum().backward()
        for chunk in (1, 4, 5, 16):
            with self.subTest(chunk=chunk):
                u_chunk = u0.clone().requires_grad_(True)
                log_chunk = log0.clone().requires_grad_(True)
                y = selective_scan(u_chunk, log_chunk, c, chunk=chunk)
                self.assertEqual(y.shape, (batch, length, d_inner))
                np.testing.assert_allclose(y.detach(), y_ref.detach(), atol=1e-9)
                (y * weight).sum().backward()
                np.testing.assert_allclose(u_chunk.grad, u_ref.grad, atol=1e-9)
                np.testing.assert_allclose(log_chunk.grad, log_ref.grad, atol=1e-9)

    def test_s6_is_causal(self):
        torch.manual_seed(1)
        ssm = S6(8, d_state=4, dt_rank=1, chunk=4).eval()
        x = torch.randn(2, 10, 8)
        altered = x.clone()
        altered[:, 6:] += 1.0
        with torch.no_grad():
            before, after = ssm(x), ssm(altered)
        self.assertEqual(before.shape, (2, 10, 8))
        np.testing.assert_allclose(before[:, :6].numpy(), after[:, :6].numpy(), atol=1e-6)
        self.assertFalse(torch.allclose(before[:, 6:], after[:, 6:]))

    def test_block_is_flip_equivariant_only_with_tied_directions(self):
        torch.manual_seed(2)
        block = BiMambaBlock(16, d_state=4, dropout=0, chunk=4).eval()
        x = torch.randn(3, 10, 16)
        with torch.no_grad():
            untied = block(x.flip(1)).flip(1)
            self.assertFalse(torch.allclose(untied, block(x), atol=1e-5))
            block.ssm_bwd.load_state_dict(block.ssm_fwd.state_dict())
            tied = block(x.flip(1)).flip(1)
            np.testing.assert_allclose(tied.numpy(), block(x).numpy(), atol=1e-5)

    def test_outputs_stay_finite_for_large_inputs(self):
        X, _ = self.data()
        net = self.net().eval()
        with torch.no_grad():
            out = net.outputs(torch.from_numpy(X) * 100)
        self.assertEqual(out["final"].shape, (20, 2))
        for key, value in out.items():
            self.assertTrue(bool(torch.isfinite(value).all()), key)

    def test_outputs_keys_gate_and_gradients_per_variant(self):
        X, _ = self.data()
        x = torch.from_numpy(X[:4])
        target = torch.tensor([0, 1, 0, 1])
        expected = dict(full={"spectral", "depth", "fused", "gate", "embedding", "final"},
                        spectral={"spectral", "embedding", "final"},
                        depth={"depth", "embedding", "final"},
                        concat={"spectral", "depth", "fused", "embedding", "final"})
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                net = self.net(variant=variant).train()
                out = net.outputs(x)
                self.assertEqual(set(out), expected[variant])
                self.assertEqual(out["final"].shape, (4, 2))
                self.assertEqual(out["embedding"].shape, (4, 16))
                if variant == "full":
                    self.assertEqual(out["gate"].shape, (4, 2))
                    np.testing.assert_allclose(out["gate"].sum(1).detach(), 1, atol=1e-6)
                F.cross_entropy(out["final"], target).backward()
                params = dict(net.named_parameters())
                keys = []
                if net.spectral is not None:
                    keys += ["spectral.patch_embed.weight", "spectral.channel_embed",
                             "spectral.layers.0.ssm_fwd.A_log", "spectral.layers.0.ssm_bwd.A_log",
                             "spectral.layers.0.ssm_fwd.dt_proj.weight",
                             "spectral.layers.0.ssm_bwd.dt_proj.weight",
                             "spectral.score.weight", "head_spectral.3.weight"]
                if net.depth is not None:
                    keys += ["depth.branches.0.0.weight", "depth.projection.weight",
                             "depth.layers.0.ssm_fwd.A_log", "depth.layers.0.ssm_bwd.A_log",
                             "depth.layers.0.ssm_fwd.dt_proj.weight",
                             "depth.layers.0.ssm_bwd.dt_proj.weight",
                             "depth.summary.weight", "head_depth.3.weight"]
                if variant == "full":
                    keys += ["fusion.queries", "fusion.source", "fusion.gate.weight",
                             "fusion.attention.in_proj_weight", "head_fused.3.weight"]
                if variant == "concat":
                    keys += ["concat.0.weight", "head_fused.3.weight"]
                for key in keys:
                    grad = params[key].grad
                    self.assertIsNotNone(grad, key)
                    self.assertGreater(float(grad.abs().sum()), 0, key)

    def test_every_variant_fits_and_predicts(self):
        X, y = self.data()
        expected = {"full": {"spectral", "depth", "fused", "final", "gate"},
                    "spectral": {"spectral", "final"},
                    "depth": {"depth", "final"},
                    "concat": {"spectral", "depth", "fused", "final"}}
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.model(variant=variant).fit(X, y)
                p = model.predict_proba(X)
                self.assertEqual(p.shape, (20, 2))
                np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
                self.assertTrue(np.isfinite(p).all())
                np.testing.assert_array_equal(model.classes_, [1, 5])
                self.assertTrue(set(model.predict(X)).issubset({1, 5}))
                self.assertGreaterEqual(model.best_epoch_, 1)
                heads = model.predict_heads(X)
                self.assertEqual(set(heads), expected[variant])
                for key, value in heads.items():
                    self.assertEqual(value.shape, (20, 2), key)
                    np.testing.assert_allclose(value.sum(1), 1, atol=1e-6, err_msg=key)
                np.testing.assert_allclose(heads["final"], p, atol=1e-6)
                if variant in ("spectral", "depth"):
                    np.testing.assert_allclose(heads[variant], heads["final"], atol=1e-6)
                embedding = model.transform(X)
                self.assertEqual(embedding.shape, (20, 16))
                self.assertTrue(np.isfinite(embedding).all())
                self.assertEqual(model.transform(X[:0]).shape, (0, 16))

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

    def test_seed_clone_determinism_and_joblib_round_trip(self):
        X, y = self.data()
        model = self.model(val_fraction=0).fit(X, y)
        self.assertEqual(model.best_epoch_, 2)
        other = clone(model).fit(X, y)
        np.testing.assert_array_equal(model.predict_proba(X), other.predict_proba(X))
        reseeded = self.model(val_fraction=0, random_state=7).fit(X, y)
        self.assertFalse(np.allclose(model.predict_proba(X), reseeded.predict_proba(X)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(model.predict_proba(X), restored.predict_proba(X))
            np.testing.assert_array_equal(model.transform(X), restored.transform(X))

    def test_depth_and_wavelength_order_affect_predictions(self):
        X, y = self.data()
        model = self.model(val_fraction=0).fit(X, y)
        p = model.predict_proba(X)
        self.assertFalse(np.allclose(p, model.predict_proba(X[:, ::-1])))
        reversed_bins = X.copy()
        reversed_bins[..., :16] = X[..., 15::-1]
        # Standardisation is one scalar per channel, so a within-channel permutation can
        # only reach the prediction through the wavelength order of the two pathways.
        np.testing.assert_allclose(model._stats(reversed_bins)[0], model._stats(X)[0],
                                   atol=1e-4)
        self.assertFalse(np.allclose(p, model.predict_proba(reversed_bins)))

    def test_augment_batch_semantics(self):
        torch.manual_seed(3)
        x = torch.randn(6, 8, 51)
        self.assertTrue(torch.equal(augment_batch(x, 0.0, 0.0), x))
        shifted = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        self.assertTrue(torch.equal(augment_batch(x, 0.0, 1.0), shifted))
        jittered = augment_batch(x, 0.0, 0.5)
        self.assertTrue(torch.equal(jittered[:, 0], x[:, 0]))
        self.assertFalse(torch.equal(jittered, x))
        kept = (jittered == x).all(-1)
        moved = (jittered == shifted).all(-1)
        self.assertTrue(bool((kept | moved).all()))
        noisy = augment_batch(x, 0.5, 0.0)
        self.assertEqual(noisy.shape, x.shape)
        self.assertAlmostEqual(float((noisy - x).std()), 0.5, delta=0.05)
        single = x[:, :1]
        self.assertTrue(torch.equal(augment_batch(single, 0.0, 0.5), single))

    def test_invalid_inputs_fail_clearly(self):
        X, y = self.data()
        coordinates, weights = depth_geometry(12, 4, 2, 8)
        self.assertEqual(coordinates.shape, (8, 2))
        self.assertEqual(weights.shape, (3, 8))
        np.testing.assert_allclose(weights.sum(1), 1, atol=1e-6)
        np.testing.assert_array_equal((weights > 0).sum(1), [4, 2, 2])
        with self.assertRaises(ValueError):
            self.model().fit(X[:, :-1], y)
        with self.assertRaises(ValueError):
            self.model().fit(X[:, :, :-1], y)
        for kwargs in (dict(patch=5), dict(variant="both"), dict(chunk=0), dict(bulk_start=7),
                       dict(d_model=18), dict(token_dropout=1.0), dict(n_queries=0)):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.model(**kwargs).fit(X, y)
        with self.assertRaises(ValueError):
            SpectralPath((16, 16, 16), 5, 3, 16)
        with self.assertRaises(ValueError):
            depth_geometry(12, 4, 2, 7)       # not a window boundary
        with self.assertRaises(ValueError):
            depth_geometry(12, 4, 2, 4)       # empty intermediate region
        with self.assertRaises(ValueError):
            sinusoidal_positions(4, 15)
        with self.assertRaises(ValueError):
            selective_scan(torch.zeros(1, 3, 1, 1), torch.zeros(1, 3, 1, 1),
                           torch.zeros(1, 3, 1), chunk=0)
        with self.assertRaisesRegex(ValueError, "Expected input"):
            self.net().outputs(torch.zeros(2, 7, 51))
        X[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.model().fit(X, y)

    def test_production_shape_and_parameter_budget(self):
        counts = {}
        for variant in VARIANTS:
            model = BiMambaClassifier(variant=variant)
            model.classes_ = np.arange(1, 6)
            net = model._net().eval()
            counts[variant] = sum(p.numel() for p in net.parameters())
            if variant == "full":
                self.assertEqual(net.spectral.n_tokens, 279)
                with torch.no_grad():
                    out = net.outputs(torch.zeros(2, 65, 3072))
                self.assertEqual(out["final"].shape, (2, 5))
                self.assertEqual(out["embedding"].shape, (2, 64))
                self.assertEqual(out["gate"].shape, (2, 2))
        print(f"bimamba parameters: {counts}")
        self.assertLess(counts["full"], 200000)
        self.assertLess(counts["spectral"], counts["full"])
        self.assertLess(counts["depth"], counts["full"])
        self.assertLess(counts["concat"], counts["full"])

    def test_model_can_learn_a_planted_signal(self):
        rng = np.random.default_rng(9)
        X = rng.normal(0, 0.1, (20, 8, 51)).astype(np.float32)
        y = np.tile([1, 5], 10)
        X[y == 5, :4, :16] += 2
        for variant in ("full", "spectral", "depth"):
            with self.subTest(variant=variant):
                model = self.model(variant=variant, epochs=40, val_fraction=0, lr=0.003)
                model.fit(X, y)
                self.assertGreaterEqual(float(np.mean(model.predict(X) == y)), 0.9)

    def test_benchmark_and_recipe_submission_round_trip(self):
        scripts = ROOT / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("bimamba_cli", scripts / "33_bimamba.py")
            cli = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli)
        finally:
            sys.path.pop(0)
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
            filenames = [f"test_{i:03d}.csv" for i in reversed(range(4))]
            pd.DataFrame(dict(filename=filenames, predicted_label=1)).to_csv(
                root / "raw" / "sample_submission" / "predictions.csv", index=False)
            cfg = Config(raw=dict(
                paths=dict(raw_data=str(root / "raw"), cache=str(root / "cache"),
                           results=str(root / "results"), submissions=str(root / "submissions")),
                data=dict(n_shots=12, channel_bounds=[0, 32, 64, 96]),
                preprocessing=dict(repair_outlier_shots=False),
                cv=dict(n_splits=2, n_repeats=1, random_state=42),
                bimamba=dict(bin_factor=2, surface_shots=4, late_bin=2, bulk_start=8, patch=4,
                             chunk=4, d_model=16, d_state=4, n_queries=2, stem_bins=2,
                             dropout=0.0, noise_std=0.0, token_dropout=0.0),
            ))
            args = argparse.Namespace(device="cpu", variants=["full", "depth"], labels="both",
                                      seeds=[42, 7], folds=2, repeats=1, epochs=1, smoke=False,
                                      tag="test")
            cli.benchmark(cfg, args, corrections=corrections)
            run = root / "results" / "bimamba" / "test"

            folds = pd.read_csv(run / "folds.csv")
            self.assertEqual(len(folds), 20)
            self.assertEqual(set(folds.columns), {"sample_id", "repeat", "fold"})
            self.assertEqual(set(folds["fold"]), {0, 1})

            summary = pd.read_csv(run / "summary.csv")
            self.assertEqual(len(summary), 4)
            for column in ["model", "labels", "all120_accuracy", "all120_accuracy_std",
                           "all120_balanced_accuracy", "all120_macro_f1",
                           "unchanged110_accuracy", "unchanged110_accuracy_std",
                           "unchanged110_balanced_accuracy", "unchanged110_macro_f1",
                           "n_params", "fit_seconds_mean"]:
                self.assertIn(column, summary.columns)
            self.assertEqual(set(zip(summary["model"], summary["labels"])),
                             {(m, s) for m in ("full", "depth") for s in ("original", "corrected")})
            n_params = summary.groupby("model")["n_params"].first()
            self.assertGreater(n_params["full"], n_params["depth"])

            # 2 variants x 2 label sets x 1 repeat x (2 seeds + ensemble) x 20 samples.
            oof = pd.read_csv(run / "oof.csv")
            self.assertEqual(len(oof), 240)
            np.testing.assert_allclose(oof[["p1", "p5"]].sum(axis=1), 1, atol=1e-6)
            self.assertEqual(set(oof["seed"].astype(str)), {"42", "7", "ensemble"})
            keys = ["model", "labels", "sample_id"]
            members = oof[oof["seed"].astype(str) != "ensemble"]
            ensemble = oof[oof["seed"].astype(str) == "ensemble"]
            means = members.groupby(keys)[["p1", "p5"]].mean().sort_index()
            np.testing.assert_allclose(means, ensemble.set_index(keys)[["p1", "p5"]].sort_index(),
                                       atol=1e-9)
            first = oof[oof["sample_id"] == "train_000"]
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

            recipe_path = run / "recipe_full_corrected.json"
            recipe = json.loads(recipe_path.read_text())
            self.assertEqual(recipe["labels"], "corrected")
            self.assertEqual(recipe["seeds"], [42, 7])
            self.assertEqual(recipe["preparation"], dict(bin_factor=2, surface_shots=4, late_bin=2))
            self.assertEqual(recipe["training_ids"][0], "train_000")
            self.assertEqual(recipe["training_labels"][0], 1)
            self.assertEqual(recipe["training_labels"][1:],
                             [1 if i % 2 else 5 for i in range(1, 20)])
            self.assertEqual(recipe["corrections"], [["train_000", 5, 1]])
            self.assertEqual(recipe["classes"], [1, 5])
            self.assertEqual(recipe["epochs_by_seed"], {"42": 1, "7": 1})
            self.assertEqual(len(recipe["epoch_selection"]), 4)   # 2 seeds x 2 folds
            self.assertEqual(len(recipe["repeat_metrics"]), 1)
            self.assertIn("unchanged110", recipe["repeat_metrics"][0])
            self.assertEqual(recipe["model_params"]["variant"], "full")
            self.assertEqual(recipe["model_params"]["channel_widths"], [16, 16, 16])
            self.assertEqual(recipe["model_params"]["epochs"], 1)
            self.assertNotIn("device", recipe["model_params"])
            self.assertNotIn("random_state", recipe["model_params"])
            self.assertEqual(recipe["cv"], dict(n_splits=2, n_repeats=1, random_state=42))
            self.assertGreater(recipe["n_params"], 0)
            source = ROOT / "src" / "libs2026"
            expected_hash = hashlib.sha256(
                (source / "bimamba.py").read_bytes()
                + (source / "depth_transformer.py").read_bytes()).hexdigest()
            self.assertEqual(recipe["source_sha256"], expected_hash)
            original = json.loads((run / "recipe_depth_original.json").read_text())
            self.assertEqual(original["labels"], "original")
            self.assertEqual(original["corrections"], [])
            self.assertEqual(original["training_labels"][0], 5)
            self.assertEqual(original["model_params"]["variant"], "depth")

            out = root / "predictions.csv"
            cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None, device="cpu",
                                           output=str(out)), corrections=corrections)
            submission = pd.read_csv(out)
            self.assertEqual(submission["filename"].tolist(), filenames)
            self.assertTrue(submission["predicted_label"].isin([1, 5]).all())
            checkpoint = out.with_suffix(".joblib")
            self.assertTrue(checkpoint.exists())
            saved = joblib.load(checkpoint)
            self.assertEqual(set(saved), {"models", "recipe", "probabilities", "sample_ids"})
            self.assertEqual(len(saved["models"]), 2)
            self.assertEqual(saved["probabilities"].shape, (4, 2))
            self.assertEqual(saved["sample_ids"], [f"test_{i:03d}" for i in range(4)])
            self.assertEqual(saved["models"][0].best_epoch_, 1)
            with self.assertRaises((FileExistsError, ValueError)):
                cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None,
                                               device="cpu", output=str(out)),
                            corrections=corrections)
            # An original-label recipe needs no corrections at all.
            cli.predict(argparse.Namespace(recipe=str(run / "recipe_depth_original.json"),
                                           config=None, device="cpu",
                                           output=str(root / "original.csv")))
            self.assertEqual(pd.read_csv(root / "original.csv")["filename"].tolist(), filenames)
            # Corrections that differ from the benchmarked label set are refused.
            with self.assertRaises(ValueError):
                cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None,
                                               device="cpu", output=str(root / "mismatch.csv")),
                            corrections={"train_000": (5, 2)})
            self.assertFalse((root / "mismatch.csv").exists())
            # The training fingerprint prevents using a recipe with different spectra.
            np.save(root / "cache" / "shots" / "train_000.npy", np.ones((12, 96), np.float32))
            with self.assertRaisesRegex(ValueError, "Training data/preprocessing changed"):
                cli.predict(argparse.Namespace(recipe=str(recipe_path), config=None,
                                               device="cpu", output=str(root / "other.csv")),
                            corrections=corrections)


if __name__ == "__main__":
    unittest.main()
