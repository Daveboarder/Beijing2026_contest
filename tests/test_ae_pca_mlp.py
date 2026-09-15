"""Unit tests for AE CLS ∥ PCA ∥ MLP fusion (including g4 CLS broadcast)."""

import unittest

import numpy as np
import torch

from libs2026.ae_pca_mlp import (
    AEPCAClassifier,
    remap_row_indices,
    row_sample_indices,
)
from libs2026.autotransformer import AutoDepthClassifier


class AEPCATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def depth_data(self, n=24, depth=8, n_spectral=48, n_intensity=3):
        rng = np.random.default_rng(11)
        feats = n_spectral + n_intensity
        X = rng.normal(size=(n, depth, feats)).astype(np.float32)
        y = np.tile([1, 5], n // 2)
        X[y == 5, :4, :16] += 2.5
        return X, y, n_spectral, feats

    def classical_data(self, n, n_features=40, y=None):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(n, n_features)).astype(np.float64)
        if y is not None:
            X[y == 5, :8] += 1.5
        return X

    def ae_params(self, n_spectral, n_features):
        return dict(
            n_spectral=n_spectral, n_features=n_features,
            n_shots=12, surface_shots=4, late_bin=2,
            d_model=16, n_layers=1, n_heads=2, ff_dim=32,
            epochs=2, patience=1, batch_size=4, dropout=0, val_fraction=0,
        )

    def test_autoencoder_transform_cls_shape(self):
        X, y, n_spectral, feats = self.depth_data()
        model = AutoDepthClassifier(
            **self.ae_params(n_spectral, feats), device="cpu", random_state=0,
        ).fit(X, y)
        cls = model.transform(X)
        self.assertEqual(cls.shape, (len(X), 16))
        self.assertTrue(np.isfinite(cls).all())

    def test_fused_width_and_proba(self):
        Xd, y, n_spectral, feats = self.depth_data()
        Xc = self.classical_data(len(y), n_features=40, y=y)
        model = AEPCAClassifier(
            ae_params=self.ae_params(n_spectral, feats),
            seeds=[0], n_pca=8, mlp_hidden=(32, 16), mlp_max_iter=400,
            random_state=0, device="cpu",
        ).fit(Xd, Xc, y)
        self.assertEqual(model.d_model_, 16)
        self.assertEqual(model.n_pca_, 8)
        fused = model.transform(Xd, Xc)
        self.assertEqual(fused.shape, (len(y), 16 + 8))
        p = model.predict_proba(Xd, Xc)
        self.assertEqual(p.shape, (len(y), 2))
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
        self.assertTrue(set(model.predict(Xd, Xc)).issubset({1, 5}))

    def test_broadcast_cls_onto_g4_rows(self):
        Xd, y, n_spectral, feats = self.depth_data(n=12)
        n_groups = 4
        Xc = self.classical_data(len(y) * n_groups, n_features=40)
        y_rows = np.repeat(y, n_groups)
        Xc[y_rows == 5, :8] += 1.5
        row_idx = np.repeat(np.arange(len(y)), n_groups)
        model = AEPCAClassifier(
            ae_params=self.ae_params(n_spectral, feats),
            seeds=[0], n_pca=6, mlp_hidden=(32,), mlp_max_iter=300,
            random_state=0, device="cpu",
        ).fit(Xd, Xc, y_rows, row_sample_idx=row_idx)
        fused = model.transform(Xd, Xc, row_sample_idx=row_idx)
        self.assertEqual(fused.shape, (len(y) * n_groups, 16 + 6))
        np.testing.assert_allclose(fused[0, :16], fused[1, :16], atol=1e-5)
        p = model.predict_proba(Xd, Xc, row_sample_idx=row_idx)
        self.assertEqual(p.shape, (len(y) * n_groups, 2))
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)

    def test_row_sample_helpers(self):
        depth_ids = np.array(["a", "b", "c"])
        classical_ids = np.array(["a", "a", "b", "b", "c", "c"])
        idx = row_sample_indices(depth_ids, classical_ids)
        np.testing.assert_array_equal(idx, [0, 0, 1, 1, 2, 2])
        keep, local = remap_row_indices(idx, np.array([0, 2]))
        np.testing.assert_array_equal(keep, [0, 1, 4, 5])
        np.testing.assert_array_equal(local, [0, 0, 1, 1])

    def test_pca_ignores_held_out_classical_rows(self):
        Xd, y, n_spectral, feats = self.depth_data(n=30)
        Xc = self.classical_data(len(y), n_features=40, y=y)
        tr = np.arange(0, 24)
        te = np.arange(24, 30)
        first = AEPCAClassifier(
            ae_params=self.ae_params(n_spectral, feats),
            seeds=[1], n_pca=5, mlp_hidden=(16,), mlp_max_iter=200,
            random_state=1, device="cpu",
        ).fit(Xd[tr], Xc[tr], y[tr])
        changed = Xc.copy()
        changed[te] += 50
        second = AEPCAClassifier(
            ae_params=self.ae_params(n_spectral, feats),
            seeds=[1], n_pca=5, mlp_hidden=(16,), mlp_max_iter=200,
            random_state=1, device="cpu",
        ).fit(Xd[tr], changed[tr], y[tr])
        np.testing.assert_allclose(first.pca_.components_, second.pca_.components_)
        np.testing.assert_allclose(first.classical_scaler_.mean_, second.classical_scaler_.mean_)

    def test_multi_seed_cls_average_smoke(self):
        Xd, y, n_spectral, feats = self.depth_data()
        Xc = self.classical_data(len(y), y=y)
        model = AEPCAClassifier(
            ae_params=self.ae_params(n_spectral, feats),
            seeds=[2, 3], n_pca=6, mlp_hidden=(32,), mlp_max_iter=300,
            random_state=2, device="cpu",
        ).fit(Xd, Xc, y)
        self.assertEqual(len(model.ae_models_), 2)
        p = model.predict_proba(Xd, Xc)
        self.assertTrue(np.isfinite(p).all())


if __name__ == "__main__":
    unittest.main()
