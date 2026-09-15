"""Unit tests for the depth-profile Transformer autoencoder + CLS MLP."""

import unittest

import numpy as np
import torch
from sklearn.base import clone

from libs2026.autotransformer import (
    AutoDepthClassifier,
    AutoDepthTransformer,
    class_one_hot,
    sinusoidal_wavelength_encoding,
)


class AutoTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self, n=20, depth=8, n_spectral=48, n_intensity=3):
        rng = np.random.default_rng(5)
        feats = n_spectral + n_intensity
        X = rng.normal(size=(n, depth, feats)).astype(np.float32)
        y = np.tile([1, 5], n // 2)
        X[y == 5, :4, :16] += 2
        return X, y, n_spectral, feats

    def model(self, n_spectral, n_features, **kwargs):
        params = dict(
            n_spectral=n_spectral, n_features=n_features,
            n_shots=12, surface_shots=4, late_bin=2,
            d_model=16, n_layers=1, n_heads=2, ff_dim=32,
            epochs=2, patience=1, batch_size=4, dropout=0, device="cpu",
        )
        return AutoDepthClassifier(**{**params, **kwargs})

    def test_sinusoidal_pe_shape_and_finite(self):
        pe = sinusoidal_wavelength_encoding(64, 16)
        self.assertEqual(tuple(pe.shape), (64, 16))
        self.assertTrue(torch.isfinite(pe).all())
        pe_odd = sinusoidal_wavelength_encoding(10, 7)
        self.assertEqual(tuple(pe_odd.shape), (10, 7))

    def test_class_one_hot_matches_eye(self):
        y = np.array([0, 2, 1, 4])
        oh = class_one_hot(y, 5).numpy()
        expected = np.eye(5, dtype=np.float32)[y]
        np.testing.assert_array_equal(oh, expected)

    def test_cls_glued_and_recon_excludes_cls_length(self):
        X, y, n_spectral, feats = self.data()
        net = AutoDepthTransformer(
            n_features=feats, n_spectral=n_spectral, n_classes=2,
            d_model=16, n_layers=1, n_heads=2, ff_dim=32, n_depth=8,
        )
        xb = torch.from_numpy(X[:4])
        logits, recon = net(xb)
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(tuple(recon.shape), (4, 8, feats))
        # Reconstruction is only for depth tokens, not CLS.
        self.assertEqual(recon.shape[1], xb.shape[1])

    def test_forward_gradients_reach_encoder_and_heads(self):
        X, y, n_spectral, feats = self.data()
        net = AutoDepthTransformer(
            n_features=feats, n_spectral=n_spectral, n_classes=2,
            d_model=16, n_layers=1, n_heads=2, ff_dim=32, n_depth=8,
        )
        xb = torch.from_numpy(X[:4])
        yb = torch.tensor([0, 1, 0, 1])
        logits, recon = net(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb) + torch.nn.functional.mse_loss(recon, xb)
        loss.backward()
        self.assertGreater(float(net.cls.grad.abs().sum()), 0)
        self.assertGreater(float(net.recon_head[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(net.classifier[-1].weight.grad.abs().sum()), 0)

    def test_fit_predict_proba_and_one_hot_path(self):
        X, y, n_spectral, feats = self.data()
        model = self.model(n_spectral, feats).fit(X, y)
        p = model.predict_proba(X)
        self.assertEqual(p.shape, (20, 2))
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(set(model.predict(X)).issubset({1, 5}))
        self.assertGreaterEqual(model.best_epoch_, 1)
        recon = model.reconstruct(X)
        self.assertEqual(recon.shape, X.shape)

    def test_seed_reproducibility(self):
        X, y, n_spectral, feats = self.data()
        model = self.model(n_spectral, feats, val_fraction=0).fit(X, y)
        other = clone(model).fit(X, y)
        np.testing.assert_array_equal(model.predict_proba(X), other.predict_proba(X))

    def test_inner_validation_scaler_isolation(self):
        X, y, n_spectral, feats = self.data()
        first = self.model(n_spectral, feats).fit(X, y)
        changed = X.copy()
        changed[first.inner_val_indices_] += 100
        second = self.model(n_spectral, feats).fit(changed, y)
        np.testing.assert_array_equal(first.selection_mean_, second.selection_mean_)
        np.testing.assert_array_equal(first.selection_std_, second.selection_std_)

    def test_can_learn_simple_depth_signal(self):
        X, y, n_spectral, feats = self.data()
        model = self.model(n_spectral, feats, epochs=40, val_fraction=0, lr=0.003).fit(X, y)
        self.assertGreaterEqual(float(np.mean(model.predict(X) == y)), 0.9)

    def test_invalid_shape_rejected(self):
        X, y, n_spectral, feats = self.data()
        with self.assertRaises(ValueError):
            self.model(n_spectral, feats).fit(X[:, :, :-1], y)


if __name__ == "__main__":
    unittest.main()
