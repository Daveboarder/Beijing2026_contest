"""Unit tests for the PyTorch embedding MLP."""

import unittest

import numpy as np
import torch
from sklearn.base import clone

from libs2026.embedding_mlp import EmbeddingMLP, EmbeddingMLPClassifier
from libs2026.evaluation import cross_validate_model


def block_data(n_samples=30, rows=3, n_features=200, seed=0):
    """Consecutive rows per sample; class 5 carries a shifted pixel band."""
    rng = np.random.default_rng(seed)
    labels = np.tile([1, 3, 5], n_samples // 3)
    X = rng.normal(size=(n_samples * rows, n_features)).astype(np.float32)
    y = np.repeat(labels, rows)
    X[y == 5, :20] += 2.0
    X[y == 3, 20:40] -= 2.0
    groups = np.repeat(np.arange(n_samples), rows)
    return X, y, groups


class EmbeddingMLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def params(self, **kw):
        base = dict(embedding=8, hidden=(16,), epochs=60, patience=10, batch_size=16,
                    rows_per_sample=3, device="cpu")
        base.update(kw)
        return base

    def test_network_shapes_and_linear_embedding(self):
        net = EmbeddingMLP(50, 5, embedding=7, hidden=(12, 6))
        self.assertEqual(net(torch.zeros(4, 50)).shape, (4, 5))
        self.assertNotIsInstance(net.head[0], torch.nn.ReLU)
        relu = EmbeddingMLP(50, 5, embedding=7, embedding_activation="relu")
        self.assertIsInstance(relu.head[0], torch.nn.ReLU)

    def test_fit_predict_learns_signal(self):
        X, y, _ = block_data()
        clf = EmbeddingMLPClassifier(**self.params()).fit(X, y)
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        self.assertGreater((clf.predict(X) == y).mean(), 0.9)
        self.assertEqual(clf.transform(X).shape, (len(X), 8))
        self.assertTrue(1 <= clf.best_epoch_ <= 60)

    def test_deterministic_under_seed(self):
        X, y, _ = block_data()
        a = EmbeddingMLPClassifier(**self.params()).fit(X, y).predict_proba(X)
        b = EmbeddingMLPClassifier(**self.params()).fit(X, y).predict_proba(X)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_rows_per_sample_mismatch_raises(self):
        X, y, _ = block_data()
        with self.assertRaises(ValueError):
            EmbeddingMLPClassifier(**self.params(rows_per_sample=4)).fit(X, y)
        shuffled = np.random.default_rng(1).permutation(len(y))
        with self.assertRaises(ValueError):
            EmbeddingMLPClassifier(**self.params()).fit(X[shuffled], y[shuffled])

    def test_no_validation_trains_full_epochs(self):
        X, y, _ = block_data()
        clf = EmbeddingMLPClassifier(**self.params(val_fraction=0.0, epochs=5)).fit(X, y)
        self.assertEqual(clf.best_epoch_, 5)

    def test_cross_validation_compatible(self):
        X, y, groups = block_data()
        sample_ids = np.array([f"s{g:02d}" for g in groups])
        model = clone(EmbeddingMLPClassifier(**self.params(class_weight="balanced")))
        result = cross_validate_model(model, X, y, groups, sample_ids, n_splits=3, n_repeats=1)
        self.assertEqual(len(result.oof), 30)
        self.assertGreater(result.accuracy, 0.8)


if __name__ == "__main__":
    unittest.main()
