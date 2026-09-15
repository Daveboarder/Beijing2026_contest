"""Fuse depth-autoencoder CLS embeddings with classical PCA scores for MLP classification.

Per fold/train set:
1. Fit one or more ``AutoDepthClassifier`` seeds on **one depth sequence per sample**.
2. Average CLS vectors across seeds.
3. **Broadcast** each sample CLS onto all classical feature rows of that sample
   (e.g. ``n_groups=4``).
4. Fit ``StandardScaler`` + ``PCA`` on classical rows; concat CLS ∥ PCA; MLP.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .autotransformer import AutoDepthClassifier


def row_sample_indices(depth_sample_ids, classical_sample_ids) -> np.ndarray:
    """Map each classical row to an index into the depth / CLS sample axis."""
    depth_sample_ids = np.asarray(depth_sample_ids)
    classical_sample_ids = np.asarray(classical_sample_ids)
    order = {sid: i for i, sid in enumerate(depth_sample_ids)}
    try:
        return np.asarray([order[sid] for sid in classical_sample_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Classical features missing depth sample_id {exc}") from exc


def remap_row_indices(row_sample_idx: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
    """Restrict classical rows to ``sample_indices`` and remap to local 0..n-1."""
    row_sample_idx = np.asarray(row_sample_idx, dtype=np.int64)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    keep = np.isin(row_sample_idx, sample_indices)
    local = {int(g): i for i, g in enumerate(sample_indices)}
    remapped = np.asarray([local[int(g)] for g in row_sample_idx[keep]], dtype=np.int64)
    return np.flatnonzero(keep), remapped


class AEPCAClassifier(ClassifierMixin, BaseEstimator):
    """CLS (autoencoder) broadcast onto classical rows ∥ PCA → MLP."""

    def __init__(
        self,
        ae_params: dict | None = None,
        seeds: list[int] | tuple[int, ...] | None = None,
        n_pca: int = 30,
        mlp_hidden: tuple[int, ...] = (256, 128),
        mlp_alpha: float = 1e-4,
        mlp_max_iter: int = 2000,
        random_state: int = 42,
        device: str = "cpu",
    ):
        self.ae_params = ae_params
        self.seeds = seeds
        self.n_pca = n_pca
        self.mlp_hidden = mlp_hidden
        self.mlp_alpha = mlp_alpha
        self.mlp_max_iter = mlp_max_iter
        self.random_state = random_state
        self.device = device

    def _ae_seeds(self) -> list[int]:
        if self.seeds is None:
            return [self.random_state]
        seeds = list(self.seeds)
        if not seeds:
            raise ValueError("seeds must be non-empty")
        return seeds

    def _fit_autoencoders(self, X_depth, y):
        base = dict(self.ae_params or {})
        epochs_by_seed = base.pop("epochs_by_seed", None) or {}
        models = []
        for seed in self._ae_seeds():
            params = dict(base)
            key = str(seed)
            if key in epochs_by_seed:
                params["epochs"] = int(epochs_by_seed[key])
            model = AutoDepthClassifier(**params, device=self.device, random_state=seed)
            model.fit(X_depth, y)
            models.append(model)
        return models

    def _mean_cls(self, X_depth) -> np.ndarray:
        parts = [model.transform(X_depth) for model in self.ae_models_]
        return np.mean(parts, axis=0).astype(np.float32)

    def _broadcast_cls(self, cls: np.ndarray, row_sample_idx: np.ndarray | None,
                       n_rows: int) -> np.ndarray:
        if row_sample_idx is None:
            if len(cls) != n_rows:
                raise ValueError(
                    "Without row_sample_idx, depth samples and classical rows must match"
                )
            return cls
        idx = np.asarray(row_sample_idx, dtype=np.int64)
        if idx.ndim != 1 or len(idx) != n_rows:
            raise ValueError("row_sample_idx must have one entry per classical row")
        if idx.min() < 0 or idx.max() >= len(cls):
            raise ValueError("row_sample_idx out of range for depth / CLS samples")
        return cls[idx]

    def _labels_for_depth(self, y_rows, row_sample_idx, n_depth):
        if row_sample_idx is None:
            return np.asarray(y_rows)
        idx = np.asarray(row_sample_idx, dtype=np.int64)
        y_rows = np.asarray(y_rows)
        y_depth = np.empty(n_depth, dtype=y_rows.dtype)
        for sample in range(n_depth):
            mask = idx == sample
            if not np.any(mask):
                raise ValueError(f"No classical rows for depth sample {sample}")
            labs = np.unique(y_rows[mask])
            if len(labs) != 1:
                raise ValueError("Classical rows for one sample must share one label")
            y_depth[sample] = labs[0]
        return y_depth

    def fit(self, X_depth, X_classical, y, row_sample_idx=None):
        """Fit AE on depth samples; PCA+MLP on classical rows with broadcast CLS.

        ``row_sample_idx[i]`` is the depth-sample index for classical row ``i``.
        Omit it when there is exactly one classical row per depth sample.
        """
        X_classical = np.asarray(X_classical, dtype=np.float64)
        y = np.asarray(y)
        if X_classical.ndim != 2 or len(X_classical) != len(y) or len(y) == 0:
            raise ValueError("Classical features must be 2-D with one row per label")
        if not np.isfinite(X_classical).all():
            raise ValueError("Classical features must be finite")
        if self.n_pca < 1:
            raise ValueError("n_pca must be positive")
        n_pca = min(self.n_pca, X_classical.shape[0], X_classical.shape[1])

        y_depth = self._labels_for_depth(y, row_sample_idx, len(X_depth))
        self.ae_models_ = self._fit_autoencoders(X_depth, y_depth)
        self.classes_ = self.ae_models_[0].classes_
        for model in self.ae_models_[1:]:
            if not np.array_equal(model.classes_, self.classes_):
                raise ValueError("Autoencoder seeds disagree on class labels")

        cls = self._mean_cls(X_depth)
        cls_rows = self._broadcast_cls(cls, row_sample_idx, len(X_classical))
        self.classical_scaler_ = StandardScaler().fit(X_classical)
        scaled = self.classical_scaler_.transform(X_classical)
        self.pca_ = PCA(n_components=n_pca, random_state=self.random_state).fit(scaled)
        scores = self.pca_.transform(scaled)
        fused = np.concatenate([cls_rows, scores], axis=1)
        self.fuse_scaler_ = StandardScaler().fit(fused)
        self.mlp_ = MLPClassifier(
            hidden_layer_sizes=tuple(self.mlp_hidden),
            alpha=self.mlp_alpha,
            max_iter=self.mlp_max_iter,
            random_state=self.random_state,
        )
        self.mlp_.fit(self.fuse_scaler_.transform(fused), y)
        self.n_features_in_ = (X_depth.shape[-1], X_classical.shape[1])
        self.d_model_ = cls.shape[1]
        self.n_pca_ = int(self.pca_.n_components_)
        return self

    def _fused(self, X_depth, X_classical, row_sample_idx=None) -> np.ndarray:
        check_is_fitted(self, "mlp_")
        X_classical = np.asarray(X_classical, dtype=np.float64)
        if X_classical.ndim != 2 or X_classical.shape[1] != self.n_features_in_[1]:
            raise ValueError("Classical feature width does not match training")
        if not np.isfinite(X_classical).all():
            raise ValueError("Classical features must be finite")
        cls = self._mean_cls(X_depth)
        cls_rows = self._broadcast_cls(cls, row_sample_idx, len(X_classical))
        scores = self.pca_.transform(self.classical_scaler_.transform(X_classical))
        return self.fuse_scaler_.transform(np.concatenate([cls_rows, scores], axis=1))

    def predict_proba(self, X_depth, X_classical, row_sample_idx=None):
        """Return per-classical-row probabilities ``(n_rows, n_classes)``."""
        fused = self._fused(X_depth, X_classical, row_sample_idx=row_sample_idx)
        proba = self.mlp_.predict_proba(fused)
        order = [int(np.where(self.mlp_.classes_ == c)[0][0]) for c in self.classes_]
        return proba[:, order]

    def predict(self, X_depth, X_classical, row_sample_idx=None):
        return self.classes_[
            self.predict_proba(X_depth, X_classical, row_sample_idx).argmax(axis=1)
        ]

    def transform(self, X_depth, X_classical, row_sample_idx=None):
        """Return fused features after scaling (broadcast CLS ∥ PCA scores)."""
        return self._fused(X_depth, X_classical, row_sample_idx=row_sample_idx)
