"""Transformer autoencoder on raw depth profiles with a glued CLS + MLP head.

Each physical sample is a depth sequence of binned spectra (from
``depth_transformer.prepare_sample``). Wavelength positions receive fixed
sinusoidal encodings; a CLS token is prepended; the encoder reconstructs the
depth tokens and the CLS state feeds a small MLP. Class labels are 5-dim
one-hot targets for the classification loss only.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .depth_transformer import depth_windows


def sinusoidal_wavelength_encoding(n_wl: int, d_model: int) -> torch.Tensor:
    """Fixed sin/cos positional encoding over wavelength indices, shape ``(n_wl, d_model)``."""
    if n_wl < 1 or d_model < 1:
        raise ValueError("n_wl and d_model must be positive")
    position = torch.arange(n_wl, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
    )
    pe = torch.zeros(n_wl, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[0: pe[:, 1::2].shape[1]])
    return pe


def class_one_hot(y: np.ndarray | torch.Tensor, n_classes: int) -> torch.Tensor:
    """Integer class indices → one-hot float matrix ``(n, n_classes)``."""
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(np.asarray(y, dtype=np.int64))
    else:
        y = y.long()
    if y.ndim != 1:
        raise ValueError("class indices must be 1-D")
    return torch.nn.functional.one_hot(y, num_classes=n_classes).float()


class AutoDepthTransformer(nn.Module):
    """Encode depth tokens with CLS; reconstruct depth features; classify from CLS."""

    def __init__(self, n_features: int, n_spectral: int, n_classes: int = 5,
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 ff_dim: int = 128, dropout: float = 0.1, n_depth: int = 65):
        super().__init__()
        if n_spectral < 1 or n_spectral > n_features:
            raise ValueError("n_spectral must be in [1, n_features]")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_features = n_features
        self.n_spectral = n_spectral
        self.n_classes = n_classes
        self.d_model = d_model

        pe = sinusoidal_wavelength_encoding(n_spectral, d_model)
        self.register_buffer("wavelength_pe", pe)

        # Intensity-weighted sinusoidal bases over wavelength + linear map of the row.
        self.row_projection = nn.Linear(n_features, d_model)
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.depth_pe = nn.Parameter(torch.zeros(1, n_depth + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.depth_pe, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ff_dim, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, n_features),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, n_classes),
        )

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, depth, features)`` → ``(batch, depth, d_model)``."""
        spectral = x[..., : self.n_spectral]
        weights = spectral / spectral.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
        # (batch, depth, wl) x (wl, d_model) → intensity-weighted PE token.
        pe_token = torch.einsum("bdw,we->bde", weights, self.wavelength_pe)
        pe_token = pe_token / self.n_spectral * self.pe_scale
        return self.row_projection(x) + pe_token

    def encode(self, x: torch.Tensor):
        """Return ``(cls, depth_states)`` after the encoder + LayerNorm."""
        batch, depth, _ = x.shape
        tokens = self.tokenize(x)
        cls = self.cls.expand(batch, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = seq + self.depth_pe[:, : depth + 1]
        encoded = self.norm(self.encoder(seq))
        return encoded[:, 0], encoded[:, 1:]

    def forward(self, x: torch.Tensor):
        """Return ``(logits, reconstruction)``; recon has shape ``(batch, depth, features)``."""
        cls_out, depth_out = self.encode(x)
        return self.classifier(cls_out), self.recon_head(depth_out)


class AutoDepthClassifier(ClassifierMixin, BaseEstimator):
    """Joint reconstruction + one-hot classification with inner epoch selection."""

    def __init__(self, n_spectral: int = 3069, n_features: int = 3072,
                 n_shots: int = 200, surface_shots: int = 20, late_bin: int = 4,
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 ff_dim: int = 128, dropout: float = 0.1, lambda_recon: float = 1.0,
                 epochs: int = 100, patience: int = 15, val_fraction: float = 0.2,
                 batch_size: int = 8, lr: float = 0.0003, weight_decay: float = 0.001,
                 random_state: int = 42, device: str = "cpu"):
        self.n_spectral = n_spectral
        self.n_features = n_features
        self.n_shots = n_shots
        self.surface_shots = surface_shots
        self.late_bin = late_bin
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.lambda_recon = lambda_recon
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

    def _n_depth(self) -> int:
        return len(depth_windows(self.n_shots, self.surface_shots, self.late_bin))

    def _array(self, X):
        x = np.asarray(X, dtype=np.float32)
        expected = (self._n_depth(), self.n_features)
        if x.ndim != 3 or x.shape[1:] != expected or not np.isfinite(x).all():
            raise ValueError(f"Expected finite input (samples, {expected[0]}, {expected[1]})")
        return x

    def _stats(self, x):
        mean = x.mean(axis=(0, 1)).astype(np.float32)
        std = np.maximum(x.std(axis=(0, 1)).astype(np.float32), 1e-6)
        return mean, std

    def _net(self):
        torch.manual_seed(self.random_state)
        return AutoDepthTransformer(
            n_features=self.n_features, n_spectral=self.n_spectral,
            n_classes=len(self.classes_), d_model=self.d_model,
            n_layers=self.n_layers, n_heads=self.n_heads, ff_dim=self.ff_dim,
            dropout=self.dropout, n_depth=self._n_depth(),
        ).to(self.device)

    def _joint_loss(self, logits, recon, xb, y_onehot, class_weight):
        # Soft CE against one-hot targets (equivalent to hard CE for hard one-hots).
        log_probs = torch.log_softmax(logits, dim=-1)
        cls = -(y_onehot * log_probs * class_weight.unsqueeze(0)).sum(dim=-1).mean()
        cls = cls / class_weight.mean()
        recon_loss = torch.nn.functional.mse_loss(recon, xb)
        return cls + self.lambda_recon * recon_loss, cls, recon_loss

    def _train(self, x, y, epochs, validation=None):
        net = self._net()
        generator = torch.Generator().manual_seed(self.random_state)
        y_idx = torch.from_numpy(y.astype(np.int64))
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x), y_idx),
            batch_size=self.batch_size, shuffle=True, generator=generator,
        )
        counts = np.bincount(y, minlength=len(self.classes_))
        weights = len(y) / (len(counts) * np.maximum(counts, 1))
        class_weight = torch.tensor(weights, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        best_loss, best_epoch, stale = float("inf"), 0, 0
        for epoch in range(epochs):
            net.train()
            for xb, yb in loader:
                xb = xb.to(self.device)
                y_onehot = class_one_hot(yb, len(self.classes_)).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits, recon = net(xb)
                loss, _, _ = self._joint_loss(logits, recon, xb, y_onehot, class_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
            if validation is not None:
                net.eval()
                vx, vy = validation
                total_cls = 0.0
                with torch.no_grad():
                    for start in range(0, len(vy), self.batch_size):
                        xb = torch.from_numpy(vx[start:start + self.batch_size]).to(self.device)
                        yb = torch.from_numpy(vy[start:start + self.batch_size])
                        y_onehot = class_one_hot(yb, len(self.classes_)).to(self.device)
                        logits, recon = net(xb)
                        _, cls, _ = self._joint_loss(
                            logits, recon, xb, y_onehot, class_weight,
                        )
                        total_cls += float(cls) * len(yb)
                val_loss = total_cls / len(vy)
                if not np.isfinite(val_loss):
                    raise RuntimeError("Non-finite validation loss")
                if val_loss < best_loss - 1e-6:
                    best_loss, best_epoch, stale = val_loss, epoch + 1, 0
                else:
                    stale += 1
                if stale >= self.patience:
                    break
        return net.cpu().eval(), best_epoch if validation is not None else epochs

    def fit(self, X, y):
        x = self._array(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(x) or len(x) == 0:
            raise ValueError("One label is required per physical sample")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.lambda_recon < 0:
            raise ValueError("lambda_recon must be non-negative")
        self.classes_, labels = np.unique(y, return_inverse=True)
        labels = labels.astype(np.int64)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required")
        self.n_features_in_ = x.shape[-1]
        self.best_epoch_ = self.epochs
        if self.val_fraction:
            tr, va = train_test_split(
                np.arange(len(y)), test_size=self.val_fraction,
                stratify=labels, random_state=self.random_state,
            )
            self.inner_train_indices_, self.inner_val_indices_ = tr, va
            mean, std = self._stats(x[tr])
            self.selection_mean_, self.selection_std_ = mean, std
            _, self.best_epoch_ = self._train(
                (x[tr] - mean) / std, labels[tr], self.epochs,
                ((x[va] - mean) / std, labels[va]),
            )
        self.mean_, self.std_ = self._stats(x)
        self.net_, _ = self._train((x - self.mean_) / self.std_, labels, self.best_epoch_)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "net_")
        x = (self._array(X) - self.mean_) / self.std_
        if len(x) == 0:
            return np.empty((0, len(self.classes_)))
        net = self.net_.to(self.device).eval()
        output = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                xb = torch.from_numpy(x[start:start + self.batch_size]).to(self.device)
                logits, _ = net(xb)
                output.append(torch.softmax(logits, dim=-1).cpu().numpy())
        net.cpu()
        return np.concatenate(output)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def transform(self, X):
        """Return CLS embeddings ``(n_samples, d_model)`` from the fitted encoder."""
        check_is_fitted(self, "net_")
        x = (self._array(X) - self.mean_) / self.std_
        if len(x) == 0:
            return np.empty((0, self.d_model), dtype=np.float32)
        net = self.net_.to(self.device).eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                xb = torch.from_numpy(x[start:start + self.batch_size]).to(self.device)
                cls_out, _ = net.encode(xb)
                parts.append(cls_out.cpu().numpy())
        net.cpu()
        return np.concatenate(parts)

    def reconstruct(self, X):
        """Return reconstructed depth features (same shape as ``X``)."""
        check_is_fitted(self, "net_")
        x = (self._array(X) - self.mean_) / self.std_
        net = self.net_.to(self.device).eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                xb = torch.from_numpy(x[start:start + self.batch_size]).to(self.device)
                _, recon = net(xb)
                parts.append(recon.cpu().numpy())
        net.cpu()
        return np.concatenate(parts) * self.std_ + self.mean_
