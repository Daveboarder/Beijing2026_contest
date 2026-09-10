"""Shared spectral CNN and a small depth encoder with an end-to-end MLP head.

Input is one physical sample per row: (samples, depth tokens, wavelengths +
channel intensities). No learned preprocessing is performed outside ``fit``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_index, load_shots
from .features import bin_spectrum
from .preprocessing import Preprocessor


def depth_windows(n_shots=200, surface_shots=20, late_bin=4):
    """Keep early shots separately; cover every later shot exactly once."""
    if not 0 < surface_shots < n_shots or late_bin < 1:
        raise ValueError("Require 0 < surface_shots < n_shots and late_bin >= 1")
    return ([(i, i + 1) for i in range(surface_shots)]
            + [(i, min(i + late_bin, n_shots))
               for i in range(surface_shots, n_shots, late_bin)])


def prepare_sample(shots, pre, bin_factor=4, surface_shots=20, late_bin=4):
    """Repair first, then retain spectral shape and bulk-relative intensity.

    All scales are derived from this sample alone. Channel L2 norms are measured
    before wavelength/depth averaging. No continuum subtraction is performed.
    """
    shots = np.asarray(shots, dtype=np.float32)
    bounds = tuple(pre.channel_bounds)
    if (shots.ndim != 2 or bounds[0] != 0 or bounds[-1] != shots.shape[1]
            or any(b <= a for a, b in zip(bounds[:-1], bounds[1:]))
            or bin_factor < 1 or not np.isfinite(shots).all()):
        raise ValueError("Invalid spectra, channel boundaries, or bin factor")
    if any((b - a) // bin_factor < 8 for a, b in zip(bounds[:-1], bounds[1:])):
        raise ValueError("Each binned channel must have at least 8 wavelengths")
    cleaned = replace(pre, normalization="none", baseline="none", smooth_window=0)(shots)
    normalized, intensities = [], []
    bulk = pre.bulk_slice(len(shots))
    for a, b in zip(bounds[:-1], bounds[1:]):
        channel = cleaned[:, a:b]
        scale = np.linalg.norm(channel, axis=1, keepdims=True)
        normalized.append(channel / np.maximum(scale, 1e-8) * np.sqrt(b - a))
        reference = max(float(scale[bulk].mean()), 1e-8)
        intensities.append(np.log1p(scale / reference))
    spectra = bin_spectrum(np.concatenate(normalized, axis=1), bin_factor, bounds)
    packed = np.concatenate([spectra, *intensities], axis=1)
    return np.stack([packed[a:b].mean(axis=0)
                     for a, b in depth_windows(len(shots), surface_shots, late_bin)])


def build_depth_sequences(cfg, split="train", bin_factor=4, surface_shots=20, late_bin=4):
    """Read the existing raw-shot cache; avoid stale learned-feature caches."""
    index = load_index(cfg)
    index = index[index["split"] == split].reset_index(drop=True)
    if index.empty:
        raise ValueError(f"No samples for split {split!r}")
    pre = Preprocessor.from_config(cfg)
    arrays = []
    for sid in index["sample_id"]:
        shots = load_shots(cfg, sid, mmap=False)
        if shots.shape[0] != cfg["data"]["n_shots"]:
            raise ValueError(f"Unexpected shot count in {sid}")
        arrays.append(prepare_sample(shots, pre, bin_factor, surface_shots, late_bin))
    return np.stack(arrays).astype(np.float32), index


class SpectralDepthNet(nn.Module):
    """Convolve only within detector channels; share each branch across depths."""

    def __init__(self, channel_widths, positions, bin_widths, regions, n_classes,
                 d_model=64, heads=4, ff_dim=128, dropout=0.2,
                 depth_encoder="transformer", use_intensity=True):
        super().__init__()
        if depth_encoder not in {"transformer", "pool", "conv"}:
            raise ValueError("depth_encoder must be transformer, pool, or conv")
        self.channel_widths = tuple(channel_widths)
        self.use_intensity = use_intensity
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3), nn.GroupNorm(4, 16), nn.GELU(),
                nn.AvgPool1d(2),
                nn.Conv1d(16, 32, 5, padding=2), nn.GroupNorm(4, 32), nn.GELU(),
                nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            ) for _ in channel_widths
        ])
        self.spectral_projection = nn.Linear(len(channel_widths) * 32 * 8, d_model)
        self.position_projection = nn.Linear(2, d_model)
        self.intensity_projection = nn.Linear(len(channel_widths), d_model, bias=False)
        self.register_buffer("coordinates", torch.tensor(
            np.column_stack([positions, bin_widths]), dtype=torch.float32))
        self.register_buffer("regions", torch.tensor(regions, dtype=torch.long))
        self.register_buffer("bin_widths", torch.tensor(bin_widths, dtype=torch.float32))
        self.depth_encoder = depth_encoder
        if depth_encoder == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model, heads, ff_dim, dropout, activation="gelu", batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        elif depth_encoder == "conv":
            self.encoder = nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU(),
                nn.Dropout(dropout), nn.Conv1d(d_model, d_model, 3, padding=1),
            )
        else:
            self.encoder = nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(3 * d_model, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        batch, depth, _ = x.shape
        offset, features = 0, []
        for width, branch in zip(self.channel_widths, self.branches):
            channel = x[..., offset:offset + width].reshape(batch * depth, 1, width)
            features.append(branch(channel).reshape(batch, depth, -1))
            offset += width
        z = self.spectral_projection(torch.cat(features, dim=-1))
        z = z + self.position_projection(self.coordinates)[None]
        if self.use_intensity:
            z = z + self.intensity_projection(x[..., offset:])
        if self.depth_encoder == "conv":
            z = z + self.encoder(z.transpose(1, 2)).transpose(1, 2)
        else:
            z = self.encoder(z)
        z = self.norm(z)
        # Weight unequal bins by their shot counts inside each physical region.
        pooled = []
        for region in range(3):
            mask = self.regions == region
            weights = self.bin_widths[mask]
            pooled.append((z[:, mask] * weights[None, :, None]).sum(1) / weights.sum())
        return self.head(torch.cat(pooled, dim=1))


class DepthTransformerClassifier(ClassifierMixin, BaseEstimator):
    """Inner-validation epoch selection followed by refitting the outer train set.

    ``fit`` receives one row per physical sample. Validation samples never affect
    selection-stage normalization, gradients, or class weights. The selected
    epoch count is then used to refit on all samples supplied to ``fit``.
    """

    def __init__(self, channel_widths=(1023, 1023, 1023), n_shots=200,
                 surface_shots=20, late_bin=4, bulk_start=140, d_model=64,
                 heads=4, ff_dim=128, dropout=0.2, depth_encoder="transformer",
                 use_intensity=True, epochs=100, patience=15, val_fraction=0.2,
                 batch_size=8, lr=0.0003, weight_decay=0.001,
                 random_state=42, device="cpu"):
        self.channel_widths = channel_widths
        self.n_shots = n_shots
        self.surface_shots = surface_shots
        self.late_bin = late_bin
        self.bulk_start = bulk_start
        self.d_model = d_model
        self.heads = heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.depth_encoder = depth_encoder
        self.use_intensity = use_intensity
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

    def _array(self, X):
        x = np.asarray(X, dtype=np.float32)
        expected = (len(depth_windows(self.n_shots, self.surface_shots, self.late_bin)),
                    sum(self.channel_widths) + len(self.channel_widths))
        if x.ndim != 3 or x.shape[1:] != expected or not np.isfinite(x).all():
            raise ValueError(f"Expected finite input (samples, {expected[0]}, {expected[1]})")
        return x

    def _stats(self, x):
        # One scalar per detector channel, preserving relative wavelength intensities.
        mean = np.zeros(x.shape[-1], np.float32)
        std = np.ones_like(mean)
        offset = 0
        for width in self.channel_widths:
            block = x[..., offset:offset + width]
            mean[offset:offset + width] = block.mean()
            std[offset:offset + width] = max(float(block.std()), 1e-6)
            offset += width
        mean[offset:] = x[..., offset:].mean(axis=(0, 1))
        std[offset:] = np.maximum(x[..., offset:].std(axis=(0, 1)), 1e-6)
        return mean, std

    def _net(self):
        windows = depth_windows(self.n_shots, self.surface_shots, self.late_bin)
        centers = np.array([(a + b - 1) / 2 for a, b in windows])
        widths = np.array([b - a for a, b in windows])
        if self.bulk_start not in [a for a, _ in windows]:
            raise ValueError("bulk_start must align with a depth-window boundary")
        regions = np.where(centers < self.surface_shots, 0,
                           np.where(centers < self.bulk_start, 1, 2))
        if len(np.unique(regions)) != 3:
            raise ValueError("Early, intermediate and bulk regions must all be nonempty")
        torch.manual_seed(self.random_state)
        return SpectralDepthNet(
            self.channel_widths, centers / max(self.n_shots - 1, 1),
            widths / self.n_shots, regions, len(self.classes_), self.d_model,
            self.heads, self.ff_dim, self.dropout, self.depth_encoder, self.use_intensity,
        ).to(self.device)

    def _train(self, x, y, epochs, validation=None):
        net = self._net()
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                            batch_size=self.batch_size, shuffle=True, generator=generator)
        counts = np.bincount(y, minlength=len(self.classes_))
        weights = len(y) / (len(counts) * np.maximum(counts, 1))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device))
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        best_loss, best_epoch, stale = float("inf"), 0, 0
        for epoch in range(epochs):
            net.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(net(xb.to(self.device)), yb.to(self.device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
            if validation is not None:
                net.eval()
                vx, vy = validation
                total = 0.0
                with torch.no_grad():
                    for start in range(0, len(vy), self.batch_size):
                        logits = net(torch.from_numpy(vx[start:start + self.batch_size])
                                     .to(self.device))
                        target = torch.from_numpy(vy[start:start + self.batch_size]).to(self.device)
                        total += nn.functional.cross_entropy(logits, target, reduction="sum").item()
                val_loss = total / len(vy)
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
        self.classes_, labels = np.unique(y, return_inverse=True)
        labels = labels.astype(np.int64)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required")
        self.n_features_in_ = x.shape[-1]
        self.best_epoch_ = self.epochs
        if self.val_fraction:
            tr, va = train_test_split(np.arange(len(y)), test_size=self.val_fraction,
                                     stratify=labels, random_state=self.random_state)
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
                logits = net(torch.from_numpy(x[start:start + self.batch_size]).to(self.device))
                output.append(torch.softmax(logits, dim=-1).cpu().numpy())
        net.cpu()
        return np.concatenate(output)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
