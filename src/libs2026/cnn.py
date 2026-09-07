"""2-D CNN that treats each sample as a depth-spectrum image.

Input layout is ``(batch, 1, n_shots, n_wavelengths)``: the shot axis is depth
(aging goes surface → bulk) and the spectral axis is a per-fold PCA projection
of the LIBS spectrum. Kept deliberately small — with 120 labels, capacity hurts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for the CNN models. Install with:\n"
            "  uv sync --extra cnn"
        ) from _TORCH_IMPORT_ERROR


if torch is not None:

    def _conv_block(in_ch: int, out_ch: int, k_depth: int = 3, k_wl: int = 7,
                    pool: tuple[int, int] = (2, 2)) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(k_depth, k_wl),
                      padding=(k_depth // 2, k_wl // 2), bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=(k_depth, k_wl),
                      padding=(k_depth // 2, k_wl // 2), bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=pool),
        )

    class DepthSpectrumCNN(nn.Module):
        """Compact 2-D CNN for (shots × spectral-PCA) LIBS images.

        Wavelength is pooled away; ``depth_bins`` rows are kept so the
        surface→bulk curve reaches the classifier (global pool erases aging).
        """

        def __init__(self, n_classes: int = 5, channels: tuple[int, ...] = (32, 64, 128),
                     dropout: float = 0.4, depth_bins: int = 12):
            super().__init__()
            layers = []
            in_ch = 1
            for out_ch in channels:
                layers.append(_conv_block(in_ch, out_ch))
                in_ch = out_ch
            self.backbone = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d((depth_bins, 1))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(channels[-1] * depth_bins, channels[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels[-1], n_classes),
            )
            self.depth_bins = depth_bins

        def forward(self, x):
            return self.head(self.pool(self.backbone(x)))

else:  # pragma: no cover
    DepthSpectrumCNN = None  # type: ignore[misc, assignment]


def _to_images(X: np.ndarray, n_shots: int, n_wavelengths: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4:
        return X
    if X.ndim == 3:
        return X[:, None, :, :]
    if X.ndim == 2:
        if X.shape[1] != n_shots * n_wavelengths:
            raise ValueError(
                f"Flat X has {X.shape[1]} features, expected "
                f"{n_shots}*{n_wavelengths}={n_shots * n_wavelengths}"
            )
        return X.reshape(-1, 1, n_shots, n_wavelengths)
    raise ValueError(f"Unsupported X ndim={X.ndim}")


class SpectrumCNN(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible trainer around :class:`DepthSpectrumCNN`."""

    def __init__(
        self,
        n_shots: int = 50,
        n_wavelengths: int = 1535,
        spectral_pca: int = 48,
        channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.4,
        depth_bins: int = 12,
        epochs: int = 150,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 40,
        val_fraction: float = 0.0,
        noise_std: float = 0.03,
        wl_shift: int = 2,
        mixup_alpha: float = 0.0,
        label_smoothing: float = 0.05,
        log_intensity: bool = False,
        device: str | None = None,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.n_shots = n_shots
        self.n_wavelengths = n_wavelengths
        self.spectral_pca = spectral_pca
        self.channels = channels
        self.dropout = dropout
        self.depth_bins = depth_bins
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.noise_std = noise_std
        self.wl_shift = wl_shift
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.log_intensity = log_intensity
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_device(self):
        _require_torch()
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _augment(self, batch: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        out = batch
        if self.noise_std > 0:
            out = out + self.noise_std * torch.randn_like(out)
        if self.wl_shift > 0 and out.shape[0] > 0:
            shifts = rng.integers(-self.wl_shift, self.wl_shift + 1, size=out.shape[0])
            w = out.shape[-1]
            base = torch.arange(w, device=out.device)
            idx = (base.unsqueeze(0) - torch.as_tensor(shifts, device=out.device).unsqueeze(1)) % w
            idx = idx.view(out.shape[0], 1, 1, w).expand_as(out)
            out = torch.gather(out, -1, idx)
        return out

    def _mixup(self, xb, yb, n_classes, rng):
        if self.mixup_alpha <= 0 or xb.shape[0] < 2:
            return xb, yb, False
        lam = float(rng.beta(self.mixup_alpha, self.mixup_alpha))
        perm = torch.randperm(xb.shape[0], device=xb.device)
        y_a = torch.nn.functional.one_hot(yb, n_classes).float()
        return lam * xb + (1.0 - lam) * xb[perm], lam * y_a + (1.0 - lam) * y_a[perm], True

    def _prepare_images(self, images: np.ndarray, fit_pca: bool,
                        train_idx: np.ndarray | None = None) -> np.ndarray:
        from sklearn.decomposition import PCA

        if self.log_intensity:
            images = np.log1p(np.maximum(images, 0.0))

        if self.spectral_pca and self.spectral_pca > 0:
            n, h, w = images.shape
            flat = images.reshape(-1, w)
            if fit_pca:
                n_comp = min(self.spectral_pca, flat.shape[0] - 1, w)
                src = images[train_idx].reshape(-1, w) if train_idx is not None else flat
                self.pca_ = PCA(n_components=n_comp, random_state=self.random_state).fit(src)
            images = self.pca_.transform(flat).reshape(n, h, -1).astype(np.float32)

        if fit_pca:
            src = images[train_idx] if train_idx is not None else images
            self.mean_ = float(src.mean())
            self.std_ = float(src.std()) + 1e-6
            self.log_intensity_ = bool(self.log_intensity)
            self.width_ = int(images.shape[-1])

        return ((images - self.mean_) / self.std_)[:, None, :, :].astype(np.float32)

    def fit(self, X, y):
        _require_torch()
        device = self._resolve_device()
        images = _to_images(X, self.n_shots, self.n_wavelengths)[:, 0]
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        rng = np.random.default_rng(self.random_state)
        train_idx = np.arange(len(y_idx))
        prepared = self._prepare_images(images, fit_pca=True, train_idx=train_idx)

        x_train = torch.from_numpy(prepared[train_idx])
        y_train = torch.from_numpy(y_idx[train_idx].astype(np.int64))
        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=min(self.batch_size, len(train_idx)),
            shuffle=True, drop_last=False,
            pin_memory=device.type == "cuda",
        )

        counts = np.bincount(y_idx[train_idx], minlength=n_classes).astype(np.float64)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        hard_criterion = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=self.label_smoothing,
        )

        net = DepthSpectrumCNN(
            n_classes=n_classes, channels=tuple(self.channels),
            dropout=self.dropout, depth_bins=self.depth_bins,
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(self.epochs, 1), eta_min=self.lr * 0.05,
        )

        for epoch in range(self.epochs):
            net.train()
            for xb, yb in train_loader:
                xb = self._augment(xb.to(device, non_blocking=True), rng)
                yb = yb.to(device, non_blocking=True)
                xb_m, y_m, mixed = self._mixup(xb, yb, n_classes, rng)
                opt.zero_grad(set_to_none=True)
                logits = net(xb_m)
                if mixed:
                    log_probs = torch.log_softmax(logits, dim=1)
                    loss = -(y_m * log_probs * class_weight.unsqueeze(0)).sum(dim=1).mean()
                    loss = loss / class_weight.mean()
                else:
                    loss = hard_criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            if self.verbose and (epoch + 1) % 50 == 0:
                net.eval()
                with torch.no_grad():
                    pred = net(x_train.to(device)).argmax(dim=1).cpu().numpy()
                print(
                    f"epoch {epoch + 1:3d}  "
                    f"train_acc={float((pred == y_idx[train_idx]).mean()):.3f}"
                )

        net.eval()
        self.net_ = net.to("cpu")
        self.n_features_in_ = self.n_shots * self.n_wavelengths
        self.device_ = str(device)
        return self

    def _logits(self, X) -> np.ndarray:
        check_is_fitted(self, "net_")
        _require_torch()
        images = _to_images(X, self.n_shots, self.n_wavelengths)[:, 0]
        prepared = self._prepare_images(images, fit_pca=False)
        device = self._resolve_device()
        self.net_.to(device).eval()
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(prepared).to(device))
        return logits.cpu().numpy()

    def predict_proba(self, X):
        logits = self._logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._logits(X).argmax(axis=1)]

    def __sklearn_tags__(self) -> Any:  # pragma: no cover
        return super().__sklearn_tags__()
