"""2-D CNN that treats each sample as a depth-spectrum image.

Input layout is ``(batch, 1, n_shots, n_wavelengths)``: the shot axis is depth
(aging goes surface → bulk from top to bottom) and the wavelength axis is the
LIBS spectrum. With only 120 labelled samples the network has to stay small and
heavily regularised; wavelength binning (see :mod:`.images`) keeps the width
manageable.
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
except ImportError as exc:  # pragma: no cover - optional dependency
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

    def _conv_block(in_ch: int, out_ch: int, k_depth: int = 3, k_wl: int = 9,
                    pool: tuple[int, int] = (2, 4)) -> nn.Sequential:
        """Anisotropic conv + GroupNorm (stable with tiny batches).

        Pooling is stronger along wavelength than along depth so the aged-layer
        structure (only a few shots thick) is not erased in the first layers.
        """
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
        """Compact 2-D CNN for (shots x wavelengths) LIBS images.

        Global average pooling over the *whole* image would erase the depth
        axis (where the aged-layer dip lives). Wavelength is pooled away; a
        few depth bins are kept so the head still sees surface vs bulk.
        """

        def __init__(self, n_classes: int = 5, channels: tuple[int, ...] = (32, 64, 128),
                     dropout: float = 0.3, depth_bins: int = 8):
            super().__init__()
            pools = [(2, 4), (2, 4), (2, 4)]
            layers = []
            in_ch = 1
            for i, out_ch in enumerate(channels):
                pool = pools[min(i, len(pools) - 1)]
                layers.append(_conv_block(in_ch, out_ch, pool=pool))
                in_ch = out_ch
            self.backbone = nn.Sequential(*layers)
            # Keep ``depth_bins`` rows, collapse wavelength to 1 column.
            self.pool = nn.AdaptiveAvgPool2d((depth_bins, 1))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(channels[-1] * depth_bins, channels[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels[-1], n_classes),
            )

        def forward(self, x):
            return self.head(self.pool(self.backbone(x)))

else:  # pragma: no cover
    DepthSpectrumCNN = None  # type: ignore[misc, assignment]

def _to_images(X: np.ndarray, n_shots: int, n_wavelengths: int) -> np.ndarray:
    """Accept flat ``(N, H*W)`` or already-shaped ``(N, H, W)`` / ``(N, 1, H, W)``."""
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
    """Sklearn-compatible trainer around :class:`DepthSpectrumCNN`.

    ``X`` may be the flattened image matrix from :meth:`ImageSet.as_flat` so the
    estimator plugs into the existing grouped cross-validation helpers.
    """

    def __init__(
        self,
        n_shots: int = 200,
        n_wavelengths: int = 1535,
        channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.3,
        epochs: int = 100,
        batch_size: int = 8,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        patience: int = 25,
        val_fraction: float = 0.2,
        noise_std: float = 0.03,
        wl_shift: int = 8,
        label_smoothing: float = 0.05,
        log_intensity: bool = True,
        device: str | None = None,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.n_shots = n_shots
        self.n_wavelengths = n_wavelengths
        self.channels = channels
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.noise_std = noise_std
        self.wl_shift = wl_shift
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

    def _augment(self, batch: "torch.Tensor", rng: np.random.Generator) -> "torch.Tensor":
        """Cheap on-the-fly augmentations that respect the depth axis."""
        out = batch
        if self.noise_std > 0:
            out = out + self.noise_std * torch.randn_like(out)
        if self.wl_shift > 0 and out.shape[0] > 0:
            # One shared wavelength roll per batch item, applied with gather —
            # never scramble the depth (shot) axis.
            shifts = rng.integers(-self.wl_shift, self.wl_shift + 1, size=out.shape[0])
            w = out.shape[-1]
            base = torch.arange(w, device=out.device)
            # idx[b, j] = (j - shift[b]) mod W
            idx = (base.unsqueeze(0) - torch.as_tensor(shifts, device=out.device).unsqueeze(1)) % w
            idx = idx.view(out.shape[0], 1, 1, w).expand_as(out)
            out = torch.gather(out, -1, idx)
        return out

    def fit(self, X, y):
        _require_torch()
        device = self._resolve_device()
        images = _to_images(X, self.n_shots, self.n_wavelengths)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        # Stratified hold-out inside the fold for early stopping.
        rng = np.random.default_rng(self.random_state)
        val_idx = np.array([], dtype=int)
        train_idx = np.arange(len(y_idx))
        if self.val_fraction > 0 and len(y_idx) >= 20:
            val_idx = []
            train_idx = []
            for c in range(n_classes):
                members = np.flatnonzero(y_idx == c)
                rng.shuffle(members)
                n_val = max(1, int(round(len(members) * self.val_fraction)))
                val_idx.extend(members[:n_val].tolist())
                train_idx.extend(members[n_val:].tolist())
            val_idx = np.asarray(val_idx, dtype=int)
            train_idx = np.asarray(train_idx, dtype=int)

        # Channel-wise standardisation from the training split only.
        train_imgs = images[train_idx]
        if self.log_intensity:
            train_imgs = np.log1p(np.maximum(train_imgs, 0.0))
            images = np.log1p(np.maximum(images, 0.0))
        self.mean_ = float(train_imgs.mean())
        self.std_ = float(train_imgs.std()) + 1e-6
        self.log_intensity_ = bool(self.log_intensity)

        def pack(idxs):
            arr = (images[idxs] - self.mean_) / self.std_
            return (
                torch.from_numpy(arr),
                torch.from_numpy(y_idx[idxs].astype(np.int64)),
            )

        x_train, y_train = pack(train_idx)
        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=min(self.batch_size, len(train_idx)),
            shuffle=True,
            drop_last=False,
        )

        counts = np.bincount(y_idx[train_idx], minlength=n_classes).astype(np.float64)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device),
            label_smoothing=self.label_smoothing,
        )

        net = DepthSpectrumCNN(
            n_classes=n_classes, channels=tuple(self.channels), dropout=self.dropout,
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1))

        best_state, best_val, wait = None, -np.inf, 0
        for epoch in range(self.epochs):
            net.train()
            for xb, yb in train_loader:
                xb = self._augment(xb, rng).to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(net(xb), yb)
                loss.backward()
                opt.step()
            sched.step()

            if len(val_idx):
                net.eval()
                with torch.no_grad():
                    xv, yv = pack(val_idx)
                    logits = net(xv.to(device))
                    pred = logits.argmax(dim=1).cpu().numpy()
                    val_acc = float((pred == y_idx[val_idx]).mean())
                if val_acc > best_val + 1e-4:
                    best_val, wait = val_acc, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        if self.verbose:
                            print(f"early stop at epoch {epoch + 1}, val_acc={best_val:.3f}")
                        break
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"epoch {epoch + 1:3d}  val_acc={val_acc:.3f}")
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_ = net.to("cpu")
        self.n_features_in_ = self.n_shots * self.n_wavelengths
        self.device_ = str(device)
        return self

    def _logits(self, X) -> np.ndarray:
        check_is_fitted(self, "net_")
        _require_torch()
        images = _to_images(X, self.n_shots, self.n_wavelengths)
        if getattr(self, "log_intensity_", False):
            images = np.log1p(np.maximum(images, 0.0))
        images = (images - self.mean_) / self.std_
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(images))
        return logits.numpy()

    def predict_proba(self, X):
        logits = self._logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._logits(X).argmax(axis=1)]

    def __sklearn_tags__(self) -> Any:  # pragma: no cover - sklearn 1.6+
        tags = super().__sklearn_tags__()
        return tags
