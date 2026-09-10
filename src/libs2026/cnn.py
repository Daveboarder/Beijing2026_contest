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

    class DepthTokenCNN(nn.Module):
        """2-D CNN over ``(channels, depth, lines)`` spectral-line tokens.

        Columns are physical transitions rather than wavelength bins, so the
        convolution shares weights over something meaningful: a kernel can
        respond to "an ionic line of a heavy element that weakens near the
        surface" because the element, ion stage and excitation energy travel
        alongside the fitted amplitude in the static channels.

        The depth axis is preserved exactly as in :class:`DepthSpectrumCNN`;
        pooling it away costs most of the aging signal.
        """

        def __init__(self, n_classes: int = 5, in_channels: int = 16,
                     channels: tuple[int, ...] = (32, 64, 128),
                     dropout: float = 0.4, depth_bins: int = 12,
                     n_rows: int = 50):
            super().__init__()
            layers = []
            in_ch = in_channels
            # Pool the line axis hard and the depth axis gently. With only the
            # first 30 shots, skip the first depth-pool so the surface profile
            # is not crushed before the head sees it.
            pools = ([(1, 4), (1, 4), (2, 4)] if n_rows <= 32
                     else [(1, 4), (2, 4), (2, 4)])
            for idx, out_ch in enumerate(channels):
                # Narrower kernels than the pixel CNN: adjacent columns are
                # separate transitions, not neighbouring detector pixels, so a
                # wide receptive field along the line axis mixes unrelated
                # physics instead of resolving one peak.
                layers.append(_conv_block(
                    in_ch, out_ch, k_depth=3, k_wl=3,
                    pool=pools[idx] if idx < len(pools) else (1, 2),
                ))
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

    class MultiScaleSpectrumEncoder(nn.Module):
        """Tokenise one spectrum into a single embedding vector.

        Three convolution branches run in parallel over the wavelength axis
        with different receptive fields. On this spectrometer one detector
        pixel is 0.02-0.05 nm, so at ``bin_factor 8`` a width-3 kernel spans
        roughly a line core, width 7 a full Voigt profile with its wings, and
        width 15 a blend of neighbouring transitions plus the continuum under
        them. Concatenating the three gives the depth model a token that
        already carries peak shape and local background together, which a
        single kernel width cannot express.
        """

        def __init__(self, kernel_sizes: tuple[int, ...] = (3, 7, 15),
                     conv_channels: int = 32, d_model: int = 128,
                     dropout: float = 0.1):
            super().__init__()
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(1, conv_channels, kernel_size=k,
                              padding=k // 2, stride=2, bias=False),
                    nn.GroupNorm(min(8, conv_channels), conv_channels),
                    nn.GELU(),
                    nn.MaxPool1d(4),
                )
                for k in kernel_sizes
            ])
            fused = conv_channels * len(kernel_sizes)
            self.fuse = nn.Sequential(
                nn.Conv1d(fused, fused, kernel_size=5, padding=2, bias=False),
                nn.GroupNorm(min(8, fused), fused),
                nn.GELU(),
                nn.MaxPool1d(4),
                nn.Conv1d(fused, d_model, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(8, d_model), d_model),
                nn.GELU(),
            )
            # Average and max pooling over what is left of the wavelength axis:
            # the mean tracks overall emission, the max keeps the strongest
            # surviving line response, which averaging alone washes out.
            self.project = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.d_model = d_model

        def forward(self, x):
            """``(n_spectra, 1, n_wavelengths)`` to ``(n_spectra, d_model)``."""
            merged = torch.cat([branch(x) for branch in self.branches], dim=1)
            feat = self.fuse(merged)
            pooled = torch.cat([feat.mean(dim=-1), feat.amax(dim=-1)], dim=-1)
            return self.project(pooled)

    class DepthTransformer(nn.Module):
        """Multi-scale spectrum tokens, then self-attention along depth.

        Every shot becomes one token, so the sequence *is* the depth profile
        and attention can relate the aged surface directly to the bulk
        plateau however far apart they sit — the fixed receptive field of the
        2-D CNN cannot. A learned position embedding keeps shot order (and
        therefore depth) meaningful, and a CLS token reads out the sequence.
        """

        def __init__(self, n_classes: int = 5, n_shots: int = 50,
                     kernel_sizes: tuple[int, ...] = (3, 7, 15),
                     conv_channels: int = 32, d_model: int = 128,
                     n_layers: int = 2, n_heads: int = 4, ff_mult: int = 2,
                     dropout: float = 0.0, encoder_dropout: float = 0.0):
            super().__init__()
            self.encoder = MultiScaleSpectrumEncoder(
                kernel_sizes=kernel_sizes, conv_channels=conv_channels,
                d_model=d_model, dropout=encoder_dropout,
            )
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.position = nn.Parameter(torch.zeros(1, n_shots + 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
            nn.init.trunc_normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * ff_mult,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            # CLS and mean-of-shots are concatenated: with 120 samples the CLS
            # token alone trains slowly, and the mean is a stable fallback.
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, x, shot_mask=None):
            """``x`` is ``(batch, n_shots, n_wavelengths)``."""
            b, s, w = x.shape
            tokens = self.encoder(x.reshape(b * s, 1, w)).reshape(b, s, -1)
            tokens = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1)
            tokens = tokens + self.position[:, : s + 1]
            pad = None
            if shot_mask is not None:
                keep_cls = torch.zeros(b, 1, dtype=torch.bool, device=x.device)
                pad = torch.cat([keep_cls, shot_mask], dim=1)
            out = self.transformer(tokens, src_key_padding_mask=pad)
            out = self.norm(out)
            if pad is None:
                pooled = out[:, 1:].mean(dim=1)
            else:
                visible = (~pad[:, 1:]).float().unsqueeze(-1)
                pooled = (out[:, 1:] * visible).sum(dim=1) / visible.sum(dim=1).clamp(min=1.0)
            return self.head(torch.cat([out[:, 0], pooled], dim=-1))

else:  # pragma: no cover
    DepthSpectrumCNN = None  # type: ignore[misc, assignment]
    DepthTokenCNN = None  # type: ignore[misc, assignment]
    MultiScaleSpectrumEncoder = None  # type: ignore[misc, assignment]
    DepthTransformer = None  # type: ignore[misc, assignment]


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


class SpectrumTransformer(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible trainer around :class:`DepthTransformer`.

    ``X`` is the same flattened ``(n_shots x n_wavelengths)`` image the pixel
    CNN consumes, but no spectral PCA is applied: the multi-scale convolution
    needs neighbouring wavelength bins to stay neighbours. Each shot is
    tokenised to ``d_model`` numbers first, so the transformer only ever sees
    a length-``n_shots`` sequence and the attention cost stays trivial.
    """

    def __init__(
        self,
        n_shots: int = 50,
        n_wavelengths: int = 1533,
        kernel_sizes: tuple[int, ...] = (3, 7, 15),
        conv_channels: int = 32,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 2,
        # Defaults match the ``no_reg`` winner of ``16_tune_transformer.py``:
        # dropout 0.3 + weight_decay 1e-2 collapsed every seed to a constant
        # 1/5 posterior on N=120. Light weight decay and no stochastic
        # regularisation are what actually train.
        dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        epochs: int = 150,
        batch_size: int = 16,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_frac: float = 0.1,
        noise_std: float = 0.0,
        wl_shift: int = 0,
        shot_mask_p: float = 0.0,
        label_smoothing: float = 0.05,
        device: str | None = None,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.n_shots = n_shots
        self.n_wavelengths = n_wavelengths
        self.kernel_sizes = kernel_sizes
        self.conv_channels = conv_channels
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.encoder_dropout = encoder_dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_frac = warmup_frac
        self.noise_std = noise_std
        self.wl_shift = wl_shift
        self.shot_mask_p = shot_mask_p
        self.label_smoothing = label_smoothing
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_device(self):
        _require_torch()
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare(self, images: np.ndarray, fit_stats: bool) -> np.ndarray:
        if fit_stats:
            self.mean_ = float(images.mean())
            self.std_ = float(images.std()) + 1e-6
        return ((images - self.mean_) / self.std_).astype(np.float32)

    def _augment(self, batch: torch.Tensor, rng: np.random.Generator):
        out = batch
        if self.noise_std > 0:
            out = out + self.noise_std * torch.randn_like(out)
        if self.wl_shift > 0:
            # One shared shift per sample: a wavelength miscalibration moves
            # the whole depth profile, not individual shots.
            shifts = rng.integers(-self.wl_shift, self.wl_shift + 1, size=out.shape[0])
            if np.any(shifts != 0):
                w = out.shape[-1]
                base = torch.arange(w, device=out.device)
                idx = (base.unsqueeze(0)
                       - torch.as_tensor(shifts, device=out.device).unsqueeze(1)) % w
                out = torch.gather(out, -1, idx.unsqueeze(1).expand_as(out))
        mask = None
        if self.shot_mask_p > 0 and out.shape[1] > 4:
            drawn = torch.rand(out.shape[0], out.shape[1], device=out.device)
            mask = drawn < self.shot_mask_p
            # Never hide a whole profile, otherwise the attention has nothing
            # to pool over and the loss goes to NaN.
            mask[:, 0] = False
        return out, mask

    def _build_net(self, n_classes: int):
        return DepthTransformer(
            n_classes=n_classes, n_shots=self.n_shots,
            kernel_sizes=tuple(self.kernel_sizes),
            conv_channels=self.conv_channels, d_model=self.d_model,
            n_layers=self.n_layers, n_heads=self.n_heads, ff_mult=self.ff_mult,
            dropout=self.dropout, encoder_dropout=self.encoder_dropout,
        )

    def fit(self, X, y):
        _require_torch()
        device = self._resolve_device()
        images = _to_images(X, self.n_shots, self.n_wavelengths)[:, 0]
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        prepared = self._prepare(images, fit_stats=True)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(prepared),
                          torch.from_numpy(y_idx.astype(np.int64))),
            batch_size=min(self.batch_size, len(y_idx)),
            shuffle=True, drop_last=False, pin_memory=device.type == "cuda",
        )

        counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=self.label_smoothing,
        )

        net = self._build_net(n_classes).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        steps = max(self.epochs * max(len(loader), 1), 1)
        warmup = max(int(steps * self.warmup_frac), 1)
        # Transformers do not tolerate a cold start at this batch size; warm
        # up linearly, then anneal.
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: (
                (step + 1) / warmup if step < warmup
                else 0.05 + 0.95 * 0.5 * (
                    1.0 + np.cos(np.pi * (step - warmup) / max(steps - warmup, 1))
                )
            ),
        )

        for epoch in range(self.epochs):
            net.train()
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                xb, mask = self._augment(xb, rng)
                opt.zero_grad(set_to_none=True)
                loss = criterion(net(xb, shot_mask=mask), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"epoch {epoch + 1:3d}  loss={float(loss):.3f}")

        net.eval()
        self.net_ = net
        self.device_ = str(device)
        self.n_features_in_ = self.n_shots * self.n_wavelengths
        return self

    def _logits(self, X) -> np.ndarray:
        check_is_fitted(self, "net_")
        _require_torch()
        images = _to_images(X, self.n_shots, self.n_wavelengths)[:, 0]
        prepared = self._prepare(images, fit_stats=False)
        device = self._resolve_device()
        self.net_.to(device).eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(prepared), self.batch_size):
                xb = torch.from_numpy(prepared[start:start + self.batch_size]).to(device)
                outputs.append(self.net_(xb).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X):
        logits = self._logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._logits(X).argmax(axis=1)]

    def __sklearn_tags__(self) -> Any:  # pragma: no cover
        return super().__sklearn_tags__()


class TokenCNN(ClassifierMixin, BaseEstimator):
    """2-D CNN over spectral-line tokens, wrapped for sklearn.

    ``X`` carries only the channels that vary per sample, flattened from
    ``(n_rows, n_lines, n_sample_features)``. The static physics channels are
    identical for every sample, so they are passed once through ``static`` and
    broadcast inside the forward pass instead of being stored 180 x 50 times.

    Unlike :class:`SpectrumCNN` there is no spectral PCA: the dictionary has
    already reduced 12282 bins to ~1000 physically meaningful columns, and a
    PCA over tokens would destroy the per-column identity that makes weight
    sharing across the line axis meaningful in the first place.
    """

    def __init__(
        self,
        static: np.ndarray | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        n_rows: int = 50,
        n_lines: int = 722,
        n_features: int = 7,
        channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.4,
        depth_bins: int = 12,
        epochs: int = 150,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        noise_std: float = 0.05,
        token_dropout: float = 0.1,
        depth_roll: int = 1,
        mixup_alpha: float = 0.0,
        label_smoothing: float = 0.05,
        valid_index: int = 5,
        n_dynamic: int = 5,
        include_static: bool = True,
        device: str | None = None,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.static = static
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.n_rows = n_rows
        self.n_lines = n_lines
        self.n_features = n_features
        self.channels = channels
        self.dropout = dropout
        self.depth_bins = depth_bins
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.noise_std = noise_std
        self.token_dropout = token_dropout
        self.depth_roll = depth_roll
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.valid_index = valid_index
        self.n_dynamic = n_dynamic
        self.include_static = include_static
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _resolve_device(self):
        _require_torch()
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare(self, X: np.ndarray) -> torch.Tensor:
        """Flat rows to a z-scored ``(n, features, depth, lines)`` tensor."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X.reshape(-1, self.n_rows, self.n_lines, self.n_features)
        mean = np.asarray(self.feature_mean, dtype=np.float32)
        std = np.asarray(self.feature_std, dtype=np.float32)
        out = (X - mean[None, None, :, :]) / std[None, None, :, :]

        # A failed fit stores raw zeros; z-scoring would turn them into a
        # spurious constant, so restore an exact zero. When fit_valid is in
        # the tensor it also carries the "nothing here" flag; without it
        # (R² + Δλ only) an all-zero token is the failed-fit marker.
        n_feat = out.shape[-1]
        if 0 <= self.valid_index < n_feat:
            invalid = X[..., self.valid_index] <= 0.5
            n_dyn = min(self.n_dynamic, n_feat)
            out[invalid, : n_dyn] = 0.0
            out[..., self.valid_index] = X[..., self.valid_index]
        else:
            failed = np.all(np.abs(X) < 1e-12, axis=-1)
            out[failed] = 0.0

        return torch.from_numpy(np.ascontiguousarray(out.transpose(0, 3, 1, 2)))

    def _static_tensor(self, device) -> torch.Tensor:
        static = np.asarray(self.static, dtype=np.float32)
        mu = static.mean(axis=0, keepdims=True)
        sigma = static.std(axis=0, keepdims=True)
        static = (static - mu) / np.where(sigma > 1e-8, sigma, 1.0)
        # (n_lines, n_static) -> (1, n_static, 1, n_lines) for broadcasting.
        return torch.from_numpy(static.T[None, :, None, :].copy()).to(device)

    def _append_static(self, xb: "torch.Tensor") -> "torch.Tensor":
        if self.static_ is None:
            return xb
        static = self.static_.to(xb.device)
        return torch.cat(
            [xb, static.expand(xb.shape[0], -1, xb.shape[2], -1)], dim=1,
        )

    def _augment(self, batch: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        out = batch
        if self.noise_std > 0:
            noise = torch.randn_like(out) * self.noise_std
            # The mask channel is categorical; perturbing it is meaningless.
            if 0 <= self.valid_index < out.shape[1]:
                noise[:, self.valid_index] = 0.0
            out = out + noise
        if self.token_dropout > 0:
            keep = (torch.rand(out.shape[0], 1, 1, out.shape[3], device=out.device)
                    >= self.token_dropout).float()
            out = out * keep
        if self.depth_roll > 0 and out.shape[2] > 4:
            shifts = rng.integers(-self.depth_roll, self.depth_roll + 1, size=out.shape[0])
            if np.any(shifts != 0):
                out = torch.stack([
                    torch.roll(out[i], int(s), dims=1) if s else out[i]
                    for i, s in enumerate(shifts)
                ], dim=0)
        return out

    def _mixup(self, xb, yb, n_classes, rng):
        if self.mixup_alpha <= 0 or xb.shape[0] < 2:
            return xb, yb, False
        lam = float(rng.beta(self.mixup_alpha, self.mixup_alpha))
        perm = torch.randperm(xb.shape[0], device=xb.device)
        y_a = torch.nn.functional.one_hot(yb, n_classes).float()
        return lam * xb + (1.0 - lam) * xb[perm], lam * y_a + (1.0 - lam) * y_a[perm], True

    def fit(self, X, y):
        _require_torch()
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("TokenCNN needs feature_mean and feature_std from the TokenSet")
        if self.include_static and self.static is None:
            raise ValueError("TokenCNN needs static from the TokenSet when include_static=True")
        device = self._resolve_device()
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        rng = np.random.default_rng(self.random_state)

        x_train = self._prepare(X)
        y_train = torch.from_numpy(y_idx.astype(np.int64))
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=min(self.batch_size, len(y_idx)),
            shuffle=True, drop_last=False, pin_memory=device.type == "cuda",
        )

        counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=self.label_smoothing,
        )

        n_static = 0
        self.static_ = None
        if self.include_static:
            self.static_ = self._static_tensor(device)
            n_static = int(self.static_.shape[1])
        net = DepthTokenCNN(
            n_classes=n_classes, in_channels=self.n_features + n_static,
            channels=tuple(self.channels), dropout=self.dropout,
            depth_bins=self.depth_bins, n_rows=self.n_rows,
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(self.epochs, 1), eta_min=self.lr * 0.05,
        )

        for epoch in range(self.epochs):
            net.train()
            for xb, yb in loader:
                xb = self._augment(xb.to(device, non_blocking=True), rng)
                yb = yb.to(device, non_blocking=True)
                xb, y_m, mixed = self._mixup(xb, yb, n_classes, rng)
                xb = self._append_static(xb)
                opt.zero_grad(set_to_none=True)
                logits = net(xb)
                if mixed:
                    log_probs = torch.log_softmax(logits, dim=1)
                    loss = -(y_m * log_probs * class_weight.unsqueeze(0)).sum(dim=1).mean()
                    loss = loss / class_weight.mean()
                else:
                    loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"epoch {epoch + 1:3d}  loss={float(loss):.3f}")

        net.eval()
        self.net_ = net
        self.device_ = str(device)
        self.n_features_in_ = self.n_rows * self.n_lines * self.n_features
        return self

    def _logits(self, X) -> np.ndarray:
        check_is_fitted(self, "net_")
        device = self._resolve_device()
        self.net_.to(device).eval()
        prepared = self._prepare(X)
        outputs = []
        with torch.no_grad():
            # Chunked: the token tensor is far larger than a PCA image stack.
            for start in range(0, prepared.shape[0], 16):
                xb = prepared[start:start + 16].to(device)
                xb = self._append_static(xb)
                outputs.append(self.net_(xb).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X):
        logits = self._logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._logits(X).argmax(axis=1)]

    def __sklearn_tags__(self) -> Any:  # pragma: no cover
        return super().__sklearn_tags__()
