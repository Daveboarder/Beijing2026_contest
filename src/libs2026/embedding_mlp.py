"""Raw-spectrum MLP with a learned embedding layer (PyTorch, GPU-ready).

Replaces the ``StandardScaler -> PCA -> MLP`` pipeline: every pixel is
standardised on the training rows, then a narrow first layer learns the
projection PCA used to provide. By default that embedding is *linear* (the
learned counterpart of PCA scores), followed by the ReLU layers of ``pca_mlp``.

The estimator follows the scikit-learn API, so ``evaluation.cross_validate_model``
runs it unchanged. Several feature rows per sample (shot blocks) are expected
to be stored consecutively, ``rows_per_sample`` at a time, which is how
``features.build_features`` lays them out; the inner validation split used for
early stopping then keeps all rows of one sample on the same side.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch import nn


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class EmbeddingMLP(nn.Module):
    """``Linear(n_in, embedding)`` -> [activation] -> ReLU MLP -> class logits."""

    def __init__(self, n_in: int, n_classes: int, embedding: int = 30, hidden=(128, 64),
                 embedding_activation: str = "linear", dropout: float = 0.2):
        super().__init__()
        if embedding_activation not in ("linear", "relu"):
            raise ValueError("embedding_activation must be 'linear' or 'relu'")
        self.embed = nn.Linear(n_in, embedding)
        layers: list[nn.Module] = [nn.ReLU()] if embedding_activation == "relu" else []
        layers.append(nn.Dropout(dropout))
        width = embedding
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class EmbeddingMLPClassifier(ClassifierMixin, BaseEstimator):
    """Standardise -> embedding MLP, with early stopping on a sample-grouped inner split.

    Training: AdamW with cross-entropy (optionally class-balanced). When
    ``val_fraction > 0`` a stratified set of samples is held out, the epoch with
    the lowest validation loss is found (``patience`` epochs without
    improvement stop the run), and the network is retrained on all rows for
    that many epochs.
    """

    def __init__(self, embedding: int = 30, hidden=(128, 64), embedding_activation: str = "linear",
                 dropout: float = 0.2, lr: float = 1e-3, weight_decay: float = 1e-2,
                 batch_size: int = 64, epochs: int = 300, patience: int = 30,
                 val_fraction: float = 0.2, rows_per_sample: int = 1, class_weight: str | None = None,
                 random_state: int = 42, device: str = "auto"):
        self.embedding = embedding
        self.hidden = hidden
        self.embedding_activation = embedding_activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.rows_per_sample = rows_per_sample
        self.class_weight = class_weight
        self.random_state = random_state
        self.device = device

    # -- helpers ---------------------------------------------------------------------

    def _sample_ids(self, n_rows: int) -> np.ndarray:
        k = self.rows_per_sample
        if k < 1 or n_rows % k:
            raise ValueError(f"{n_rows} rows do not split into samples of {k} consecutive rows")
        return np.repeat(np.arange(n_rows // k), k)

    def _weights(self, labels: np.ndarray, device) -> torch.Tensor | None:
        if self.class_weight is None:
            return None
        if self.class_weight != "balanced":
            raise ValueError("class_weight must be None or 'balanced'")
        counts = np.bincount(labels, minlength=len(self.classes_))
        w = len(labels) / (len(counts) * np.maximum(counts, 1))
        return torch.tensor(w, dtype=torch.float32, device=device)

    def _standardise(self, x: torch.Tensor):
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        return mean, torch.where(std > 0, std, torch.ones_like(std))

    def _train(self, x, y, epochs, device, validation=None):
        torch.manual_seed(self.random_state)
        net = EmbeddingMLP(x.shape[1], len(self.classes_), self.embedding, tuple(self.hidden),
                           self.embedding_activation, self.dropout).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=self._weights(y.cpu().numpy(), device))
        generator = torch.Generator(device="cpu").manual_seed(self.random_state)
        best_loss, best_epoch, stale, history = float("inf"), epochs, 0, []
        for epoch in range(epochs):
            net.train()
            order = torch.randperm(len(y), generator=generator).to(device)
            for start in range(0, len(y), self.batch_size):
                idx = order[start:start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(net(x[idx]), y[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                optimizer.step()
            if validation is not None:
                net.eval()
                vx, vy = validation
                with torch.no_grad():
                    val_loss = float(loss_fn(net(vx), vy))
                history.append(val_loss)
                if val_loss < best_loss - 1e-5:
                    best_loss, best_epoch, stale = val_loss, epoch + 1, 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        return net.eval(), best_epoch, history

    # -- scikit-learn API ------------------------------------------------------------

    def fit(self, X, y):
        x_np = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if x_np.ndim != 2 or len(x_np) != len(y) or not np.isfinite(x_np).all():
            raise ValueError("Expected a finite 2-D feature matrix with one label per row")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        self.classes_, labels = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required")
        self.n_features_in_ = x_np.shape[1]
        device = resolve_device(self.device)
        x = torch.from_numpy(x_np).to(device)
        yt = torch.from_numpy(labels.astype(np.int64)).to(device)

        self.best_epoch_, self.val_history_ = self.epochs, []
        if self.val_fraction:
            samples = self._sample_ids(len(y))
            first = np.flatnonzero(np.r_[True, samples[1:] != samples[:-1]])
            if not all((labels[samples == s] == labels[i]).all() for s, i in enumerate(first)):
                raise ValueError("Rows of one sample carry different labels; check rows_per_sample")
            tr_s, va_s = train_test_split(np.arange(len(first)), test_size=self.val_fraction,
                                          stratify=labels[first], random_state=self.random_state)
            tr = torch.from_numpy(np.flatnonzero(np.isin(samples, tr_s))).to(device)
            va = torch.from_numpy(np.flatnonzero(np.isin(samples, va_s))).to(device)
            mean, std = self._standardise(x[tr])
            _, self.best_epoch_, self.val_history_ = self._train(
                (x[tr] - mean) / std, yt[tr], self.epochs, device,
                validation=((x[va] - mean) / std, yt[va]),
            )
        mean, std = self._standardise(x)
        net, _, _ = self._train((x - mean) / std, yt, self.best_epoch_, device)
        self.net_ = net.cpu()
        self.mean_, self.std_ = mean.cpu(), std.cpu()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "net_")
        device = resolve_device(self.device)
        x = torch.from_numpy(np.asarray(X, dtype=np.float32))
        net = self.net_.to(device)
        out = []
        with torch.no_grad():
            for start in range(0, len(x), 1024):
                xb = ((x[start:start + 1024] - self.mean_) / self.std_).to(device)
                out.append(torch.softmax(net(xb), dim=1).cpu().numpy())
        self.net_ = net.cpu()
        return np.concatenate(out) if out else np.empty((0, len(self.classes_)))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def transform(self, X):
        """Embedding vectors ``(n_rows, embedding)`` of the fitted network."""
        check_is_fitted(self, "net_")
        x = (torch.from_numpy(np.asarray(X, dtype=np.float32)) - self.mean_) / self.std_
        with torch.no_grad():
            return self.net_.embed(x).numpy()
