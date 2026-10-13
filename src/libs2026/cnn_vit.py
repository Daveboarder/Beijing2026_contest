"""2-D CNN + vision-transformer embedding of depth-spectrum images, classified by SVM or MLP.

Adapted from the few-shot hyperspectral classifier "ViT-CNN-BS": a residual
2-D CNN turns the input into a grid of feature vectors, the grid is flattened
into a token sequence with a class token and read by a transformer encoder,
the embedding is trained episodically with prototype losses (optionally with
attention-guided mixing of query samples and CenterCrop+Resize augmentation)
after a supervised band selection, and an RBF SVM classifies the embedding.

Here one sample is one image: rows are depth (shots 21-200, the unstable
first 20 omitted, averaged in groups of ``shot_bin``) and columns are
wavelength bins. Deviations from the article, all forced by the data: there
is no source domain to pre-train on, so training is episodic on the training
fold only; "bands" are the wavelength columns of the image and the PCA that
the article applies after band selection is omitted because wavelength is a
spatial axis here; the final classifier can be the article's SVM, an MLP, or
the nearest prototype.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from torch import nn

from .evaluation import predict_scores

LOSSES = ("proto", "ce")
HEADS = ("svm", "mlp", "proto")


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------


def depth_images(X, drop_shots=20, shot_bin=4):
    """Drop the first ``drop_shots`` rows of every image and average rows in groups.

    ``X`` is ``(n, n_shots, n_wl)``; returns ``(n, (n_shots - drop_shots) // shot_bin, n_wl)``.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or drop_shots < 0 or shot_bin < 1 or X.shape[1] - drop_shots < shot_bin:
        raise ValueError("Need (n, n_shots, n_wl) images with at least shot_bin rows left")
    rows = X[:, drop_shots:]
    n = (rows.shape[1] // shot_bin) * shot_bin
    return rows[:, :n].reshape(len(X), n // shot_bin, shot_bin, X.shape[2]).mean(axis=2)


def select_bands(X, y, n_bands, top_per_pair=None):
    """Supervised band selection of the article, on the depth-averaged spectra.

    Class-mean spectra are compared for every pair of classes; the ``top_per_pair``
    columns with the largest absolute difference are counted for each pair, and
    the ``n_bands`` most frequently selected columns (ties broken by the total
    difference) are kept, in their original wavelength order. Uses the labels,
    so it must be fitted on training samples only.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_cols = X.shape[-1]
    if n_bands >= n_cols or n_bands < 1:
        return np.arange(n_cols)
    top = n_bands if top_per_pair is None else int(top_per_pair)
    spectra = X.mean(axis=1) if X.ndim == 3 else X
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("Band selection needs at least two classes")
    means = np.stack([spectra[y == c].mean(axis=0) for c in classes])
    frequency = np.zeros(n_cols)
    total = np.zeros(n_cols)
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            diff = np.abs(means[i] - means[j])
            frequency[np.argsort(diff)[::-1][:top]] += 1
            total += diff
    order = np.lexsort((-total, -frequency))
    return np.sort(order[:n_bands])


def crop_resize(x, crop_min, rng):
    """CenterCrop + Resize augmentation on a batch ``(B, 1, H, W)``.

    Every image keeps a random fraction in ``[crop_min, 1]`` of its rows and of
    its columns around a jittered centre and is resized back to ``(H, W)``.
    """
    if crop_min >= 1.0:
        return x
    batch, _, height, width = x.shape
    out = []
    for i in range(batch):
        h = max(1, int(round(height * rng.uniform(crop_min, 1.0))))
        w = max(1, int(round(width * rng.uniform(crop_min, 1.0))))
        top = int(rng.integers(0, height - h + 1))
        left = int(rng.integers(0, width - w + 1))
        crop = x[i:i + 1, :, top:top + h, left:left + w]
        out.append(F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False))
    return torch.cat(out, dim=0)


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------


class ViTLayer(nn.Module):
    """Pre-norm transformer encoder layer that also returns its head-averaged attention."""

    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ff_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        attended, weights = self.attention(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, weights


class CnnVitNet(nn.Module):
    """Residual 2-D CNN -> pooled grid of tokens + class token -> transformer encoder."""

    def __init__(self, n_classes, channels=(16, 32), d_model=64, pool=(9, 24), n_layers=2,
                 n_heads=4, ff_dim=128, dropout=0.1, stride=4):
        super().__init__()
        c1, c2 = channels
        if d_model % n_heads or c2 % 2 or stride < 1 or n_layers < 1:
            raise ValueError("d_model must be divisible by n_heads, channels[1] even, stride >= 1, "
                             "n_layers >= 1")
        kernel = (3, 2 * stride - 1)
        padding = (1, stride - 1)
        # Mapping layer and two wavelength-strided convolutions shrink the long spectral axis.
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.Conv2d(c1, c2, kernel, stride=(1, stride), padding=padding), nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(c2, c2, kernel, stride=(1, stride), padding=padding), nn.BatchNorm2d(c2),
            nn.ReLU(),
        )
        # Residual bottleneck block of the article: 1x1 -> 3x3 -> 1x1 convolutions with a skip.
        mid = c2 // 2
        self.block = nn.Sequential(
            nn.Conv2d(c2, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, c2, 1), nn.BatchNorm2d(c2),
        )
        self.project = nn.Sequential(nn.Conv2d(c2, d_model, 1), nn.BatchNorm2d(d_model))
        self.pool = nn.AdaptiveAvgPool2d(tuple(pool))
        self.n_tokens = int(pool[0] * pool[1])
        self.pool_shape = (int(pool[0]), int(pool[1]))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positions = nn.Parameter(torch.zeros(1, self.n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.positions, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ViTLayer(d_model, n_heads, ff_dim, dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """``x`` is ``(B, 1, H, W)``; returns a dict with embedding, logits, tokens, attention."""
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected input (batch, 1, height, width), got {tuple(x.shape)}")
        features = self.stem(x)
        features = F.relu(features + self.block(features))
        grid = self.pool(self.project(features))                      # (B, D, Hp, Wp)
        tokens = grid.flatten(2).transpose(1, 2)                      # (B, T, D)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1) + self.positions
        tokens = self.dropout(tokens)
        weights = None
        for layer in self.layers:
            tokens, weights = layer(tokens)
        tokens = self.norm(tokens)
        embedding = tokens[:, 0]
        attention = weights[:, 0, 1:].reshape(len(x), *self.pool_shape)
        return {"embedding": embedding, "logits": self.head(embedding),
                "tokens": tokens[:, 1:], "attention": attention}


class PrototypeClassifier(ClassifierMixin, BaseEstimator):
    """Nearest class prototype in embedding space, softmax over negative squared distances."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        self.classes_ = np.unique(y)
        self.prototypes_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return -self.scale * ((X[:, None, :] - self.prototypes_[None]) ** 2).sum(-1)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


# ----------------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------------


class CnnVitClassifier(ClassifierMixin, BaseEstimator):
    """Episodically trained CNN+ViT embedding with an SVM / MLP / prototype head.

    ``fit`` receives one image per physical sample, ``(samples, height, width)``.
    Band selection (supervised) and per-column standardisation are fitted on
    training samples only. With ``val_fraction > 0`` a stratified inner split
    selects the number of training episodes (validation loss, ``patience``
    evaluations without improvement), then everything is refitted on all rows.
    """

    def __init__(self, n_bands=0, top_per_pair=None, channels=(16, 32), d_model=64,
                 pool=(9, 24), n_layers=2, n_heads=4, ff_dim=128, dropout=0.1, stride=4,
                 loss="proto", n_support=3, n_query=4, mix_queries=False, mix_weight=0.5,
                 crop_min=0.8, noise_std=0.0, episodes=600, eval_every=20, patience=5,
                 val_fraction=0.2, lr=0.001, weight_decay=0.001, head="svm", svm_c=1.0,
                 head_class_weight=None, mlp_hidden=(128, 64), mlp_alpha=0.001,
                 random_state=42, device="cpu"):
        self.n_bands = n_bands
        self.top_per_pair = top_per_pair
        self.channels = channels
        self.d_model = d_model
        self.pool = pool
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.stride = stride
        self.loss = loss
        self.n_support = n_support
        self.n_query = n_query
        self.mix_queries = mix_queries
        self.mix_weight = mix_weight
        self.crop_min = crop_min
        self.noise_std = noise_std
        self.episodes = episodes
        self.eval_every = eval_every
        self.patience = patience
        self.val_fraction = val_fraction
        self.lr = lr
        self.weight_decay = weight_decay
        self.head = head
        self.svm_c = svm_c
        self.head_class_weight = head_class_weight
        self.mlp_hidden = mlp_hidden
        self.mlp_alpha = mlp_alpha
        self.random_state = random_state
        self.device = device

    # -- validation and preprocessing ---------------------------------------------------

    def _array(self, X, check_shape=True):
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 3 or not np.isfinite(x).all():
            raise ValueError("Expected finite images shaped (samples, height, width)")
        if check_shape and hasattr(self, "n_features_in_") and x.shape[1:] != self.image_shape_:
            raise ValueError(f"Expected images of shape {self.image_shape_}, got {x.shape[1:]}")
        return x

    def _fit_preprocessing(self, x, labels):
        bands = select_bands(x, labels, self.n_bands, self.top_per_pair)
        selected = x[..., bands]
        mean = selected.mean(axis=(0, 1))
        std = np.maximum(selected.std(axis=(0, 1)), 1e-6)
        return bands, mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def _apply(x, bands, mean, std):
        return ((x[..., bands] - mean) / std).astype(np.float32)

    def _net(self):
        torch.manual_seed(self.random_state)
        return CnnVitNet(len(self.classes_), self.channels, self.d_model, self.pool, self.n_layers,
                         self.n_heads, self.ff_dim, self.dropout, self.stride).to(self.device)

    # -- training -----------------------------------------------------------------------

    def _episode(self, labels, rng):
        """Support and query row indices of one episode (every class, K + Q samples each)."""
        support, query = [], []
        for c in range(len(self.classes_)):
            rows = np.flatnonzero(labels == c)
            need = self.n_support + self.n_query
            picked = rng.choice(rows, need, replace=len(rows) < need)
            support.append(picked[:self.n_support])
            query.append(picked[self.n_support:])
        return np.concatenate(support), np.concatenate(query)

    def _augment(self, xb, rng):
        xb = crop_resize(xb, self.crop_min, rng)
        if self.noise_std > 0:
            xb = xb + self.noise_std * torch.randn_like(xb)
        return xb

    def _to_tensor(self, x):
        return torch.from_numpy(x).unsqueeze(1).to(self.device)

    def _mixing_loss(self, net, xq, yq, attention, prototypes):
        """Attention-guided synthetic queries (article eq. 10-15), paired by rolling the batch."""
        shift = torch.roll(torch.arange(len(xq), device=xq.device), 1)
        a1, a2 = attention, attention[shift]
        threshold = a1.flatten(1).median(dim=1).values[:, None, None]
        mask = (a1 >= threshold).float()                                           # (B, Hp, Wp)
        full = F.interpolate(mask[:, None], size=xq.shape[-2:], mode="nearest")   # (B, 1, H, W)
        synthetic = full * xq + (1 - full) * xq[shift]
        mass1 = (a1 * mask).flatten(1).sum(1)
        mass2 = (a2 * (1 - mask)).flatten(1).sum(1)
        lam1 = mass1 / (mass1 + mass2).clamp(min=1e-8)
        z = net(synthetic)["embedding"]
        logits = -torch.cdist(z, prototypes) ** 2
        loss1 = F.cross_entropy(logits, yq, reduction="none")
        loss2 = F.cross_entropy(logits, yq[shift], reduction="none")
        return (lam1 * loss1 + (1 - lam1) * loss2).mean()

    def _embed_all(self, net, x, batch_size=32):
        net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                out.append(net(self._to_tensor(x[start:start + batch_size]))["embedding"])
        return torch.cat(out) if out else torch.zeros(0, self.d_model, device=self.device)

    def _validation_loss(self, net, x_train, y_train, validation):
        """Unweighted cross-entropy of the validation samples against training prototypes."""
        vx, vy = validation
        with torch.no_grad():
            z_val = self._embed_all(net, vx)
            target = torch.from_numpy(vy).to(self.device)
            if self.loss == "ce":
                logits = net.head(z_val)
            else:
                z_train = self._embed_all(net, x_train)
                y_t = torch.from_numpy(y_train).to(self.device)
                prototypes = torch.stack([z_train[y_t == c].mean(0)
                                          for c in range(len(self.classes_))])
                logits = -torch.cdist(z_val, prototypes) ** 2
            return float(F.cross_entropy(logits, target))

    def _train(self, x, labels, episodes, validation=None):
        net = self._net()
        rng = np.random.default_rng(self.random_state)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        counts = np.bincount(labels, minlength=len(self.classes_))
        weights = len(labels) / (len(counts) * np.maximum(counts, 1))
        class_weight = torch.tensor(weights, dtype=torch.float32, device=self.device)
        best_loss, best_episodes, stale = float("inf"), episodes, 0
        for episode in range(1, episodes + 1):
            net.train()
            support, query = self._episode(labels, rng)
            optimizer.zero_grad(set_to_none=True)
            if self.loss == "proto":
                xs = self._augment(self._to_tensor(x[support]), rng)
                xq = self._augment(self._to_tensor(x[query]), rng)
                ys = torch.from_numpy(labels[support]).to(self.device)
                yq = torch.from_numpy(labels[query]).to(self.device)
                z_s = net(xs)["embedding"]
                out_q = net(xq)
                prototypes = torch.stack([z_s[ys == c].mean(0) for c in range(len(self.classes_))])
                logits = -torch.cdist(out_q["embedding"], prototypes) ** 2
                loss = F.cross_entropy(logits, yq)
                if self.mix_queries and len(query) > 1:
                    loss = loss + self.mix_weight * self._mixing_loss(
                        net, xq, yq, out_q["attention"].detach(), prototypes)
            else:
                rows = np.concatenate([support, query])
                xb = self._augment(self._to_tensor(x[rows]), rng)
                yb = torch.from_numpy(labels[rows]).to(self.device)
                loss = F.cross_entropy(net(xb)["logits"], yb, weight=class_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            if validation is not None and (episode % self.eval_every == 0 or episode == episodes):
                val_loss = self._validation_loss(net, x, labels, validation)
                if not np.isfinite(val_loss):
                    raise RuntimeError("Non-finite validation loss")
                if val_loss < best_loss - 1e-6:
                    best_loss, best_episodes, stale = val_loss, episode, 0
                else:
                    stale += 1
                if stale >= self.patience:
                    break
        return net.cpu().eval(), best_episodes if validation is not None else episodes

    def _fit_head(self, z, labels):
        scaler = StandardScaler().fit(z)
        y = self.classes_[labels]
        if self.head == "svm":
            head = SVC(kernel="rbf", C=self.svm_c, gamma="scale",
                       class_weight=self.head_class_weight, random_state=self.random_state)
        elif self.head == "mlp":
            head = MLPClassifier(hidden_layer_sizes=tuple(self.mlp_hidden), alpha=self.mlp_alpha,
                                 max_iter=2000, random_state=self.random_state)
        else:
            head = PrototypeClassifier()
        return scaler, head.fit(scaler.transform(z), y)

    # -- scikit-learn API ----------------------------------------------------------------

    def fit(self, X, y):
        if self.loss not in LOSSES or self.head not in HEADS:
            raise ValueError(f"loss must be one of {LOSSES} and head one of {HEADS}")
        if (self.episodes < 1 or self.eval_every < 1 or self.patience < 1 or self.n_support < 1
                or self.n_query < 1 or self.n_layers < 1 or not 0 < self.crop_min <= 1
                or self.noise_std < 0 or self.mix_weight < 0 or self.n_bands < 0):
            raise ValueError("episodes, eval_every, patience, n_support, n_query, n_layers must "
                             "be >= 1; 0 < crop_min <= 1; noise_std, mix_weight, n_bands >= 0")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        x = self._array(X, check_shape=False)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(x) or len(x) == 0:
            raise ValueError("One label is required per image")
        self.classes_, labels = np.unique(y, return_inverse=True)
        labels = labels.astype(np.int64)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required")
        self.image_shape_ = x.shape[1:]
        self.n_features_in_ = int(np.prod(x.shape[1:]))
        self.best_episodes_ = self.episodes
        if self.val_fraction:
            tr, va = train_test_split(np.arange(len(y)), test_size=self.val_fraction,
                                     stratify=labels, random_state=self.random_state)
            self.inner_train_indices_, self.inner_val_indices_ = tr, va
            bands, mean, std = self._fit_preprocessing(x[tr], labels[tr])
            self.selection_bands_, self.selection_mean_, self.selection_std_ = bands, mean, std
            _, self.best_episodes_ = self._train(
                self._apply(x[tr], bands, mean, std), labels[tr], self.episodes,
                (self._apply(x[va], bands, mean, std), labels[va]))
        self.bands_, self.mean_, self.std_ = self._fit_preprocessing(x, labels)
        prepared = self._apply(x, self.bands_, self.mean_, self.std_)
        self.net_, _ = self._train(prepared, labels, self.best_episodes_)
        z = self._embed_all(self.net_.to(self.device), prepared).cpu().numpy()
        self.net_.cpu()
        self.train_embedding_, self.train_labels_ = z, labels
        self.scaler_, self.head_ = self._fit_head(z, labels)
        return self

    def refit_head(self, head):
        """Copy of this fitted estimator with another head trained on the same embedding.

        The embedding network is shared (not copied), so evaluating the SVM, MLP
        and prototype heads costs one network fit per fold.
        """
        check_is_fitted(self, "train_embedding_")
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}")
        other = copy.copy(self)
        other.head = head
        other.scaler_, other.head_ = other._fit_head(self.train_embedding_, self.train_labels_)
        return other

    def transform(self, X):
        """Embeddings ``(n_samples, d_model)`` of the fitted network (no augmentation)."""
        check_is_fitted(self, "net_")
        prepared = self._apply(self._array(X), self.bands_, self.mean_, self.std_)
        z = self._embed_all(self.net_.to(self.device), prepared).cpu().numpy()
        self.net_.cpu()
        return z

    def attention_maps(self, X, batch_size=32):
        """Class-token attention over the token grid, ``(n_samples, pool[0], pool[1])``."""
        check_is_fitted(self, "net_")
        prepared = self._apply(self._array(X), self.bands_, self.mean_, self.std_)
        net = self.net_.to(self.device).eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(prepared), batch_size):
                out.append(net(self._to_tensor(prepared[start:start + batch_size]))["attention"]
                           .cpu().numpy())
        self.net_.cpu()
        return np.concatenate(out) if out else np.zeros((0, *self.net_.pool_shape), np.float32)

    def predict_proba(self, X):
        check_is_fitted(self, "head_")
        z = self.transform(X)
        if len(z) == 0:
            return np.empty((0, len(self.classes_)))
        return predict_scores(self.head_, self.scaler_.transform(z), self.classes_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
