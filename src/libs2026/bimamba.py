"""Bidirectional Mamba over wavelength and over depth, fused adaptively.

Input is the depth sequence built by :func:`libs2026.depth_transformer.prepare_sample`:
one row per physical sample, shaped ``(samples, depth tokens, spectral bins +
channel intensities)``. Two pathways read it:

* the **spectral pathway** pools the depth tokens into physical regions
  (surface, aged layer, bulk), cuts every detector channel into non-overlapping
  patches of ``patch`` bins and runs a bidirectional selective state-space
  (Mamba) block along wavelength;
* the **depth pathway** embeds every depth token with a small per-channel
  convolutional stem and runs a bidirectional Mamba block along depth.

An adaptive fusion module combines them: a few learned queries attend over the
tokens and summaries of both paths (cross-path attention whose cost is linear
in the sequence length) and a content-dependent gate weights the two path
summaries. Three heads (spectral, depth, fused) are averaged into the final
logits; the single-path heads also receive auxiliary losses. The selective
scan is written in plain PyTorch (chunked, linear time), so the model runs on
CPU for tests and needs no compiled kernels.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .depth_transformer import DepthTransformerClassifier, depth_windows

VARIANTS = ("full", "spectral", "depth", "concat")


# ----------------------------------------------------------------------------
# Geometry and positional helpers
# ----------------------------------------------------------------------------


def depth_geometry(n_shots=200, surface_shots=20, late_bin=4, bulk_start=140):
    """Per-token (centre, width) coordinates and width-weighted region pooling matrix.

    Returns ``(coordinates (L, 2) float32, region_weights (3, L) float32)``; the
    three regions are the surface shots, the intermediate (aged) layer up to
    ``bulk_start`` and the bulk. Each row of ``region_weights`` sums to one.
    """
    windows = depth_windows(n_shots, surface_shots, late_bin)
    centers = np.array([(a + b - 1) / 2 for a, b in windows], dtype=np.float64)
    widths = np.array([b - a for a, b in windows], dtype=np.float64)
    if bulk_start not in [a for a, _ in windows]:
        raise ValueError("bulk_start must align with a depth-window boundary")
    regions = np.where(centers < surface_shots, 0, np.where(centers < bulk_start, 1, 2))
    if len(np.unique(regions)) != 3:
        raise ValueError("Early, intermediate and bulk regions must all be nonempty")
    coordinates = np.column_stack([centers / max(n_shots - 1, 1), widths / n_shots])
    weights = np.zeros((3, len(windows)), dtype=np.float32)
    for region in range(3):
        mask = regions == region
        weights[region, mask] = widths[mask] / widths[mask].sum()
    return coordinates.astype(np.float32), weights


def sinusoidal_positions(n_positions, d_model):
    """Fixed sin/cos encoding ``(n_positions, d_model)``; ``d_model`` must be even."""
    if n_positions < 1 or d_model < 2 or d_model % 2:
        raise ValueError("Need n_positions >= 1 and an even d_model >= 2")
    position = torch.arange(n_positions, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    table = torch.zeros(n_positions, d_model)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div)
    return table


# ----------------------------------------------------------------------------
# Selective scan
# ----------------------------------------------------------------------------


def selective_scan_reference(u, decay, c):
    """Naive sequential scan, the test oracle.

    ``u`` and ``decay`` are ``(B, L, D, N)``, ``c`` is ``(B, L, N)``; the state
    recursion is ``h_t = decay_t * h_{t-1} + u_t`` and ``y_t = <h_t, c_t>``.
    Returns ``y`` shaped ``(B, L, D)``.
    """
    batch, length, d_inner, d_state = u.shape
    h = u.new_zeros(batch, d_inner, d_state)
    outputs = []
    for t in range(length):
        h = decay[:, t] * h + u[:, t]
        outputs.append((h * c[:, t, None, :]).sum(-1))
    return torch.stack(outputs, dim=1)


def selective_scan(u, log_decay, c, chunk=16):
    """Chunked linear-time scan of ``h_t = exp(log_decay_t) * h_{t-1} + u_t``.

    Two levels: a loop of ``chunk`` steps run for all chunks at once from a zero
    state, then a loop over the chunks to carry the entering state, and a
    vectorised combination ``state = local + (cumulative decay) * carry``. The
    number of sequential steps is ``chunk + ceil(L / chunk)`` instead of ``L``.
    ``log_decay`` must be non-positive, which the caller guarantees, so the
    exponentials cannot overflow. ``chunk=1`` is the plain sequential scan.
    """
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    batch, length, d_inner, d_state = u.shape
    pad = (-length) % chunk
    if pad:
        # Zero input and zero log-decay (state kept); the padded outputs are dropped.
        u = F.pad(u, (0, 0, 0, 0, 0, pad))
        log_decay = F.pad(log_decay, (0, 0, 0, 0, 0, pad))
    n_chunks = (length + pad) // chunk
    u = u.reshape(batch, n_chunks, chunk, d_inner, d_state)
    log_decay = log_decay.reshape(batch, n_chunks, chunk, d_inner, d_state)
    decay = torch.exp(log_decay)
    prefix = torch.exp(torch.cumsum(log_decay, dim=2))   # product of decays inside the chunk
    h = u.new_zeros(batch, n_chunks, d_inner, d_state)
    local = []
    for t in range(chunk):
        h = decay[:, :, t] * h + u[:, :, t]
        local.append(h)
    local = torch.stack(local, dim=2)                     # (B, nC, chunk, D, N)
    carry = u.new_zeros(batch, d_inner, d_state)
    carries = []
    for i in range(n_chunks):
        carries.append(carry)
        carry = prefix[:, i, -1] * carry + local[:, i, -1]
    states = local + prefix * torch.stack(carries, dim=1)[:, :, None]
    states = states.reshape(batch, n_chunks * chunk, d_inner, d_state)[:, :length]
    return (states * c[:, :, None, :]).sum(-1)


# ----------------------------------------------------------------------------
# Mamba blocks
# ----------------------------------------------------------------------------


class S6(nn.Module):
    """One direction of a selective state-space layer (Mamba S6 recurrence)."""

    def __init__(self, d_inner, d_state=16, dt_rank=4, d_conv=4, dt_min=1e-3, dt_max=0.1,
                 chunk=16):
        super().__init__()
        self.d_state, self.dt_rank, self.chunk = d_state, dt_rank, chunk
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        # Step sizes start log-uniform in [dt_min, dt_max]; the bias is the inverse softplus.
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
                                  .repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        """``x`` is ``(B, L, D)``; returns the same shape."""
        length = x.shape[1]
        x = F.silu(self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2))   # causal
        dt, b, c = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                                      # (B, L, D) > 0
        a = -torch.exp(self.A_log.clamp(max=20.0))                                # (D, N) < 0
        u = delta.unsqueeze(-1) * b.unsqueeze(2) * x.unsqueeze(-1)                # (B, L, D, N)
        log_decay = delta.unsqueeze(-1) * a                                       # <= 0
        y = selective_scan(u, log_decay, c, self.chunk)
        return y + x * self.D


class BiMambaBlock(nn.Module):
    """Pre-norm residual block: shared projections, forward and backward S6 branches."""

    def __init__(self, d_model, d_state=16, expand=2, dropout=0.1, chunk=16):
        super().__init__()
        d_inner = expand * d_model
        dt_rank = max(1, math.ceil(d_model / 16))
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.ssm_fwd = S6(d_inner, d_state, dt_rank, chunk=chunk)
        self.ssm_bwd = S6(d_inner, d_state, dt_rank, chunk=chunk)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """``x`` is ``(B, L, d_model)``; returns the same shape."""
        h, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        y = self.ssm_fwd(h) + self.ssm_bwd(h.flip(1)).flip(1)
        return x + self.dropout(self.out_proj(y * F.silu(z)))


# ----------------------------------------------------------------------------
# Pathways, fusion and the full network
# ----------------------------------------------------------------------------


class SpectralPath(nn.Module):
    """Wavelength-ordered patch tokens of the region-pooled spectra -> BiMamba -> summary."""

    def __init__(self, channel_widths, patch, n_regions, d_model, d_state=16, n_layers=1,
                 dropout=0.2, chunk=16):
        super().__init__()
        self.channel_widths = tuple(int(w) for w in channel_widths)
        if patch < 1 or any(w % patch for w in self.channel_widths):
            raise ValueError(f"patch={patch} must divide every channel width {self.channel_widths}")
        self.patch, self.n_regions = patch, n_regions
        counts = [w // patch for w in self.channel_widths]
        self.n_tokens = sum(counts)
        self.register_buffer("positions", sinusoidal_positions(self.n_tokens, d_model))
        self.register_buffer("token_channel", torch.repeat_interleave(
            torch.arange(len(self.channel_widths)), torch.tensor(counts)))
        self.patch_embed = nn.Linear(patch * n_regions, d_model)
        self.channel_embed = nn.Parameter(torch.zeros(len(self.channel_widths), d_model))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, dropout=dropout, chunk=chunk) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)
        self.intensity = nn.Linear(n_regions * len(self.channel_widths), d_model)

    def tokenize(self, regions):
        """``regions`` is ``(B, R, n_spectral)``; returns tokens ``(B, n_tokens, d_model)``."""
        batch = regions.shape[0]
        pieces, offset = [], 0
        for width in self.channel_widths:
            block = regions[:, :, offset:offset + width]
            block = block.reshape(batch, self.n_regions, width // self.patch, self.patch)
            pieces.append(block.permute(0, 2, 1, 3).reshape(batch, width // self.patch, -1))
            offset += width
        tokens = self.patch_embed(torch.cat(pieces, dim=1))
        return tokens + self.positions[None] + self.channel_embed[self.token_channel][None]

    def forward(self, regions, region_intensity):
        """Returns ``(tokens (B, n_tokens, d), summary (B, d))``."""
        tokens = self.dropout(self.tokenize(regions))
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        attention = torch.softmax(self.score(tokens), dim=1)
        summary = 0.5 * (tokens.mean(1) + (attention * tokens).sum(1))
        return tokens, summary + self.intensity(region_intensity.flatten(1))


class DepthPath(nn.Module):
    """Per-token convolutional spectral stem -> BiMamba along depth -> region summary."""

    def __init__(self, channel_widths, coordinates, region_weights, d_model, d_state=16,
                 n_layers=1, dropout=0.2, chunk=16, stem_bins=4):
        super().__init__()
        self.channel_widths = tuple(int(w) for w in channel_widths)
        if stem_bins < 1:
            raise ValueError("stem_bins must be >= 1")
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3), nn.GroupNorm(4, 16), nn.GELU(),
                nn.AvgPool1d(2),
                nn.Conv1d(16, 32, 5, padding=2), nn.GroupNorm(4, 32), nn.GELU(),
                nn.AdaptiveAvgPool1d(stem_bins), nn.Flatten(),
            ) for _ in self.channel_widths
        ])
        self.projection = nn.Linear(len(self.channel_widths) * 32 * stem_bins, d_model)
        self.position = nn.Linear(2, d_model)
        self.intensity = nn.Linear(len(self.channel_widths), d_model, bias=False)
        self.register_buffer("coordinates", torch.as_tensor(coordinates, dtype=torch.float32))
        self.register_buffer("region_weights",
                             torch.as_tensor(region_weights, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, dropout=dropout, chunk=chunk) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.summary = nn.Linear(self.region_weights.shape[0] * d_model, d_model)

    def forward(self, x):
        """``x`` is ``(B, L, n_spectral + n_channels)``; returns ``(tokens, summary)``."""
        batch, depth, _ = x.shape
        offset, features = 0, []
        for width, branch in zip(self.channel_widths, self.branches):
            channel = x[..., offset:offset + width].reshape(batch * depth, 1, width)
            features.append(branch(channel).reshape(batch, depth, -1))
            offset += width
        z = self.projection(torch.cat(features, dim=-1))
        z = z + self.position(self.coordinates)[None] + self.intensity(x[..., offset:])
        z = self.dropout(z)
        for layer in self.layers:
            z = layer(z)
        tokens = self.norm(z)
        pooled = torch.einsum("rl,blf->brf", self.region_weights, tokens)
        return tokens, self.summary(pooled.flatten(1))


class AdaptiveFusion(nn.Module):
    """Learned queries attend over both token sets; a gate weights the two summaries."""

    def __init__(self, d_model, n_queries=4, heads=4, dropout=0.1):
        super().__init__()
        if n_queries < 1 or d_model % heads:
            raise ValueError("Need n_queries >= 1 and d_model divisible by the number of heads")
        self.queries = nn.Parameter(torch.zeros(n_queries, d_model))
        self.source = nn.Parameter(torch.zeros(4, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        nn.init.trunc_normal_(self.source, std=0.02)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(2 * d_model, 2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens_s, tokens_d, h_s, h_d):
        """Returns ``(fused (B, d), gate (B, 2))``; cost is linear in the token counts."""
        memory = torch.cat([
            tokens_s + self.source[0], tokens_d + self.source[1],
            (h_s + self.source[2])[:, None], (h_d + self.source[3])[:, None],
        ], dim=1)
        queries = self.queries[None].expand(h_s.shape[0], -1, -1)
        cross, _ = self.attention(queries, memory, memory, need_weights=False)
        gate = torch.softmax(self.gate(torch.cat([h_s, h_d], dim=-1)), dim=-1)
        fused = gate[:, :1] * h_s + gate[:, 1:] * h_d + cross.mean(1)
        return self.norm(fused), gate


class BiMambaNet(nn.Module):
    """Spectral and depth BiMamba pathways, adaptive fusion and a three-head logit ensemble."""

    def __init__(self, channel_widths, coordinates, region_weights, n_classes, d_model=64,
                 d_state=16, n_layers=1, patch=11, dropout=0.2, variant="full", n_queries=4,
                 chunk=16, stem_bins=4):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if d_model % 4:
            raise ValueError("d_model must be divisible by 4")
        self.variant = variant
        self.channel_widths = tuple(int(w) for w in channel_widths)
        self.n_spectral = sum(self.channel_widths)
        self.register_buffer("region_weights",
                             torch.as_tensor(region_weights, dtype=torch.float32))
        n_regions = self.region_weights.shape[0]
        self.spectral = None if variant == "depth" else SpectralPath(
            self.channel_widths, patch, n_regions, d_model, d_state, n_layers, dropout, chunk)
        self.depth = None if variant == "spectral" else DepthPath(
            self.channel_widths, coordinates, region_weights, d_model, d_state, n_layers,
            dropout, chunk, stem_bins)
        self.fusion = AdaptiveFusion(d_model, n_queries, 4, dropout) if variant == "full" else None
        self.concat = (nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
                       if variant == "concat" else None)

        def head():
            return nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 32), nn.GELU(),
                                 nn.Linear(32, n_classes))

        self.head_spectral = head() if self.spectral is not None else None
        self.head_depth = head() if self.depth is not None else None
        self.head_fused = head() if variant in ("full", "concat") else None

    def outputs(self, x):
        """Per-head logits, the fused embedding and (for ``full``) the gate, as a dict."""
        expected = (self.region_weights.shape[1], self.n_spectral + len(self.channel_widths))
        if x.dim() != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected input (batch, {expected[0]}, {expected[1]}), "
                             f"got {tuple(x.shape)}")
        out, logits = {}, []
        tokens_s = tokens_d = h_s = h_d = None
        if self.spectral is not None:
            regions = torch.einsum("rl,blf->brf", self.region_weights, x[..., :self.n_spectral])
            intensity = torch.einsum("rl,blf->brf", self.region_weights, x[..., self.n_spectral:])
            tokens_s, h_s = self.spectral(regions, intensity)
            out["spectral"] = self.head_spectral(h_s)
            logits.append(out["spectral"])
        if self.depth is not None:
            tokens_d, h_d = self.depth(x)
            out["depth"] = self.head_depth(h_d)
            logits.append(out["depth"])
        if self.variant == "full":
            fused, out["gate"] = self.fusion(tokens_s, tokens_d, h_s, h_d)
        elif self.variant == "concat":
            fused = self.concat(torch.cat([h_s, h_d], dim=-1))
        else:
            fused = h_s if self.variant == "spectral" else h_d
        if self.head_fused is not None:
            out["fused"] = self.head_fused(fused)
            logits.append(out["fused"])
        out["embedding"] = fused
        out["final"] = torch.stack(logits).mean(0)
        return out

    def forward(self, x):
        return self.outputs(x)["final"]


def augment_batch(x, noise_std=0.0, token_dropout=0.0):
    """Training-time augmentation of a standardised batch ``(B, L, F)``.

    Gaussian noise, then "depth jitter": every token is replaced by its shallower
    neighbour with probability ``token_dropout`` (the first token is never
    replaced), which keeps region means realistic.
    """
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    if token_dropout > 0 and x.shape[1] > 1:
        keep = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= token_dropout
        keep[:, 0] = True
        x = torch.where(keep, x, torch.cat([x[:, :1], x[:, :-1]], dim=1))
    return x


# ----------------------------------------------------------------------------
# scikit-learn estimator
# ----------------------------------------------------------------------------


class BiMambaClassifier(DepthTransformerClassifier):
    """Dual-pathway BiMamba with the inner-validation / refit protocol of the parent.

    ``fit`` receives one row per physical sample, shaped
    ``(samples, depth tokens, sum(channel_widths) + len(channel_widths))``.
    Input validation, standardisation (one scalar per detector channel), the
    inner epoch selection and ``predict_proba`` are inherited; the network,
    augmentation and the auxiliary head losses are defined here.
    """

    def __init__(self, channel_widths=(1023, 1023, 1023), n_shots=200, surface_shots=20,
                 late_bin=4, bulk_start=140, d_model=64, d_state=16, n_layers=1, patch=11,
                 chunk=16, n_queries=4, stem_bins=4, dropout=0.2, variant="full",
                 lambda_aux=0.3, noise_std=0.05, token_dropout=0.1, epochs=100, patience=15,
                 val_fraction=0.2, batch_size=8, lr=0.0003, weight_decay=0.001,
                 random_state=42, device="cpu"):
        self.channel_widths = channel_widths
        self.n_shots = n_shots
        self.surface_shots = surface_shots
        self.late_bin = late_bin
        self.bulk_start = bulk_start
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.patch = patch
        self.chunk = chunk
        self.n_queries = n_queries
        self.stem_bins = stem_bins
        self.dropout = dropout
        self.variant = variant
        self.lambda_aux = lambda_aux
        self.noise_std = noise_std
        self.token_dropout = token_dropout
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.lambda_aux < 0 or self.noise_std < 0 or not 0 <= self.token_dropout < 1:
            raise ValueError("lambda_aux and noise_std must be >= 0, token_dropout in [0, 1)")
        if (self.patch < 1 or self.chunk < 1 or self.n_layers < 1 or self.d_state < 1
                or self.n_queries < 1 or self.stem_bins < 1 or self.d_model % 4):
            raise ValueError("patch, chunk, n_layers, d_state, n_queries and stem_bins must be "
                             ">= 1 and d_model divisible by 4")
        return super().fit(X, y)

    def _net(self):
        coordinates, region_weights = depth_geometry(
            self.n_shots, self.surface_shots, self.late_bin, self.bulk_start)
        torch.manual_seed(self.random_state)
        return BiMambaNet(
            self.channel_widths, coordinates, region_weights, len(self.classes_),
            d_model=self.d_model, d_state=self.d_state, n_layers=self.n_layers,
            patch=self.patch, dropout=self.dropout, variant=self.variant,
            n_queries=self.n_queries, chunk=self.chunk, stem_bins=self.stem_bins,
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
                xb, yb = xb.to(self.device), yb.to(self.device)
                xb = augment_batch(xb, self.noise_std, self.token_dropout)
                optimizer.zero_grad(set_to_none=True)
                out = net.outputs(xb)
                loss = criterion(out["final"], yb)
                if self.lambda_aux > 0 and "spectral" in out and "depth" in out:
                    loss = loss + self.lambda_aux * (criterion(out["spectral"], yb)
                                                     + criterion(out["depth"], yb))
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
                        total += F.cross_entropy(logits, target, reduction="sum").item()
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

    def _collect(self, X):
        check_is_fitted(self, "net_")
        x = (self._array(X) - self.mean_) / self.std_
        collected = {}
        if len(x) == 0:
            return collected
        net = self.net_.to(self.device).eval()
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                batch = torch.from_numpy(x[start:start + self.batch_size]).to(self.device)
                out = net.outputs(batch)
                for key, value in out.items():
                    collected.setdefault(key, []).append(value.cpu())
        net.cpu()
        return {key: torch.cat(parts).numpy() for key, parts in collected.items()}

    def predict_heads(self, X):
        """Per-head class probabilities (spectral, depth, fused, final) and the gate."""
        collected = self._collect(X)
        result = {}
        for key, value in collected.items():
            if key == "embedding":
                continue
            if key == "gate":
                result[key] = value
            else:
                result[key] = torch.softmax(torch.from_numpy(value), -1).numpy()
        return result

    def transform(self, X):
        """Fused embedding ``(n_samples, d_model)`` of the fitted network."""
        collected = self._collect(X)
        if not collected:
            return np.empty((0, self.d_model), dtype=np.float32)
        return collected["embedding"]
