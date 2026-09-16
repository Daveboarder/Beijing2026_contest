"""Estimate the class distribution of the 60 test samples.

    python scripts/30_test_prior.py

Uses the outputs of scripts/29 (out-of-fold predictions and test submissions
for pca_mlp and svm_linear, with original and corrected training labels).

Three estimators, from naive to adjusted:

* classify and count -- the test prediction counts as they are;
* adjusted classify and count (black-box shift estimation) -- the out-of-fold
  confusion matrix ``C[pred, true] = P(pred | true)`` maps a test class mix
  ``pi`` to the expected prediction mix ``C @ pi``; ``pi`` is solved on the
  simplex. Works for any classifier, including the SVM without probabilities.
  95 % intervals from a bootstrap over test samples and over training samples
  (which re-estimates C);
* EM prior adjustment (Saerens et al., 2002) -- pca_mlp only, since it needs
  probabilities. They are first temperature-calibrated on the out-of-fold
  predictions, because an uncalibrated MLP under-reacts to a prior shift.

Assumes label shift only: P(spectrum | class) is the same in train and test.
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402
from sklearn.base import clone  # noqa: E402

from libs2026 import Config, Preprocessor, build_features, get_model, plotting  # noqa: E402
from libs2026.evaluation import predict_scores  # noqa: E402

CLASSES = np.array([1, 2, 3, 4, 5])
PCOLS = [f"p{c}" for c in CLASSES]
K = len(CLASSES)


def confusion(y, pred):
    """Column-normalised ``C[pred, true]``."""
    m = np.zeros((K, K))
    np.add.at(m, (np.searchsorted(CLASSES, pred), np.searchsorted(CLASSES, y)), 1)
    return m / np.maximum(m.sum(axis=0, keepdims=True), 1)


def solve_simplex(c, q):
    """``argmin ||C pi - q||^2`` with ``pi >= 0, sum(pi) = 1``."""
    res = minimize(lambda p: np.sum((c @ p - q) ** 2), np.full(K, 1 / K), method="SLSQP",
                   bounds=[(0, 1)] * K, constraints={"type": "eq", "fun": lambda p: p.sum() - 1})
    return np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()


def bbse(oof, test_pred, n_boot, rng):
    """Adjusted classify-and-count estimate with a two-level bootstrap interval."""
    q = np.bincount(np.searchsorted(CLASSES, test_pred), minlength=K) / len(test_pred)
    est = solve_simplex(confusion(oof["y"], oof["pred"]), q)
    samples = oof["sample_id"].unique()
    by_sample = {s: g for s, g in oof.groupby("sample_id")}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(samples, len(samples), replace=True)
        o = pd.concat([by_sample[s] for s in pick])
        t = rng.choice(test_pred, len(test_pred), replace=True)
        qb = np.bincount(np.searchsorted(CLASSES, t), minlength=K) / len(t)
        boots.append(solve_simplex(confusion(o["y"], o["pred"]), qb))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return est, lo, hi


def temperature(oof):
    """Temperature that minimises the out-of-fold negative log-likelihood."""
    logp = np.log(np.clip(oof[PCOLS].to_numpy(), 1e-12, 1))
    idx = np.searchsorted(CLASSES, oof["y"].to_numpy())

    def nll(t):
        z = logp / t
        z = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -z[np.arange(len(z)), idx].mean()

    return minimize_scalar(nll, bounds=(0.2, 20), method="bounded").x


def calibrate(p, t):
    z = np.log(np.clip(p, 1e-12, 1)) / t
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def em_prior(p, train_prior, n_iter=1000, tol=1e-8):
    """Saerens EM: re-weight posteriors by prior ratio until the test prior converges."""
    pi = train_prior.copy()
    for _ in range(n_iter):
        w = p * (pi / train_prior)
        w /= w.sum(axis=1, keepdims=True)
        new = w.mean(axis=0)
        if np.abs(new - pi).max() < tol:
            break
        pi = new
    return pi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    rng = np.random.default_rng(args.seed)
    corrections = pd.read_csv(cfg.metrics_dir / "label_corrections.csv").set_index("sample_id")["corrected_label"]
    rows, em_rows = [], []
    features = None

    for model in ("pca_mlp", "svm_linear"):
        for labels in ("original", "corrected"):
            oof = pd.read_csv(cfg.predictions_dir / f"oof_{model}_g{args.n_groups}_{labels}_labels.csv")
            sub = pd.read_csv(cfg.submissions_dir / f"predictions_{model}_g{args.n_groups}_{labels}_labels.csv")
            test_pred = sub["predicted_label"].to_numpy()
            n_test = len(test_pred)
            counts = np.bincount(np.searchsorted(CLASSES, test_pred), minlength=K)
            train_counts = np.bincount(np.searchsorted(CLASSES, oof.drop_duplicates("sample_id")["y"]), minlength=K)
            est, lo, hi = bbse(oof, test_pred, args.n_boot, rng)
            base = {"model": model, "labels": labels}
            rows.append({**base, "method": "training set (count)", **dict(zip(CLASSES, train_counts))})
            rows.append({**base, "method": "classify and count", **dict(zip(CLASSES, counts))})
            rows.append({**base, "method": "adjusted count (BBSE)", **dict(zip(CLASSES, est * n_test)),
                         **{f"lo{c}": v * n_test for c, v in zip(CLASSES, lo)},
                         **{f"hi{c}": v * n_test for c, v in zip(CLASSES, hi)}})

            if model == "pca_mlp":
                if features is None:
                    pre = Preprocessor.from_config(cfg)
                    pre.normalization_reference = "shot"
                    features = build_features(cfg, pre, n_groups=args.n_groups, bin_factor=1, encoding="mean",
                                              augment="blocks", n_jobs=8)
                train, test = features.subset("train"), features.subset("test")
                y = train.y.astype(int)
                if labels == "corrected":
                    y = np.array([corrections.get(s, v) for s, v in zip(train.sample_ids, y)])
                fitted = clone(get_model(model)).fit(train.X, y)
                p_rows = predict_scores(fitted, test.X, CLASSES)
                p = pd.DataFrame(p_rows, columns=PCOLS).assign(sample_id=test.sample_ids) \
                    .groupby("sample_id")[PCOLS].mean().to_numpy()
                t = temperature(oof)
                train_prior = train_counts / train_counts.sum()
                pi_raw = em_prior(p, train_prior)
                pi_cal = em_prior(calibrate(p, t), train_prior)
                rows.append({**base, "method": "EM, raw MLP probabilities", **dict(zip(CLASSES, pi_raw * n_test))})
                rows.append({**base, "method": f"EM, temperature-calibrated (T={t:.2f})",
                             **dict(zip(CLASSES, pi_cal * n_test))})
                em_rows.append((labels, t))

    table = pd.DataFrame(rows)
    out = cfg.metrics_dir / f"test_prior_g{args.n_groups}.csv"
    table.to_csv(out, index=False)

    show = table.copy()
    for c in CLASSES:
        show[f"L{c}"] = [f"{v:.1f}" if pd.isna(show.at[i, f"lo{c}"]) else
                         f"{v:.1f} [{show.at[i, f'lo{c}']:.1f}-{show.at[i, f'hi{c}']:.1f}]"
                         for i, v in show[c].items()] if f"lo{c}" in show else show[c]
    with pd.option_context("display.width", 250):
        print(show[["model", "labels", "method"] + [f"L{c}" for c in CLASSES]].to_string(index=False))
    print(f"-> {out}")

    # Figure: BBSE with intervals for every model x label set, plus EM for pca_mlp.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, model in zip(axes, ("pca_mlp", "svm_linear")):
        sel = table[(table["model"] == model) & (table["method"] != "training set (count)")]
        n = len(sel)
        width = 0.8 / n
        for i, (_, r) in enumerate(sel.iterrows()):
            x = np.arange(K) + (i - (n - 1) / 2) * width
            vals = r[list(CLASSES)].to_numpy(float)
            err = None
            if not pd.isna(r.get("lo1", np.nan)):
                err = np.vstack([vals - r[[f"lo{c}" for c in CLASSES]].to_numpy(float),
                                 r[[f"hi{c}" for c in CLASSES]].to_numpy(float) - vals])
            ax.bar(x, vals, width, yerr=err, capsize=2, label=f"{r['labels']}: {r['method']}",
                   hatch="//" if r["labels"] == "corrected" else None, alpha=0.85)
        train_counts = table[(table["model"] == model) & (table["method"] == "training set (count)")
                             & (table["labels"] == "original")][list(CLASSES)].to_numpy(float)[0] / 2
        ax.plot(np.arange(K), train_counts, "k_", ms=30, mew=2, label="training mix scaled to 60")
        ax.axhline(12, color="grey", ls=":", label="uniform (12 each)")
        ax.set_xticks(np.arange(K), [f"level {c}" for c in CLASSES])
        ax.set_title(model)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("estimated number of test samples")
    fig.suptitle("Estimated class mix of the 60 test samples (95 % bootstrap intervals for BBSE)")
    fig.tight_layout()
    plotting.save(fig, cfg.figures_dir / f"test_prior_g{args.n_groups}.png")


if __name__ == "__main__":
    main()
