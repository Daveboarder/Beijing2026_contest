"""Does the Fe plasma temperature / electron density help ``pca_mlp``?

    python scripts/20_boltzmann_temperature.py      # produces the descriptors
    python scripts/21_benchmark_physics_features.py --n-repeats 10

The spectrum rows are the best classical setting so far (``mean`` encoding,
four shot blocks per sample, per-shot L2 normalisation). Plasma descriptors
are per sample and copied onto each of its rows; they bypass the spectral PCA
(see ``models.build_pca_mlp_with_extras``). Every variant runs on the same
folds, so accuracy differences are paired per CV repeat.

The descriptors were derived without labels (line selection and response fit
use all 180 samples), so they do not leak fold information.
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from libs2026 import Config, Preprocessor, build_features, cross_validate_model, get_model, summarize
from libs2026.models import RANDOM_STATE, build_pca_mlp_with_extras

FEATURE_SETS = {
    "T": ["T_boltz"],
    "ne": ["log_ne"],
    "T_ne": ["T_boltz", "log_ne"],
    "T_ne_depth": ["T_boltz", "log_ne", "T_bin*"],
}


def load_physics(cfg, tag: str) -> pd.DataFrame:
    """Per-sample plasma descriptors indexed by ``sample_id``."""
    base = cfg.results_dir / "boltzmann" / tag
    samples = pd.read_csv(base / "sample_temperatures.csv").set_index("sample_id")
    depth = pd.read_csv(base / "depth_temperatures.csv")
    per_bin = depth.pivot(index="sample_id", columns="depth_bin", values="T_boltz")
    per_bin.columns = [f"T_bin{int(c)}" for c in per_bin.columns]
    out = samples[["T_boltz"]].assign(log_ne=np.log10(samples["ne"])).join(per_bin)
    if out.isna().any().any():
        raise ValueError(f"Missing descriptors:\n{out[out.isna().any(axis=1)]}")
    return out


def expand(columns, available):
    out = []
    for c in columns:
        out += [a for a in available if a.startswith(c[:-1])] if c.endswith("*") else [c]
    return out


def per_repeat_accuracy(oof: pd.DataFrame) -> np.ndarray:
    return np.array([accuracy_score(g["y_true"], g["y_pred"]) for _, g in oof.groupby("repeat")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--boltzmann-tag", default="fe")
    parser.add_argument("--n-groups", type=int, default=4)
    parser.add_argument("--bin-factor", type=int, default=1)
    parser.add_argument("--encoding", default="mean")
    parser.add_argument("--augment", default="blocks")
    parser.add_argument("--normalization-reference", default="shot")
    parser.add_argument("--n-pca", type=int, default=30)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--tag", default="physics")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv = cfg["cv"]
    n_repeats = args.n_repeats or cv["n_repeats"]
    pre = Preprocessor.from_config(cfg)
    pre.normalization_reference = args.normalization_reference

    features = build_features(cfg, pre, n_groups=args.n_groups, bin_factor=args.bin_factor,
                              encoding=args.encoding, augment=args.augment, n_jobs=args.n_jobs)
    train = features.subset("train")
    y = train.y.astype(int)
    physics = load_physics(cfg, args.boltzmann_tag)
    extras_all = physics.loc[train.sample_ids]
    n_spectral = train.X.shape[1]
    print(f"spectral matrix {train.X.shape}; descriptors: {list(physics.columns)}")

    def run(name, model, X):
        result = cross_validate_model(model, X, y, train.groups, train.sample_ids, name=name,
                                      n_splits=cv["n_splits"], n_repeats=n_repeats,
                                      random_state=cv["random_state"])
        print(f"{name:24s} acc={result.accuracy:.3f}+-{result.accuracy_std:.3f}  "
              f"bal_acc={result.balanced_accuracy:.3f}  macroF1={result.macro_f1:.3f}  "
              f"({result.fit_seconds:.0f}s)", flush=True)
        return result

    results = [run("pca_mlp", get_model("pca_mlp", args.n_pca), train.X)]
    for set_name, cols in FEATURE_SETS.items():
        cols = expand(cols, physics.columns)
        X = np.hstack([train.X, extras_all[cols].to_numpy(np.float32)])
        results.append(run(f"pca_mlp+{set_name}", build_pca_mlp_with_extras(n_spectral, args.n_pca), X))

    # How much do the descriptors carry on their own?
    cols = expand(FEATURE_SETS["T_ne_depth"], physics.columns)
    logreg = Pipeline([("scale", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                                  random_state=RANDOM_STATE))])
    results.append(run("physics_only_logreg", logreg, extras_all[cols].to_numpy(np.float32)))

    summary = summarize(results)
    base = per_repeat_accuracy(results[0].oof)
    rows = []
    for r in results:
        acc = per_repeat_accuracy(r.oof)
        diff = acc - base
        p = stats.wilcoxon(acc, base).pvalue if np.any(diff != 0) else 1.0
        rows.append({"model": r.name, "delta_vs_pca_mlp": diff.mean(), "delta_std": diff.std(),
                     "repeats_better": int((diff > 0).sum()), "repeats_worse": int((diff < 0).sum()),
                     "wilcoxon_p": p})
    summary = summary.merge(pd.DataFrame(rows), on="model")
    summary.insert(1, "n_repeats", n_repeats)
    out = cfg.metrics_dir / f"benchmark_{args.tag}.csv"
    summary.to_csv(out, index=False)
    for r in results:
        r.oof.to_csv(cfg.predictions_dir / f"oof_{args.tag}_{r.name.replace('+', '_')}.csv", index=False)
    print()
    print(summary.drop(columns=["kappa", "fit_seconds"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
