"""Raw-spectrum MLP with a learned embedding layer instead of PCA (PyTorch, GPU).

    uv run --extra cnn python scripts/26_embedding_mlp.py --device cuda --n-repeats 10

Same as the best ``pca_mlp`` except for the feature rows and the model:

* preprocessing: outlier-shot repair, no baseline, per-shot per-channel L2;
* rows: 10 consecutive blocks of 20 shots per sample, block-mean spectrum
  (12282 pixels), every row labelled with its sample;
* model (``libs2026.embedding_mlp``): per-pixel standardisation -> linear
  embedding layer (``--embedding`` units, the learned counterpart of PCA
  scores) -> ReLU layers 128 / 64 -> 5 classes; AdamW, dropout, early stopping
  on an inner split that holds out whole samples;
* prediction: row probabilities averaged per sample;
* validation: StratifiedGroupKFold 5 folds grouped by sample, repeated, scored
  at sample level. A variant with log10 n_e from H-alpha appended is run too.

For the blend with the depth-trajectory model use ``scripts/25_blend_oof.py``
on the OOF files written here.
"""

import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from libs2026 import Config, Preprocessor, build_features, cross_validate_model
from libs2026.embedding_mlp import EmbeddingMLPClassifier, resolve_device

CLASSES = np.array([1, 2, 3, 4, 5])


def metrics(g):
    kw = {"labels": CLASSES, "zero_division": 0}
    return {"precision_macro": precision_score(g["y_true"], g["y_pred"], average="macro", **kw),
            "recall_macro": recall_score(g["y_true"], g["y_pred"], average="macro", **kw),
            "f1_macro": f1_score(g["y_true"], g["y_pred"], average="macro", **kw),
            "f1_weighted": f1_score(g["y_true"], g["y_pred"], average="weighted", **kw),
            "accuracy": accuracy_score(g["y_true"], g["y_pred"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto", help="cuda, cpu or auto")
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--embedding", type=int, default=30)
    parser.add_argument("--embedding-activation", choices=("linear", "relu"), default="linear")
    parser.add_argument("--hidden", default="128,64")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--boltzmann-tag", default="fe")
    parser.add_argument("--no-ne", action="store_true", help="skip the n_e variant")
    parser.add_argument("--tag", default="embedding_mlp_torch")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    cv = cfg["cv"]
    device = resolve_device(args.device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    pre = Preprocessor.from_config(cfg)
    pre.normalization_reference = "shot"
    features = build_features(cfg, pre, n_groups=args.n_groups, bin_factor=1, encoding="mean",
                              augment="blocks", n_jobs=8)
    train = features.subset("train")
    y = train.y.astype(int)
    print(f"rows {train.X.shape} ({args.n_groups} blocks per sample)")

    variants = {f"emb{args.embedding}_mlp": train.X}
    if not args.no_ne:
        samples = pd.read_csv(cfg.results_dir / "boltzmann" / args.boltzmann_tag / "sample_temperatures.csv")
        log_ne = np.log10(samples.set_index("sample_id")["ne"]).loc[train.sample_ids].to_numpy(np.float32)
        variants[f"emb{args.embedding}_mlp+ne"] = np.hstack([train.X, log_ne[:, None]])

    model = EmbeddingMLPClassifier(
        embedding=args.embedding, hidden=tuple(int(h) for h in args.hidden.split(",")),
        embedding_activation=args.embedding_activation, dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay, batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, val_fraction=args.val_fraction, rows_per_sample=args.n_groups,
        class_weight=None if args.class_weight == "none" else "balanced", device=str(device),
    )

    rows = []
    for name, X in variants.items():
        started = time.perf_counter()
        # Folds keep each sample's rows together and in order, as rows_per_sample requires.
        result = cross_validate_model(model, X, y, train.groups, train.sample_ids, name=name,
                                      n_splits=cv["n_splits"], n_repeats=args.n_repeats,
                                      random_state=cv["random_state"])
        result.oof.to_csv(cfg.predictions_dir / f"oof_{args.tag}_{name.replace('+', '_')}.csv", index=False)
        per_repeat = pd.DataFrame([metrics(g) for _, g in result.oof.groupby("repeat")])
        row = {"model": name, "n_repeats": args.n_repeats, **per_repeat.mean().to_dict(),
               "accuracy_std": per_repeat["accuracy"].std(ddof=0),
               "minutes": (time.perf_counter() - started) / 60}
        rows.append(row)
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in row.items() if isinstance(v, float)),
              flush=True)

    base = cfg.predictions_dir / "oof_physics_pca_mlp.csv"
    if base.exists():
        per_repeat = pd.DataFrame([metrics(g) for _, g in pd.read_csv(base).groupby("repeat")])
        rows.append({"model": "pca_mlp (4 blocks, reference)", "n_repeats": per_repeat.shape[0],
                     **per_repeat.mean().to_dict(), "accuracy_std": per_repeat["accuracy"].std(ddof=0)})
    table = pd.DataFrame(rows)
    table.insert(1, "settings", str({k: v for k, v in vars(args).items() if k not in ("config", "tag")}))
    out = cfg.metrics_dir / f"benchmark_{args.tag}.csv"
    table.to_csv(out, index=False)
    print(table.drop(columns=["settings"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
