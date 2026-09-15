"""Ablate CLS vs PCA under the same folds as ae_pca_mlp (diagnostic).

uv run --extra cnn python scripts/diagnose_ae_pca_mlp.py --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from libs2026.autotransformer import AutoDepthClassifier
from libs2026.config import Config
from libs2026.depth_transformer import build_depth_sequences
from libs2026.features import build_features
from libs2026.preprocessing import Preprocessor


def metrics(y, p, classes):
    pred = classes[p.argmax(1)]
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
    )


def mean_abs_label_corr(X, y, classes):
    Yoh = np.eye(len(classes))[np.searchsorted(classes, y)]
    Xz = (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)
    Yz = (Yoh - Yoh.mean(0)) / np.maximum(Yoh.std(0), 1e-6)
    return float(np.mean(np.abs(Xz.T @ Yz) / max(len(y) - 1, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    fusion = cfg.get("ae_pca_mlp", {})
    auto = cfg.get("autotransformer", {})
    depth = cfg.get("depth_transformer", {})
    preparation = {
        k: fusion.get(k, auto.get(k, depth.get(k, d)))
        for k, d in (("bin_factor", 4), ("surface_shots", 20), ("late_bin", 4))
    }
    ae_hparams = {
        k: fusion.get(k, auto.get(k, default))
        for k, default in dict(
            d_model=64, n_layers=2, n_heads=4, ff_dim=128, dropout=0.1,
            lambda_recon=1.0, epochs=100, patience=15, batch_size=8,
            lr=0.0003, weight_decay=0.001,
        ).items()
    }
    X_depth, index = build_depth_sequences(cfg, **preparation)
    y = index["label"].to_numpy(dtype=int)
    ids = index["sample_id"].to_numpy()
    feat_cfg = cfg.get("features", {})
    features = build_features(
        cfg, Preprocessor.from_config(cfg), n_groups=1, bin_factor=1,
        encoding=fusion.get("encoding", "mean_lines"),
        n_bins=fusion.get("n_bins", feat_cfg.get("n_bins", 8)),
        augment=fusion.get("augment", feat_cfg.get("augment", "surface")),
        n_jobs=8,
    )
    train = features.subset("train")
    order = {sid: i for i, sid in enumerate(train.sample_ids)}
    X_classical = train.X[[order[sid] for sid in ids]]
    classes = np.unique(y)
    cv = cfg["cv"]
    n_splits = args.folds or cv["n_splits"]
    repeats = args.repeats or cv["n_repeats"]
    split_seed = cv["random_state"]
    bounds = cfg["data"]["channel_bounds"]
    n_spectral = sum((b - a) // preparation["bin_factor"]
                     for a, b in zip(bounds[:-1], bounds[1:]))
    ae_params = dict(
        n_spectral=n_spectral, n_features=X_depth.shape[-1],
        n_shots=cfg["data"]["n_shots"],
        surface_shots=preparation["surface_shots"], late_bin=preparation["late_bin"],
        **ae_hparams,
    )
    seed = args.seed
    splits = [
        list(StratifiedGroupKFold(n_splits, shuffle=True,
                                  random_state=split_seed + r)
             .split(X_depth, y, ids))
        for r in range(repeats)
    ]

    def make_mlp():
        return MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-4, max_iter=2000,
            random_state=seed,
        )

    names = ["pca_only", "cls_only", "fused", "cls_linear"]
    repeat_rows = {n: [] for n in names}
    diag_rows = []
    oof = {n: np.zeros((repeats, len(y), len(classes))) for n in names}

    for repeat, folds in enumerate(splits):
        fold_p = {n: np.zeros((len(y), len(classes))) for n in names}
        for fold, (tr, te) in enumerate(folds):
            ae = AutoDepthClassifier(**ae_params, device=args.device, random_state=seed)
            ae.fit(X_depth[tr], y[tr])
            cls_tr = ae.transform(X_depth[tr])
            cls_te = ae.transform(X_depth[te])

            c_scaler = StandardScaler().fit(X_classical[tr])
            pca = PCA(n_components=30, random_state=seed).fit(
                c_scaler.transform(X_classical[tr]))
            pca_tr = pca.transform(c_scaler.transform(X_classical[tr]))
            pca_te = pca.transform(c_scaler.transform(X_classical[te]))

            blocks = {
                "pca_only": (pca_tr, pca_te),
                "cls_only": (cls_tr, cls_te),
                "fused": (
                    np.concatenate([cls_tr, pca_tr], axis=1),
                    np.concatenate([cls_te, pca_te], axis=1),
                ),
            }
            fitted = {}
            for name, (Xtr, Xte) in blocks.items():
                sc = StandardScaler().fit(Xtr)
                model = make_mlp().fit(sc.transform(Xtr), y[tr])
                fitted[name] = (sc, model, Xtr, Xte)
                proba = model.predict_proba(sc.transform(Xte))
                fold_p[name][te] = proba[:, [list(model.classes_).index(c) for c in classes]]

            sc, model, fused_tr, _ = fitted["fused"]
            W = model.coefs_[0]
            d_cls = cls_tr.shape[1]
            cls_energy = float(np.linalg.norm(W[:d_cls]) ** 2)
            pca_energy = float(np.linalg.norm(W[d_cls:]) ** 2)
            total = cls_energy + pca_energy + 1e-12

            sc_cls = StandardScaler().fit(cls_tr)
            lr = LogisticRegression(max_iter=2000, random_state=seed)
            lr.fit(sc_cls.transform(cls_tr), y[tr])
            proba = lr.predict_proba(sc_cls.transform(cls_te))
            fold_p["cls_linear"][te] = proba[
                :, [list(lr.classes_).index(c) for c in classes]]

            a = StandardScaler().fit_transform(cls_tr)
            b = StandardScaler().fit_transform(pca_tr)
            cross = (a.T @ b) / max(len(tr) - 1, 1)
            means = np.stack([cls_tr[y[tr] == c].mean(0) for c in classes])

            def nn_acc(X, yt):
                d = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)
                return float(np.mean(classes[d.argmin(1)] == yt))

            row = dict(
                repeat=repeat, fold=fold, best_epoch=ae.best_epoch_,
                cls_dims=int(cls_tr.shape[1]), pca_dims=int(pca_tr.shape[1]),
                mlp_cls_weight_frac=cls_energy / total,
                mlp_pca_weight_frac=pca_energy / total,
                cls_pca_cross_fro=float(np.linalg.norm(cross, ord="fro")),
                cls_label_corr=mean_abs_label_corr(cls_tr, y[tr], classes),
                pca_label_corr=mean_abs_label_corr(pca_tr, y[tr], classes),
                cls_nn_train_acc=nn_acc(cls_tr, y[tr]),
                cls_nn_test_acc=nn_acc(cls_te, y[te]),
                pca_var_explained=float(pca.explained_variance_ratio_.sum()),
            )
            for name, (sc, model, Xtr, Xte) in fitted.items():
                row[f"{name}_train_acc"] = float(
                    np.mean(model.predict(sc.transform(Xtr)) == y[tr]))
                row[f"{name}_test_acc"] = float(
                    np.mean(model.predict(sc.transform(Xte)) == y[te]))
            diag_rows.append(row)
            print(
                f"repeat={repeat} fold={fold} epoch={ae.best_epoch_} "
                f"pca={row['pca_only_test_acc']:.3f} "
                f"cls={row['cls_only_test_acc']:.3f} "
                f"fused={row['fused_test_acc']:.3f} "
                f"Wcls={row['mlp_cls_weight_frac']:.2f}",
                flush=True,
            )

        for name in names:
            oof[name][repeat] = fold_p[name]
            repeat_rows[name].append(metrics(y, fold_p[name], classes))

    summary = []
    for name in names:
        rows = repeat_rows[name]
        summary.append(dict(
            model=name,
            accuracy=float(np.mean([r["accuracy"] for r in rows])),
            accuracy_std=float(np.std([r["accuracy"] for r in rows])),
            balanced_accuracy=float(np.mean([r["balanced_accuracy"] for r in rows])),
            macro_f1=float(np.mean([r["macro_f1"] for r in rows])),
        ))

    pca_ens = oof["pca_only"].mean(0)
    fused_ens = oof["fused"].mean(0)
    cls_ens = oof["cls_only"].mean(0)
    pca_pred = classes[pca_ens.argmax(1)]
    fused_pred = classes[fused_ens.argmax(1)]
    agree = dict(
        pca_correct=float(np.mean(pca_pred == y)),
        fused_correct=float(np.mean(fused_pred == y)),
        cls_correct=float(np.mean(classes[cls_ens.argmax(1)] == y)),
        fused_fixes_pca=int(np.sum((fused_pred == y) & (pca_pred != y))),
        fused_breaks_pca=int(np.sum((fused_pred != y) & (pca_pred == y))),
        both_correct=int(np.sum((fused_pred == y) & (pca_pred == y))),
        both_wrong=int(np.sum((fused_pred != y) & (pca_pred != y))),
    )
    diag = pd.DataFrame(diag_rows)
    agg = dict(
        mean_mlp_cls_weight_frac=float(diag["mlp_cls_weight_frac"].mean()),
        mean_cls_pca_cross_fro=float(diag["cls_pca_cross_fro"].mean()),
        mean_cls_label_corr=float(diag["cls_label_corr"].mean()),
        mean_pca_label_corr=float(diag["pca_label_corr"].mean()),
        mean_cls_nn_train=float(diag["cls_nn_train_acc"].mean()),
        mean_cls_nn_test=float(diag["cls_nn_test_acc"].mean()),
        mean_gap_train_test_cls=float(
            (diag["cls_only_train_acc"] - diag["cls_only_test_acc"]).mean()),
        mean_gap_train_test_pca=float(
            (diag["pca_only_train_acc"] - diag["pca_only_test_acc"]).mean()),
        mean_gap_train_test_fused=float(
            (diag["fused_train_acc"] - diag["fused_test_acc"]).mean()),
    )
    out = cfg.results_dir / "ae_pca_mlp" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "ablation_summary.csv", index=False)
    diag.to_csv(out / "fold_diagnostics.csv", index=False)
    payload = dict(summary=summary, agreement=agree, aggregates=agg, seed=seed)
    (out / "aggregates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
