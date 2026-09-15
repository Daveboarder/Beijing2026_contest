"""Probability blend of two models from their out-of-fold predictions.

    python scripts/25_blend_oof.py \
        --a results/predictions/oof_depth_snv_pca_mlp_f1_macro.csv \
        --b results/predictions/oof_physics_pca_mlp.csv --name-a depth_snv --name-b pca_mlp

Repeat ``r`` of file A is paired with repeat ``r`` of file B (only repeats
present in both are used). Both files are out-of-fold for every sample, so a
blend of them is too, even when the two runs used different fold partitions.
Choosing the best weight on these same predictions is optimistic; the
pre-specified 50/50 blend is the honest number.
"""

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from libs2026 import Config

CLASSES = np.array([1, 2, 3, 4, 5])
PROBA = [f"p{c}" for c in CLASSES]


def metrics(y, pred):
    kw = {"labels": CLASSES, "zero_division": 0}
    return {"precision_macro": precision_score(y, pred, average="macro", **kw),
            "recall_macro": recall_score(y, pred, average="macro", **kw),
            "f1_macro": f1_score(y, pred, average="macro", **kw),
            "f1_weighted": f1_score(y, pred, average="weighted", **kw),
            "accuracy": accuracy_score(y, pred)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--name-a", default="a")
    parser.add_argument("--name-b", default="b")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    a, b = pd.read_csv(args.a), pd.read_csv(args.b)
    repeats = sorted(set(a["repeat"]) & set(b["repeat"]))
    frames = {}
    for r in repeats:
        fa = a[a["repeat"] == r].set_index("sample_id").sort_index()
        fb = b[b["repeat"] == r].set_index("sample_id").sort_index()
        if not fa.index.equals(fb.index) or not (fa["y_true"] == fb["y_true"]).all():
            raise ValueError(f"repeat {r}: samples or labels differ between the files")
        frames[r] = (fa, fb)
    y = frames[repeats[0]][0]["y_true"].to_numpy()
    print(f"{len(repeats)} paired repeats, {len(y)} samples")

    # Complementarity: how often each model is right where the other is wrong.
    fa, fb = frames[repeats[0]]
    ok_a = CLASSES[fa[PROBA].to_numpy().argmax(1)] == y
    ok_b = CLASSES[fb[PROBA].to_numpy().argmax(1)] == y
    corr = np.corrcoef(fa[PROBA].to_numpy().ravel(), fb[PROBA].to_numpy().ravel())[0, 1]
    print(f"repeat {repeats[0]}: both right {np.sum(ok_a & ok_b)}, both wrong {np.sum(~ok_a & ~ok_b)}, "
          f"only {args.name_a} right {np.sum(ok_a & ~ok_b)}, only {args.name_b} right {np.sum(~ok_a & ok_b)}; "
          f"probability correlation {corr:.2f}")

    rows = []
    for w in np.round(np.arange(0, 1.01, 0.1), 2):
        per_repeat = []
        for r in repeats:
            fa, fb = frames[r]
            p = w * fa[PROBA].to_numpy() + (1 - w) * fb[PROBA].to_numpy()
            per_repeat.append(metrics(y, CLASSES[p.argmax(1)]))
        per_repeat = pd.DataFrame(per_repeat)
        rows.append({f"w_{args.name_a}": w, **per_repeat.mean().to_dict(),
                     **{f"{m}_std": v for m, v in per_repeat.std(ddof=0).items()}})
    table = pd.DataFrame(rows)
    tag = args.tag or f"{args.name_a}_{args.name_b}"
    out = cfg.metrics_dir / f"blend_{tag}.csv"
    table.to_csv(out, index=False)
    cols = [f"w_{args.name_a}", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "accuracy",
            "accuracy_std"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
