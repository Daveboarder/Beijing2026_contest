"""Benchmark the CNN + ViT embedding of depth-spectrum images, or fit its recipe for submission.

uv run --extra cnn python scripts/34_cnn_vit.py benchmark --device cuda --tag initial
uv run --extra cnn python scripts/34_cnn_vit.py benchmark --device cuda --variants proto ce
uv run --extra cnn python scripts/34_cnn_vit.py predict --recipe PATH --device cuda --head svm

One sample is one image: rows are depth (shots 21-200 of 200, the unstable first
20 omitted, averaged in groups of four -> 45 rows) and columns are the 1533
wavelength bins of the three detector channels. The embedding network of
``libs2026.cnn_vit`` is fitted once per fold and every requested head (RBF SVM,
MLP, nearest prototype) is refitted on that embedding, so the summary lists one
model per ``{variant}+{head}``. Variants: ``proto`` (episodic prototype loss),
``proto_mix`` (plus the article's attention-guided query mixing) and ``ce``
(cross-entropy on a linear head); they override the config's loss / mix_queries.

``benchmark --smoke`` shrinks the grid to 2 folds, 1 repeat, seed 42 and 20 episodes
(unless the corresponding option is given explicitly) to exercise the whole
pipeline and to read the per-fit time before the full grid. Both label sets, the
original labels and the scripts/29 corrections, are evaluated on identical folds
stratified on the original labels; ``unchanged110_*`` scores only the samples no
correction touched, which is the fair comparison between the two.

Standardisation happens inside the estimator, per wavelength column over
(samples, depth), and must stay that way: the BiMamba experiment showed that a
single scalar per detector channel prevents learning entirely, while per-column
standardisation trains but deep nets still overfit (train 1.0, test 0.42-0.50
against 0.752 for pca_mlp), so expectations for this model are modest.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from importlib import import_module
from pathlib import Path

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from libs2026.cnn_vit import HEADS, CnnVitClassifier, depth_images
from libs2026.config import Config
from libs2026.images import build_images
from libs2026.preprocessing import Preprocessor

VARIANTS = ("proto", "proto_mix", "ce")
# Estimator options fixed by each variant (the config's loss / mix_queries are ignored).
VARIANT_PARAMS = {
    "proto": dict(loss="proto", mix_queries=False),
    "proto_mix": dict(loss="proto", mix_queries=True),
    "ce": dict(loss="ce", mix_queries=False),
}
LABEL_CHOICES = ("original", "corrected", "both")
# The model source and the shared image preparation code, hashed in this order.
SOURCE_FILES = ("cnn_vit.py", "images.py")
# Estimator options read from the cnn_vit config section, with the module defaults.
MODEL_DEFAULTS = dict(
    n_bands=0, channels=(16, 32), d_model=64, pool=(9, 24), n_layers=2, n_heads=4, ff_dim=128,
    dropout=0.1, stride=4, n_support=3, n_query=4, mix_weight=0.5, crop_min=0.8, noise_std=0.0,
    eval_every=20, patience=5, lr=0.001, weight_decay=0.001, svm_c=1.0, mlp_hidden=(128, 64),
    mlp_alpha=0.001,
)
# Options that YAML and JSON carry as lists; the estimator receives them as tuples.
TUPLE_KEYS = ("channels", "pool", "mlp_hidden")
METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1")
SUMMARY_COLUMNS = [
    "model", "labels",
    "all120_accuracy", "all120_accuracy_std", "all120_balanced_accuracy", "all120_macro_f1",
    "unchanged110_accuracy", "unchanged110_accuracy_std", "unchanged110_balanced_accuracy",
    "unchanged110_macro_f1", "n_params", "fit_seconds_mean",
]


def metrics(y, p, classes):
    pred = classes[p.argmax(1)]
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)))


def _preparation(section):
    """Image preparation: wavelength bin factor, omitted surface shots, depth averaging."""
    return {key: int(section.get(key, default))
            for key, default in (("bin_factor", 8), ("drop_shots", 20), ("shot_bin", 4))}


def _with_tuples(params):
    """Copy of the estimator options with the list-valued ones (YAML / JSON) as tuples."""
    out = dict(params)
    for key in TUPLE_KEYS:
        out[key] = tuple(int(v) for v in out[key])
    return out


def _model_params(section, episodes):
    """Estimator options shared by every variant: config values over the module defaults."""
    params = {key: section.get(key, default) for key, default in MODEL_DEFAULTS.items()}
    params["episodes"] = int(episodes)
    return _with_tuples(params)


def _depth_split(images, split, preparation):
    """``(X, ids, y)`` of one split: depth images, sample ids and int labels (None for test)."""
    subset = images.subset(split)
    X = depth_images(subset.X, drop_shots=preparation["drop_shots"],
                     shot_bin=preparation["shot_bin"])
    ids = subset.sample_ids
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Each physical sample must appear exactly once")
    if split != "train":
        return X, ids, None
    y = np.asarray(subset.y, dtype=float)
    if y.shape != (len(ids),) or np.isnan(y).any():
        raise ValueError("Every training sample needs a label")
    return X, ids, y.astype(int)


def _source_hash():
    """SHA-256 of the concatenated bytes of ``SOURCE_FILES`` under ``src/libs2026``."""
    return hashlib.sha256(b"".join(
        (Path(_bootstrap.SRC) / "libs2026" / name).read_bytes() for name in SOURCE_FILES
    )).hexdigest()


def _load_corrections():
    """Consensus label corrections of scripts/29; ``scripts/`` is sys.path[0] when run directly."""
    return dict(import_module("29_corrected_labels").CORRECTIONS)


def _apply_corrections(ids, y_orig, corrections):
    """Corrected label vector and the ``[sample_id, given, new]`` triples actually applied.

    Every correction whose sample is present must quote the label the sample
    currently carries; a mismatch means the corrections were written against
    different data. Corrections for absent samples are ignored.
    """
    position = {str(sid): i for i, sid in enumerate(ids)}
    applied = []
    for sid, (given, new) in corrections.items():
        if sid not in position:
            continue
        current = int(y_orig[position[sid]])
        if current != int(given):
            raise ValueError(f"{sid}: correction expects given label {given}, data has {current}")
        applied.append([str(sid), int(given), int(new)])
    y_corr = np.array([corrections.get(str(sid), (None, y))[1] for sid, y in zip(ids, y_orig)],
                      dtype=int)
    return y_corr, applied


def _label_sets(ids, y_orig, labels, corrections):
    """``([(name, y_fit), ...], changed mask, applied corrections)`` for the label choice.

    The corrections are loaded from scripts/29 only when a corrected label set is
    requested and none were passed in; with ``labels="original"`` and no
    corrections the changed mask is all False (unchanged110 then equals all120).
    """
    if labels not in LABEL_CHOICES:
        raise ValueError(f"labels must be one of {LABEL_CHOICES}, got {labels!r}")
    if corrections is None and labels != "original":
        corrections = _load_corrections()
    if corrections is None:
        y_corr, applied = y_orig.copy(), []
        changed = np.zeros(len(ids), dtype=bool)
    else:
        y_corr, applied = _apply_corrections(ids, y_orig, corrections)
        changed = np.isin(ids, list(corrections))
    label_sets = []
    if labels in ("original", "both"):
        label_sets.append(("original", y_orig))
    if labels in ("corrected", "both"):
        label_sets.append(("corrected", y_corr))
    return label_sets, changed, applied


def _setting(value, smoke_default, default, smoke):
    """Explicit argument first, then the smoke default, then the config default."""
    if value is not None:
        return value
    return smoke_default if smoke else default


def _heads(args, section):
    """Heads to evaluate: the argument, else the config, else every head of the module."""
    heads = getattr(args, "heads", None)
    if heads is None:
        heads = section.get("heads", list(HEADS))
    heads = [str(head) for head in heads]
    if not heads or len(set(heads)) != len(heads) or any(h not in HEADS for h in heads):
        raise ValueError(f"heads must be distinct entries of {HEADS}, got {heads}")
    return heads


def _rows(model, labels, repeat, seed, ids, y_fit, y_given, p, classes):
    """Out-of-fold probability rows of one (model, label set, repeat, seed)."""
    return [dict(model=model, labels=labels, repeat=repeat, seed=seed, sample_id=str(sid),
                 y_true=int(y_fit[i]), y_given=int(y_given[i]),
                 **{f"p{c}": float(p[i, j]) for j, c in enumerate(classes)})
            for i, sid in enumerate(ids)]


def _summary_row(model, labels, repeat_metrics, n_params, fit_seconds):
    row = dict(model=model, labels=labels)
    for subset in ("all120", "unchanged110"):
        values = {key: [m[subset][key] for m in repeat_metrics] for key in METRIC_KEYS}
        row[f"{subset}_accuracy"] = float(np.mean(values["accuracy"]))
        row[f"{subset}_accuracy_std"] = float(np.std(values["accuracy"]))
        row[f"{subset}_balanced_accuracy"] = float(np.mean(values["balanced_accuracy"]))
        row[f"{subset}_macro_f1"] = float(np.mean(values["macro_f1"]))
    row["n_params"] = int(n_params)
    row["fit_seconds_mean"] = float(np.mean(fit_seconds))
    return row


def benchmark(cfg, args, corrections=None):
    section = cfg.get("cnn_vit", {})
    preparation = _preparation(section)
    smoke = bool(getattr(args, "smoke", False))
    variants = list(args.variants)
    if (not variants or len(set(variants)) != len(variants)
            or any(v not in VARIANTS for v in variants)):
        raise ValueError(f"variants must be distinct entries of {VARIANTS}, got {variants}")
    heads = _heads(args, section)
    pre = Preprocessor.from_config(cfg)
    images = build_images(cfg, pre, bin_factor=preparation["bin_factor"], shot_bin=1)
    X, ids, y_orig = _depth_split(images, "train", preparation)
    classes = np.unique(y_orig)
    label_sets, changed, applied = _label_sets(ids, y_orig, args.labels, corrections)
    if changed.all():
        raise ValueError("Every sample carries a correction; nothing is left for unchanged110")
    cv = cfg["cv"]
    n_splits = int(_setting(args.folds, 2, cv["n_splits"], smoke))
    repeats = int(_setting(args.repeats, 1, cv["n_repeats"], smoke))
    seeds = [int(s) for s in _setting(args.seeds, [42], section.get("seeds", [42, 7, 123]), smoke)]
    episodes = int(_setting(args.episodes, 20, section.get("episodes", 600), smoke))
    if n_splits < 2 or repeats < 1 or min(np.unique(y_orig, return_counts=True)[1]) < n_splits:
        raise ValueError("Insufficient samples per class for the requested folds")
    if not seeds or len(set(seeds)) != len(seeds) or episodes < 1:
        raise ValueError("seeds must be distinct and episodes >= 1")
    split_seed = int(cv["random_state"])
    params = _model_params(section, episodes)
    # Folds are computed once, stratified on the ORIGINAL labels, and shared by every
    # variant and label set; they depend only on the repeat, never on the model seed.
    splits = [list(StratifiedGroupKFold(n_splits, shuffle=True,
                                        random_state=split_seed + repeat)
                   .split(X, y_orig, ids)) for repeat in range(repeats)]
    root = cfg.results_dir / "cnn_vit" / args.tag
    root.mkdir(parents=True, exist_ok=False)
    manifest = [dict(sample_id=str(ids[i]), repeat=repeat, fold=fold)
                for repeat, folds in enumerate(splits)
                for fold, (_, te) in enumerate(folds) for i in te]
    pd.DataFrame(manifest).to_csv(root / "folds.csv", index=False)
    input_hash = hashlib.sha256(X.tobytes() + y_orig.tobytes()).hexdigest()
    source_hash = _source_hash()
    environment = {name: importlib.metadata.version(name)
                   for name in ["torch", "numpy", "pandas", "scikit-learn"]}

    total_fits = len(variants) * len(label_sets) * repeats * len(seeds) * n_splits
    print(f"grid: {len(variants)} variants x {len(label_sets)} label sets x {repeats} repeats "
          f"x {len(seeds)} seeds x {n_splits} folds = {total_fits} embedding fits, heads "
          f"{heads} (images: {tuple(X.shape)}, changed samples: {int(changed.sum())}, "
          f"episodes: {episodes}, smoke: {smoke})", flush=True)
    done, elapsed = 0, 0.0
    summary, all_predictions = [], []
    for variant in variants:
        variant_params = {**params, **VARIANT_PARAMS[variant]}
        for labels, y_fit in label_sets:
            episode_records, repeat_metrics, fit_seconds = [], [], []
            n_params = None
            for repeat, folds in enumerate(splits):
                probabilities = {head: [] for head in heads}
                for seed in seeds:
                    p = {head: np.zeros((len(y_fit), len(classes))) for head in heads}
                    for fold, (tr, te) in enumerate(folds):
                        model = CnnVitClassifier(**variant_params, head=heads[0],
                                                 device=args.device, random_state=seed)
                        start = time.perf_counter()
                        model.fit(X[tr], y_fit[tr])
                        fit_s = time.perf_counter() - start
                        # One embedding per fold; the other heads are refitted on it.
                        for head in heads:
                            m = model if head == heads[0] else model.refit_head(head)
                            if not np.array_equal(m.classes_, classes):
                                raise ValueError("Training fold is missing a class")
                            p[head][te] = m.predict_proba(X[te])
                        n_params = sum(int(q.numel()) for q in model.net_.parameters())
                        selected = int(model.best_episodes_)
                        fit_seconds.append(fit_s)
                        done += 1
                        elapsed += fit_s
                        eta_min = elapsed / done * (total_fits - done) / 60
                        episode_records.append(dict(repeat=repeat, seed=seed, fold=fold,
                                                    episodes=selected))
                        print(f"{variant} labels={labels} repeat={repeat} seed={seed} "
                              f"fold={fold} selected_episodes={selected} "
                              f"fit_s={fit_s:.1f} eta_min={eta_min:.1f}", flush=True)
                    for head in heads:
                        probabilities[head].append(p[head])
                        all_predictions.extend(_rows(f"{variant}+{head}", labels, repeat,
                                                     str(seed), ids, y_fit, y_orig, p[head],
                                                     classes))
                by_head = {}
                for head in heads:
                    ensemble = np.mean(probabilities[head], axis=0)
                    by_head[head] = dict(
                        all120=metrics(y_fit, ensemble, classes),
                        unchanged110=metrics(y_fit[~changed], ensemble[~changed], classes))
                    all_predictions.extend(_rows(f"{variant}+{head}", labels, repeat,
                                                 "ensemble", ids, y_fit, y_orig, ensemble,
                                                 classes))
                repeat_metrics.append(by_head)
            rows = [_summary_row(f"{variant}+{head}", labels,
                                 [m[head] for m in repeat_metrics], n_params, fit_seconds)
                    for head in heads]
            summary.extend(rows)
            recipe = dict(
                config=cfg.raw, preparation=preparation,
                model_params=variant_params, heads=heads, seeds=seeds,
                episodes_by_seed={str(seed): int(np.median(
                    [r["episodes"] for r in episode_records if r["seed"] == seed]))
                    for seed in seeds},
                episode_selection=episode_records, repeat_metrics=repeat_metrics,
                cv=dict(n_splits=n_splits, n_repeats=repeats, random_state=split_seed),
                training_ids=[str(sid) for sid in ids], classes=classes.tolist(),
                labels=labels, training_labels=y_fit.tolist(),
                corrections=applied if labels == "corrected" else [],
                n_params=int(n_params),
                input_sha256=input_hash, source_sha256=source_hash,
                environment=environment, device=args.device,
            )
            (root / f"recipe_{variant}_{labels}.json").write_text(
                json.dumps(recipe, indent=2), encoding="utf-8")
            pd.DataFrame(summary, columns=SUMMARY_COLUMNS).to_csv(root / "summary.csv",
                                                                  index=False)
            pd.DataFrame(all_predictions).to_csv(root / "oof.csv", index=False)
            for row in rows:
                print(row, flush=True)
    print(f"Results and refit recipes: {root}")


def predict(args, corrections=None):
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    cfg = Config(raw=recipe["config"])
    # A different host can override locations, but not the evaluated preprocessing.
    if args.config:
        cfg.raw["paths"] = Config.load(args.config).raw["paths"]
    if _source_hash() != recipe["source_sha256"]:
        raise ValueError("Model source changed since benchmarking; generate a new recipe")
    head = str(getattr(args, "head", "svm"))
    if head not in recipe["heads"]:
        raise ValueError(f"head must be one of the benchmarked heads {recipe['heads']}, "
                         f"got {head!r}")
    preparation = _preparation(recipe["preparation"])
    pre = Preprocessor.from_config(cfg)
    images = build_images(cfg, pre, bin_factor=preparation["bin_factor"], shot_bin=1)
    X, ids, y_orig = _depth_split(images, "train", preparation)
    test_x, test_ids, _ = _depth_split(images, "test", preparation)
    digest = hashlib.sha256(X.tobytes() + y_orig.tobytes()).hexdigest()
    if ([str(sid) for sid in ids] != list(recipe["training_ids"])
            or digest != recipe["input_sha256"]):
        raise ValueError("Training data/preprocessing changed since benchmarking")
    if recipe["labels"] == "corrected":
        if corrections is None:
            corrections = _load_corrections()
        y_fit, _ = _apply_corrections(ids, y_orig, corrections)
    elif recipe["labels"] == "original":
        y_fit = y_orig
    else:
        raise ValueError(f"Unknown label set {recipe['labels']!r} in the recipe")
    if y_fit.tolist() != list(recipe["training_labels"]):
        raise ValueError("Label corrections changed since benchmarking; generate a new recipe")
    classes = np.asarray(recipe["classes"])
    probabilities, models = [], []
    for seed in recipe["seeds"]:
        params = _with_tuples(recipe["model_params"])
        params["episodes"] = int(recipe["episodes_by_seed"][str(seed)])
        model = CnnVitClassifier(**params, head=head, val_fraction=0, device=args.device,
                                 random_state=seed)
        model.fit(X, y_fit)
        if not np.array_equal(model.classes_, classes):
            raise ValueError("Fitted classes differ from the recipe classes")
        models.append(model)
        probabilities.append(model.predict_proba(test_x))
    p = np.mean(probabilities, axis=0)
    sample_ids = [str(sid) for sid in test_ids]
    submission = pd.DataFrame(dict(filename=[sid + ".csv" for sid in sample_ids],
                                   predicted_label=classes[p.argmax(1)]))
    template = pd.read_csv(cfg.sample_submission)
    if (submission["filename"].duplicated().any() or len(submission) != len(template)
            or set(submission["filename"]) != set(template["filename"])
            or not submission["predicted_label"].isin([1, 2, 3, 4, 5]).all()):
        raise ValueError("Submission does not match the contest template")
    submission = submission.set_index("filename").loc[template["filename"]].reset_index()
    out = Path(args.output)
    checkpoint = out.with_suffix(".joblib")
    if out.exists() or checkpoint.exists():
        raise FileExistsError("Choose a new output path; refusing to overwrite predictions/model")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(models=models, recipe=recipe, probabilities=p, sample_ids=sample_ids),
                checkpoint)
    submission.to_csv(out, index=False)
    print(f"Submission: {out}; checkpoint: {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--config")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["proto"])
    bench.add_argument("--heads", nargs="+", choices=list(HEADS), default=list(HEADS),
                       help="heads refitted on every embedding; the first one is fitted "
                            "together with the network")
    bench.add_argument("--labels", choices=list(LABEL_CHOICES), default="both")
    bench.add_argument("--seeds", nargs="+", type=int)
    bench.add_argument("--folds", type=int)
    bench.add_argument("--repeats", type=int)
    bench.add_argument("--episodes", type=int)
    bench.add_argument("--smoke", action="store_true",
                       help="2 folds, 1 repeat, seed 42, 20 episodes unless given explicitly")
    bench.add_argument("--tag", default="initial")
    pred = sub.add_parser("predict")
    pred.add_argument("--config")
    pred.add_argument("--device", default="cpu")
    pred.add_argument("--recipe", required=True)
    pred.add_argument("--output", default="submissions/cnn_vit.csv")
    pred.add_argument("--head", choices=list(HEADS), default="svm",
                      help="one of the heads evaluated in the recipe")
    args = parser.parse_args()
    if args.command == "benchmark":
        if Path(args.tag).name != args.tag or args.tag in {".", ".."}:
            parser.error("tag must be a directory name, not a path")
        benchmark(Config.load(args.config), args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
