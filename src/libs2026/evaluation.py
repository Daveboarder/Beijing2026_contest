"""Cross-validation and scoring.

With only 120 labelled samples the evaluation protocol matters more than the
models. Two rules are enforced here: folds are stratified by aging level, and
all rows belonging to one physical sample always stay in the same fold, so a
model can never see part of a sample during training and be tested on the rest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class CVResult:
    """Out-of-fold predictions and scores for one model."""

    name: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    kappa: float
    accuracy_std: float
    fit_seconds: float
    oof: pd.DataFrame = field(repr=False, default=None)
    confusion: np.ndarray = field(repr=False, default=None)

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "accuracy": self.accuracy,
            "accuracy_std": self.accuracy_std,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "kappa": self.kappa,
            "fit_seconds": self.fit_seconds,
        }


def predict_scores(model, X, classes) -> np.ndarray:
    """Per-class scores that sum to one, whatever the estimator exposes.

    Models without ``predict_proba`` (the SVMs) are handled through a softmax
    over their one-vs-rest decision values. The result is not a calibrated
    probability, but it is monotone in the decision value, which is all that
    the argmax and the averaging over shot blocks require.
    """
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
    else:
        scores = np.asarray(model.decision_function(X), dtype=float)
        scores = scores - scores.max(axis=1, keepdims=True)
        proba = np.exp(scores)
        proba /= proba.sum(axis=1, keepdims=True)
    # Align columns with the global class order (a fold may miss a rare class).
    classes = list(classes)
    out = np.zeros((X.shape[0], len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, classes.index(c)] = proba[:, j]
    return out


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    name: str = "model",
    n_splits: int = 5,
    n_repeats: int = 4,
    random_state: int = 42,
) -> CVResult:
    """Repeated stratified group k-fold CV, scored at the sample level."""
    classes = np.unique(y)
    repeat_frames, repeat_acc = [], []
    started = time.perf_counter()

    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + repeat
        )
        proba = np.zeros((X.shape[0], len(classes)))
        for train_idx, test_idx in splitter.split(X, y, groups):
            fitted = clone(model).fit(X[train_idx], y[train_idx])
            proba[test_idx] = predict_scores(fitted, X[test_idx], classes)

        frame = pd.DataFrame(proba, columns=[f"p{int(c)}" for c in classes])
        frame["sample_id"] = sample_ids
        frame["y_true"] = y
        frame["repeat"] = repeat
        # Rows of the same sample are averaged: one vote per physical sample.
        agg = frame.groupby(["sample_id", "y_true", "repeat"], as_index=False).mean()
        prob_cols = [f"p{int(c)}" for c in classes]
        agg["y_pred"] = classes[agg[prob_cols].to_numpy().argmax(axis=1)]
        repeat_frames.append(agg)
        repeat_acc.append(accuracy_score(agg["y_true"], agg["y_pred"]))

    oof = pd.concat(repeat_frames, ignore_index=True)
    elapsed = time.perf_counter() - started
    return CVResult(
        name=name,
        accuracy=float(np.mean(repeat_acc)),
        accuracy_std=float(np.std(repeat_acc)),
        balanced_accuracy=float(balanced_accuracy_score(oof["y_true"], oof["y_pred"])),
        macro_f1=float(f1_score(oof["y_true"], oof["y_pred"], average="macro")),
        kappa=float(cohen_kappa_score(oof["y_true"], oof["y_pred"])),
        fit_seconds=elapsed,
        oof=oof,
        confusion=confusion_matrix(oof["y_true"], oof["y_pred"], labels=classes),
    )


def summarize(results: list[CVResult]) -> pd.DataFrame:
    """Leaderboard sorted by accuracy."""
    frame = pd.DataFrame([r.as_row() for r in results])
    return frame.sort_values("accuracy", ascending=False).reset_index(drop=True)
