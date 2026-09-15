"""Model zoo: the classifiers compared on this task.

Every entry is a full scikit-learn ``Pipeline`` that starts from the raw
(preprocessed) spectrum, so no information leaks from the validation folds into
scaling or dimensionality reduction.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


class PLSDA(ClassifierMixin, BaseEstimator):
    """Partial least squares discriminant analysis.

    PLS regression against one-hot encoded classes; the regression outputs are
    softmax-normalised so the classifier can expose ``predict_proba``. This is
    the reference chemometric method for LIBS classification.
    """

    def __init__(self, n_components: int = 15, scale: bool = False):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        self._binarizer = LabelBinarizer()
        Y = self._binarizer.fit_transform(y)
        if Y.shape[1] == 1:  # binary case
            Y = np.hstack([1 - Y, Y])
        self.classes_ = self._binarizer.classes_
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self._pls = PLSRegression(n_components=n_comp, scale=self.scale)
        self._pls.fit(X, Y)
        return self

    def decision_function(self, X):
        return self._pls.predict(X)

    def predict_proba(self, X):
        scores = np.asarray(self.decision_function(X), dtype=float)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.asarray(self.decision_function(X)).argmax(axis=1)]


class BinnedPCA(BaseEstimator, TransformerMixin):
    """Project each depth bin onto one shared spectral basis.

    Input rows are several spectra concatenated (the ``depth_*`` encodings).
    Applying PCA to that flat vector wastes components on the noise of the
    single-shot surface bins. Fitting one PCA across all bins instead, then
    transforming each bin through it, keeps the depth trajectory but expresses
    it in a denoised, low-dimensional basis: ``n_bins x n_components`` features
    rather than ``n_bins x 12282``.
    """

    def __init__(self, n_bins: int = 9, n_components: int = 20):
        self.n_bins = n_bins
        self.n_components = n_components

    def _stack(self, X):
        X = np.asarray(X)
        if X.shape[1] % self.n_bins:
            raise ValueError(
                f"{X.shape[1]} features do not divide into {self.n_bins} bins"
            )
        return X.reshape(X.shape[0] * self.n_bins, -1)

    def fit(self, X, y=None):
        n_comp = min(self.n_components, self._stack(X).shape[1])
        self._pca = PCA(n_components=n_comp, random_state=RANDOM_STATE,
                        svd_solver="randomized").fit(self._stack(X))
        return self

    def transform(self, X):
        scores = self._pca.transform(self._stack(X))
        return scores.reshape(np.asarray(X).shape[0], -1)


def _pca(n_components: int) -> PCA:
    return PCA(n_components=n_components, random_state=RANDOM_STATE, svd_solver="randomized")


def build_depth_zoo(n_bins: int = 9, n_components: int = 20) -> dict[str, Pipeline]:
    """Classifiers for the depth-resolved encodings, via a shared bin basis.

    Only meaningful on a ``depth_bins`` feature matrix, where every row is
    ``n_bins`` concatenated spectra.
    """
    return {
        "binpca_lda": Pipeline([
            ("scale", StandardScaler()),
            ("binpca", BinnedPCA(n_bins, n_components)),
            ("clf", LinearDiscriminantAnalysis()),
        ]),
        "binpca_logreg": Pipeline([
            ("scale", StandardScaler()),
            ("binpca", BinnedPCA(n_bins, n_components)),
            ("scale2", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "binpca_svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("binpca", BinnedPCA(n_bins, n_components)),
            ("scale2", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                        class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "binpca_mlp": Pipeline([
            ("scale", StandardScaler()),
            ("binpca", BinnedPCA(n_bins, n_components)),
            ("scale2", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000,
                                  alpha=1e-3, random_state=RANDOM_STATE)),
        ]),
    }


def build_model_zoo(n_pca: int = 30) -> dict[str, Pipeline]:
    """Return the candidate classifiers keyed by a short name."""
    scaler = lambda: StandardScaler()  # noqa: E731 - fresh instance per pipeline

    zoo: dict[str, Pipeline] = {
        "plsda": Pipeline([
            ("scale", scaler()),
            ("clf", PLSDA(n_components=15)),
        ]),
        "pca_lda": Pipeline([
            ("scale", scaler()),
            ("pca", _pca(n_pca)),
            ("clf", LinearDiscriminantAnalysis()),
        ]),
        "pca_logreg": Pipeline([
            ("scale", scaler()),
            ("pca", _pca(n_pca)),
            ("clf", LogisticRegression(max_iter=5000, C=1.0,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        # The SVMs deliberately keep ``probability=False``: Platt scaling was
        # deprecated in scikit-learn 1.9, and evaluation.predict_scores turns the
        # one-vs-rest decision values into per-class scores instead. Only the
        # ranking of those scores matters here, and it avoids the internal
        # cross-validation that probability calibration would cost.
        "pca_svm_rbf": Pipeline([
            ("scale", scaler()),
            ("pca", _pca(n_pca)),
            ("scale2", scaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                        class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "svm_linear": Pipeline([
            ("scale", scaler()),
            ("clf", SVC(kernel="linear", C=1.0,
                        class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "pca_knn": Pipeline([
            ("scale", scaler()),
            ("pca", _pca(n_pca)),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]),
        "pca_gnb": Pipeline([
            ("scale", scaler()),
            ("pca", _pca(n_pca)),
            ("clf", GaussianNB()),
        ]),
        "pca_mlp": Pipeline([
            ("scale", scaler()),
            ("pca", _pca(n_pca)),
            ("scale2", scaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000,
                                  alpha=1e-3, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=750, n_jobs=-1,
                                           class_weight="balanced_subsample",
                                           random_state=RANDOM_STATE)),
        ]),
        "extra_trees": Pipeline([
            ("clf", ExtraTreesClassifier(n_estimators=750, n_jobs=-1,
                                         class_weight="balanced",
                                         random_state=RANDOM_STATE)),
        ]),
        "hist_gbdt": Pipeline([
            ("pca", _pca(n_pca)),
            ("clf", HistGradientBoostingClassifier(max_iter=300,
                                                   learning_rate=0.1,
                                                   random_state=RANDOM_STATE)),
        ]),
    }
    return zoo


def build_pca_mlp_with_extras(n_spectral: int, n_pca: int = 30,
                              hidden_layer_sizes=(128, 64), alpha: float = 1e-3) -> Pipeline:
    """``pca_mlp`` with extra descriptors appended after the PCA.

    Columns ``[:n_spectral]`` are the spectrum and go through scaling + PCA as
    in ``pca_mlp``; the remaining columns (plasma temperature, electron
    density, ...) bypass the PCA, where a handful of variables would be lost
    among thousands of pixels, and join the scores before the MLP.
    """
    return Pipeline([
        ("split", ColumnTransformer([
            ("spectrum", Pipeline([("scale", StandardScaler()), ("pca", _pca(n_pca))]),
             slice(0, n_spectral)),
            ("extras", "passthrough", slice(n_spectral, None)),
        ])),
        ("scale2", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000,
                              alpha=alpha, random_state=RANDOM_STATE)),
    ])


def get_model(name: str, n_pca: int = 30) -> Pipeline:
    zoo = build_model_zoo(n_pca)
    if name not in zoo:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(zoo)}")
    return clone(zoo[name])
