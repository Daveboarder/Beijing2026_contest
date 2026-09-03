"""LIBS 2026 steel aging-state classification contest."""

from .config import PROJECT_ROOT, Config
from .data import build_cache, load_index, load_labels, load_shots, load_wavelength
from .depth import (
    AUGMENTATIONS,
    DIAGNOSTIC_LINES_NM,
    ENCODINGS,
    encode_sample,
    line_indices,
    log_bin_edges,
)
from .evaluation import CVResult, cross_validate_model, predict_scores, summarize
from .features import FeatureSet, aggregate_predictions, build_features
from .images import ImageSet, build_images
from .models import PLSDA, BinnedPCA, build_depth_zoo, build_model_zoo, get_model
from .preprocessing import Preprocessor

try:
    from .cnn import DepthSpectrumCNN, SpectrumCNN
except ImportError:  # torch is an optional extra
    DepthSpectrumCNN = None  # type: ignore[misc, assignment]
    SpectrumCNN = None  # type: ignore[misc, assignment]

__all__ = [
    "AUGMENTATIONS",
    "DIAGNOSTIC_LINES_NM",
    "ENCODINGS",
    "PROJECT_ROOT",
    "BinnedPCA",
    "Config",
    "DepthSpectrumCNN",
    "ImageSet",
    "Preprocessor",
    "FeatureSet",
    "PLSDA",
    "SpectrumCNN",
    "CVResult",
    "build_cache",
    "build_depth_zoo",
    "build_features",
    "build_images",
    "build_model_zoo",
    "aggregate_predictions",
    "cross_validate_model",
    "encode_sample",
    "get_model",
    "line_indices",
    "load_index",
    "log_bin_edges",
    "load_labels",
    "load_shots",
    "load_wavelength",
    "predict_scores",
    "summarize",
]

__version__ = "0.1.0"
