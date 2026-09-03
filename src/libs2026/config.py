"""Project paths and YAML configuration handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    """Configuration container with resolved absolute paths."""

    raw: dict[str, Any] = field(default_factory=dict)
    source: Path = DEFAULT_CONFIG

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        path = Path(path) if path is not None else DEFAULT_CONFIG
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(raw=raw, source=path.resolve())

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def _path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p

    @property
    def raw_data_dir(self) -> Path:
        return self._path("raw_data")

    @property
    def cache_dir(self) -> Path:
        return self._path("cache")

    @property
    def results_dir(self) -> Path:
        return self._path("results")

    @property
    def submissions_dir(self) -> Path:
        return self._path("submissions")

    @property
    def train_dir(self) -> Path:
        return self.raw_data_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.raw_data_dir / "test"

    @property
    def label_file(self) -> Path:
        return self.raw_data_dir / "train_label.csv"

    @property
    def sample_submission(self) -> Path:
        return self.raw_data_dir / "sample_submission" / "predictions.csv"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.results_dir / "metrics"

    @property
    def models_dir(self) -> Path:
        return self.results_dir / "models"

    @property
    def predictions_dir(self) -> Path:
        return self.results_dir / "predictions"

    def ensure_dirs(self) -> None:
        for d in (
            self.cache_dir,
            self.results_dir,
            self.figures_dir,
            self.metrics_dir,
            self.models_dir,
            self.predictions_dir,
            self.submissions_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
