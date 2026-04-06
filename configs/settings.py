"""Centralized configuration using Pydantic Settings.

All configuration is driven by environment variables with the FND_ prefix.
Settings are validated at startup — fail fast on bad config.

Usage:
    from configs.settings import get_settings
    s = get_settings()
    print(s.tfidf_max_features)

Override via env:
    FND_TFIDF_MAX_FEATURES=20000 python -m src.training.train_classical
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


_ROOT = Path(__file__).resolve().parent.parent


class Environment(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class AppSettings(BaseSettings):
    """Validated, immutable application configuration."""

    # ── Environment ────────────────────────────────────────────────
    env: Environment = Environment.DEV

    # ── Paths ──────────────────────────────────────────────────────
    project_root: Path = _ROOT
    data_dir: Path = _ROOT / "data" / "raw"
    model_dir: Path = _ROOT / "models"
    experiment_dir: Path = _ROOT / "experiments"
    log_dir: Path = _ROOT / "logs"

    # ── Data ───────────────────────────────────────────────────────
    dataset_filename: str = "fake_news_dataset.csv"
    text_column: str = "text"
    label_column: str = "label"
    label_map: dict[str, int] = {"real": 0, "fake": 1}
    min_text_length: int = Field(default=10, ge=1)
    max_null_ratio: float = Field(default=0.1, ge=0.0, le=1.0)

    # ── Feature engineering ────────────────────────────────────────
    tfidf_max_features: int = Field(default=10_000, ge=100, le=500_000)
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = Field(default=2, ge=1)
    tfidf_max_df: float = Field(default=0.95, gt=0.0, le=1.0)
    tfidf_sublinear_tf: bool = True

    # ── Training ───────────────────────────────────────────────────
    cv_folds: int = Field(default=5, ge=2, le=20)
    random_state: int = 42
    scoring_metric: str = "f1"
    test_size: float = Field(default=0.1, gt=0.0, lt=0.5)
    n_jobs: int = Field(default=-1, ge=-1)

    # ── Hyperparameter tuning ──────────────────────────────────────
    tune_max_features: list[int] = [5000, 10000, 20000]
    tune_ngram_ranges: list[tuple[int, int]] = [(1, 1), (1, 2), (1, 3)]
    tune_C_values: list[float] = [0.1, 1.0, 10.0]
    tune_class_weights: list[str | None] = [None, "balanced"]

    # ── BERT ───────────────────────────────────────────────────────
    bert_model_name: str = "bert-base-uncased"
    bert_max_len: int = Field(default=256, ge=32, le=512)
    bert_batch_size: int = Field(default=8, ge=1)
    bert_epochs: int = Field(default=2, ge=1)
    bert_lr: float = Field(default=2e-5, gt=0.0)
    bert_warmup_ratio: float = Field(default=0.1, ge=0.0, le=0.5)
    bert_weight_decay: float = Field(default=0.01, ge=0.0)

    # ── API ────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1)
    cors_origins: list[str] = ["*"]
    api_rate_limit: int = Field(default=100, ge=1)
    api_batch_max: int = Field(default=100, ge=1, le=1000)

    # ── MLflow ─────────────────────────────────────────────────────
    mlflow_experiment: str = "fake_news_detection"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    # ── Monitoring ─────────────────────────────────────────────────
    enable_prediction_logging: bool = True
    prediction_log_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    drift_check_interval: int = Field(default=1000, ge=100)

    # ── Logging ────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_max_bytes: int = Field(default=10_000_000, ge=1_000)
    log_backup_count: int = Field(default=5, ge=1)

    model_config = {"env_prefix": "FND_", "env_file": ".env", "extra": "ignore"}

    # ── Validators ─────────────────────────────────────────────────

    @field_validator("tfidf_ngram_range")
    @classmethod
    def validate_ngram_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError(f"ngram_range min ({v[0]}) > max ({v[1]})")
        if v[0] < 1:
            raise ValueError(f"ngram_range min must be >= 1, got {v[0]}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return v

    @model_validator(mode="after")
    def ensure_directories(self) -> "AppSettings":
        """Create required directories at startup."""
        for d in [self.model_dir, self.experiment_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
        return self

    # ── Derived properties ─────────────────────────────────────────

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / self.dataset_filename

    @property
    def is_production(self) -> bool:
        return self.env == Environment.PRODUCTION


@lru_cache
def get_settings() -> AppSettings:
    """Singleton settings instance — cached after first call."""
    return AppSettings()