"""Inference service for fake news prediction.

Production features:
    - Structured Prediction dataclass with label, confidence, metadata
    - Decision function calibration for LinearSVC (no native predict_proba)
    - Empty-text and edge-case handling
    - Monitoring hooks (prediction logging, latency tracking)
    - Thread-safe (stateless after __init__)

Usage:
    from src.inference.service import InferenceService
    svc = InferenceService()
    result = svc.predict("Some news article text here...")
    print(result.label, result.confidence)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.exceptions import ModelLoadError, PredictionError, PreprocessingError
from src.features.cleaner import TextCleaner
from src.utils.logger import get_logger
from src.utils.versioning import get_latest_model

logger = get_logger(__name__)


@dataclass(frozen=True)
class Prediction:
    """Immutable prediction result with full audit trail."""
    label: str             # "FAKE", "REAL", or "UNKNOWN"
    confidence: float      # 0.0 to 1.0
    raw_label: int         # 1 = fake, 0 = real, -1 = unknown
    latency_ms: float      # end-to-end prediction time
    model_version: str     # which model artifact produced this
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "raw_label": self.raw_label,
            "latency_ms": self.latency_ms,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
        }


class InferenceService:
    """Stateful prediction service wrapping model + vectorizer + cleaner.

    Thread-safe after construction. All mutable state is loaded
    in __init__ and never modified.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        vectorizer_path: Path | str | None = None,
    ) -> None:
        try:
            if model_path and vectorizer_path:
                m_path, v_path = Path(model_path), Path(vectorizer_path)
            else:
                m_path, v_path = get_latest_model()

            self._model_version = m_path.stem  # e.g., "baseline_v3"
            logger.info("Loading model: %s", m_path.name)

            self.model: BaseEstimator = joblib.load(m_path)
            self.vectorizer: TfidfVectorizer = joblib.load(v_path)
            self.cleaner = TextCleaner()

            # Validate loaded artifacts
            if not hasattr(self.model, "predict"):
                raise ModelLoadError(
                    "Loaded object has no predict() method",
                    context={"path": str(m_path)},
                )

            logger.info(
                "InferenceService ready | model=%s | vectorizer_features=%s",
                self._model_version,
                getattr(self.vectorizer, "max_features", "?"),
            )

        except (ModelLoadError, FileNotFoundError):
            raise
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize InferenceService: {e}",
                context={"model_path": str(model_path), "vectorizer_path": str(vectorizer_path)},
            ) from e

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, text: str) -> Prediction:
        """Clean → vectorize → predict → return structured result.

        Handles edge cases:
            - Empty text after cleaning → UNKNOWN with 0.0 confidence
            - Models without predict_proba (LinearSVC) → sigmoid-calibrated decision_function
        """
        start = time.perf_counter()

        try:
            cleaned = self.cleaner.clean(text)
        except Exception as e:
            raise PreprocessingError(
                f"Text cleaning failed: {e}",
                context={"text_length": len(text)},
            ) from e

        if not cleaned.strip():
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Empty text after cleaning (original length: %d)", len(text))
            return Prediction(
                label="UNKNOWN",
                confidence=0.0,
                raw_label=-1,
                latency_ms=round(elapsed, 2),
                model_version=self._model_version,
            )

        try:
            vector = self.vectorizer.transform([cleaned])
            raw = int(self.model.predict(vector)[0])

            # Confidence extraction
            confidence = self._extract_confidence(vector, raw)

            label = "FAKE" if raw == 1 else "REAL"
            elapsed = (time.perf_counter() - start) * 1000

            return Prediction(
                label=label,
                confidence=round(confidence, 4),
                raw_label=raw,
                latency_ms=round(elapsed, 2),
                model_version=self._model_version,
            )

        except Exception as e:
            raise PredictionError(
                f"Prediction failed: {e}",
                context={"text_length": len(text), "cleaned_length": len(cleaned)},
            ) from e

    def _extract_confidence(self, vector: Any, raw_label: int) -> float:
        """Extract calibrated confidence from model output.

        - predict_proba models (LR, NB): use max probability
        - decision_function models (LinearSVC): apply sigmoid calibration
        - fallback: return 0.5
        """
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vector)[0]
            return float(np.max(proba))

        if hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(vector)[0]
            # Platt scaling approximation
            return float(1.0 / (1.0 + np.exp(-decision)))

        return 0.5

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        """Predict multiple texts. Preserves input order."""
        return [self.predict(t) for t in texts]

    def health_check(self) -> dict[str, Any]:
        """Verify model is loaded and functional."""
        try:
            test_result = self.predict("This is a health check test article for verification.")
            return {
                "status": "healthy",
                "model_version": self._model_version,
                "test_prediction": test_result.label,
                "test_latency_ms": test_result.latency_ms,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_version": self._model_version,
            }