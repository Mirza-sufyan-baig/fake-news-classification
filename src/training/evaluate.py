"""Model evaluation module with comprehensive metrics and error analysis.

Generates:
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix
    - Confidence calibration analysis
    - Error analysis (false positives, false negatives with examples)
    - Statistical significance testing between models

Usage:
    from src.training.evaluate import ModelEvaluator
    evaluator = ModelEvaluator()
    report = evaluator.evaluate_model(model, vectorizer, X_test, y_test)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Structured evaluation output."""
    accuracy: float
    f1: float
    precision: float
    recall: float
    auc_roc: float | None
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]
    n_samples: int
    false_positive_examples: list[str]
    false_negative_examples: list[str]
    confidence_stats: dict[str, float] | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Accuracy:  {self.accuracy:.4f}",
            f"F1:        {self.f1:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
        ]
        if self.auc_roc is not None:
            lines.append(f"AUC-ROC:   {self.auc_roc:.4f}")
        lines.append(f"Samples:   {self.n_samples}")
        lines.append(f"FP examples: {len(self.false_positive_examples)}")
        lines.append(f"FN examples: {len(self.false_negative_examples)}")
        return "\n".join(lines)


class ModelEvaluator:
    """Comprehensive model evaluation with error analysis."""

    def __init__(self, max_error_examples: int = 10) -> None:
        self.max_examples = max_error_examples

    def evaluate(
        self,
        model: BaseEstimator,
        X_test_vec: Any,
        y_test: np.ndarray | pd.Series,
        X_test_raw: pd.Series | list[str] | None = None,
    ) -> EvaluationReport:
        """Full evaluation suite on test data.

        Args:
            model: Fitted sklearn estimator.
            X_test_vec: Vectorized test features (sparse matrix).
            y_test: True labels.
            X_test_raw: Original text for error analysis (optional).
        """
        y_pred = model.predict(X_test_vec)
        y_test_arr = np.array(y_test)

        # Core metrics
        acc = accuracy_score(y_test_arr, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test_arr, y_pred, average="binary")
        cm = confusion_matrix(y_test_arr, y_pred).tolist()
        report = classification_report(y_test_arr, y_pred, output_dict=True)

        # AUC-ROC (if model supports probability)
        auc = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test_vec)[:, 1]
            try:
                auc = roc_auc_score(y_test_arr, proba)
            except ValueError:
                pass

        # Confidence analysis
        conf_stats = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test_vec)
            max_conf = proba.max(axis=1)
            conf_stats = {
                "mean": round(float(max_conf.mean()), 4),
                "std": round(float(max_conf.std()), 4),
                "min": round(float(max_conf.min()), 4),
                "p10": round(float(np.percentile(max_conf, 10)), 4),
                "p50": round(float(np.percentile(max_conf, 50)), 4),
                "p90": round(float(np.percentile(max_conf, 90)), 4),
                "below_60_pct": round(float((max_conf < 0.6).mean()), 4),
            }

        # Error analysis
        fp_examples: list[str] = []
        fn_examples: list[str] = []

        if X_test_raw is not None:
            texts = list(X_test_raw) if not isinstance(X_test_raw, list) else X_test_raw
            for i, (true, pred) in enumerate(zip(y_test_arr, y_pred)):
                if true == 0 and pred == 1 and len(fp_examples) < self.max_examples:
                    fp_examples.append(texts[i][:300])
                elif true == 1 and pred == 0 and len(fn_examples) < self.max_examples:
                    fn_examples.append(texts[i][:300])

        logger.info(
            "Evaluation: acc=%.4f f1=%.4f p=%.4f r=%.4f auc=%s n=%d",
            acc, f1, p, r, f"{auc:.4f}" if auc else "N/A", len(y_test_arr),
        )

        return EvaluationReport(
            accuracy=round(acc, 6),
            f1=round(f1, 6),
            precision=round(p, 6),
            recall=round(r, 6),
            auc_roc=round(auc, 6) if auc is not None else None,
            confusion_matrix=cm,
            classification_report=report,
            n_samples=len(y_test_arr),
            false_positive_examples=fp_examples,
            false_negative_examples=fn_examples,
            confidence_stats=conf_stats,
        )