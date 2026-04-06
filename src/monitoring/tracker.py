"""Production monitoring: prediction logging, drift detection, and metrics.

Tracks:
    - Prediction distribution (fake vs real ratio over time)
    - Confidence distribution (low-confidence alert threshold)
    - Text length distribution (input drift detection)
    - Latency percentiles
    - Error rates

Usage:
    from src.monitoring.tracker import PredictionTracker
    tracker = PredictionTracker()
    tracker.log(prediction)
    report = tracker.get_report()
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from configs.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionRecord:
    """Lightweight record for monitoring (no raw text stored)."""
    label: str
    confidence: float
    latency_ms: float
    text_length: int
    timestamp: float = field(default_factory=time.time)


class PredictionTracker:
    """Thread-safe sliding window prediction monitor.

    Maintains a fixed-size deque of recent predictions for
    real-time monitoring without unbounded memory growth.
    """

    def __init__(self, window_size: int = 10_000) -> None:
        self._window: deque[PredictionRecord] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._total_predictions: int = 0
        self._total_errors: int = 0
        self._settings = get_settings()

    def log(self, record: PredictionRecord) -> None:
        """Thread-safe prediction logging."""
        with self._lock:
            self._window.append(record)
            self._total_predictions += 1

    def log_error(self) -> None:
        """Track prediction errors."""
        with self._lock:
            self._total_errors += 1
            self._total_predictions += 1

    def get_report(self) -> dict[str, Any]:
        """Generate monitoring report from sliding window."""
        with self._lock:
            records = list(self._window)

        if not records:
            return {
                "status": "no_data",
                "total_predictions": self._total_predictions,
                "total_errors": self._total_errors,
            }

        labels = [r.label for r in records]
        confidences = [r.confidence for r in records]
        latencies = [r.latency_ms for r in records]
        text_lengths = [r.text_length for r in records]

        n = len(records)
        fake_count = labels.count("FAKE")
        real_count = labels.count("REAL")
        unknown_count = labels.count("UNKNOWN")

        conf_arr = np.array(confidences)
        lat_arr = np.array(latencies)
        tl_arr = np.array(text_lengths)

        report: dict[str, Any] = {
            "window_size": n,
            "total_predictions": self._total_predictions,
            "total_errors": self._total_errors,
            "error_rate": round(self._total_errors / max(self._total_predictions, 1), 4),

            # Label distribution
            "label_distribution": {
                "FAKE": fake_count,
                "REAL": real_count,
                "UNKNOWN": unknown_count,
                "fake_ratio": round(fake_count / n, 4),
            },

            # Confidence stats
            "confidence": {
                "mean": round(float(conf_arr.mean()), 4),
                "std": round(float(conf_arr.std()), 4),
                "min": round(float(conf_arr.min()), 4),
                "p50": round(float(np.percentile(conf_arr, 50)), 4),
                "p95": round(float(np.percentile(conf_arr, 95)), 4),
                "low_confidence_count": int((conf_arr < 0.6).sum()),
                "low_confidence_ratio": round(float((conf_arr < 0.6).mean()), 4),
            },

            # Latency stats
            "latency_ms": {
                "mean": round(float(lat_arr.mean()), 2),
                "p50": round(float(np.percentile(lat_arr, 50)), 2),
                "p95": round(float(np.percentile(lat_arr, 95)), 2),
                "p99": round(float(np.percentile(lat_arr, 99)), 2),
                "max": round(float(lat_arr.max()), 2),
            },

            # Input distribution (drift detection)
            "text_length": {
                "mean": round(float(tl_arr.mean()), 1),
                "std": round(float(tl_arr.std()), 1),
                "min": int(tl_arr.min()),
                "max": int(tl_arr.max()),
            },

            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Alerts
        alerts = []
        if report["confidence"]["low_confidence_ratio"] > 0.3:
            alerts.append("HIGH_LOW_CONFIDENCE: >30% predictions below 0.6 confidence")
        if report["label_distribution"]["fake_ratio"] > 0.8:
            alerts.append("LABEL_SKEW: >80% predictions are FAKE")
        if report["label_distribution"]["fake_ratio"] < 0.2:
            alerts.append("LABEL_SKEW: <20% predictions are FAKE")
        if report["latency_ms"]["p95"] > 100:
            alerts.append(f"HIGH_LATENCY: p95={report['latency_ms']['p95']}ms")
        if report["error_rate"] > 0.05:
            alerts.append(f"HIGH_ERROR_RATE: {report['error_rate']:.1%}")

        report["alerts"] = alerts

        if alerts:
            logger.warning("Monitoring alerts: %s", alerts)

        return report

    def save_report(self, path: Path | None = None) -> Path:
        """Persist monitoring report to JSON file."""
        settings = get_settings()
        path = path or settings.log_dir / "monitoring_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        report = self.get_report()
        path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Monitoring report saved to %s", path)
        return path