"""Model artifact versioning and registry.

Naming convention:
    models/baseline_v1.pkl           — serialized model
    models/baseline_v1_vectorizer.pkl — paired TF-IDF vectorizer
    models/baseline_v1_meta.json     — training metadata (config, metrics, timestamp)

Version detection uses regex to avoid the string-sort bug
where 'v10' < 'v2' lexicographically.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from configs.settings import get_settings
from src.core.exceptions import ModelNotFoundError, ModelVersionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VERSION_PATTERN = re.compile(r"^{prefix}_v(\d+)\.pkl$")


def _compile_pattern(prefix: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(prefix)}_v(\d+)\.pkl$")


def _scan_versions(prefix: str, model_dir: Path) -> list[tuple[int, str]]:
    """Scan model directory and return sorted (version_int, filename) pairs."""
    pattern = _compile_pattern(prefix)
    candidates: list[tuple[int, str]] = []

    if not model_dir.exists():
        return candidates

    for f in os.listdir(model_dir):
        m = pattern.match(f)
        if m:
            candidates.append((int(m.group(1)), f))

    candidates.sort(key=lambda x: x[0])
    return candidates


def get_next_version(prefix: str = "baseline", model_dir: Path | None = None) -> str:
    """Return the next sequential version string, e.g. 'baseline_v12'.

    Raises:
        ModelVersionError: If version conflicts are detected.
    """
    model_dir = model_dir or get_settings().model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    candidates = _scan_versions(prefix, model_dir)
    next_v = max((v for v, _ in candidates), default=0) + 1
    tag = f"{prefix}_v{next_v}"

    # Collision check
    if (model_dir / f"{tag}.pkl").exists():
        raise ModelVersionError(
            f"Version collision: {tag}.pkl already exists",
            context={"model_dir": str(model_dir), "tag": tag},
        )

    logger.info("Next model version: %s", tag)
    return tag


def get_latest_model(
    prefix: str = "baseline",
    model_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Return (model_path, vectorizer_path) for the latest version.

    Raises:
        ModelNotFoundError: If no matching model artifacts exist.
    """
    model_dir = model_dir or get_settings().model_dir
    candidates = _scan_versions(prefix, model_dir)

    if not candidates:
        raise ModelNotFoundError(
            f"No models matching '{prefix}_v*.pkl' in {model_dir}",
            context={"model_dir": str(model_dir), "prefix": prefix},
        )

    _, latest_file = candidates[-1]
    model_path = model_dir / latest_file
    vec_path = model_dir / latest_file.replace(".pkl", "_vectorizer.pkl")

    if not vec_path.exists():
        raise ModelNotFoundError(
            f"Vectorizer not found for {latest_file}",
            context={"model_path": str(model_path), "expected_vectorizer": str(vec_path)},
        )

    logger.info("Latest model: %s", model_path.name)
    return model_path, vec_path


def list_versions(prefix: str = "baseline", model_dir: Path | None = None) -> list[str]:
    """List all available model versions."""
    model_dir = model_dir or get_settings().model_dir
    return [f for _, f in _scan_versions(prefix, model_dir)]


def save_model_metadata(
    version: str,
    metrics: dict[str, float],
    config: dict[str, Any],
    model_dir: Path | None = None,
) -> Path:
    """Save training metadata alongside model artifacts.

    Creates a JSON sidecar file: baseline_v3_meta.json
    """
    model_dir = model_dir or get_settings().model_dir
    meta_path = model_dir / f"{version}_meta.json"

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "config": config,
    }

    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Metadata saved: %s", meta_path.name)
    return meta_path