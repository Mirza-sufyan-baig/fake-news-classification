"""Structured logging with JSON formatting, rotation, and context injection.

In production, logs are JSON-formatted for ingestion by ELK/Datadog/etc.
In dev, logs are human-readable.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started", extra={"model": "LR", "fold": 3})
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from configs.settings import Environment, get_settings

# ── Request ID context var (set by API middleware) ───────────────────
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Inject request ID if present
        req_id = request_id_var.get("")
        if req_id:
            log_entry["request_id"] = req_id

        # Inject any extra fields
        for key in ("model", "fold", "metric", "version", "duration_ms", "error_type"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        # Exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        req_id = request_id_var.get("")
        req_part = f" [{req_id[:8]}]" if req_id else ""

        return (
            f"{color}{record.levelname:<8}{self.RESET} "
            f"{record.asctime} | {record.name}:{record.funcName}:{record.lineno}"
            f"{req_part} | {record.getMessage()}"
        )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with console + file handlers.

    - Dev: colored console output
    - Production: JSON to file + console
    - Always: rotating file handler
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(settings.log_level)

    # ── Console handler ──────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    if settings.is_production:
        ch.setFormatter(JSONFormatter())
    else:
        ch.setFormatter(ConsoleFormatter())
        ch.formatter.datefmt = "%H:%M:%S"
    logger.addHandler(ch)

    # ── File handler (always JSON for machine parsing) ───────────
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
    )
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return uuid.uuid4().hex[:16]