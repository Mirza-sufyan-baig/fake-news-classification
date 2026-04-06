"""FastAPI application for fake news detection.

Production features:
    - Lifespan-managed model loading (no module-level side effects)
    - Request ID injection via middleware (X-Request-ID header)
    - Request timing middleware (X-Process-Time header)
    - CORS with configurable origins
    - Structured error handling (domain exceptions → HTTP responses)
    - Prediction monitoring integration
    - Health check with deep model validation
    - Batch prediction endpoint with configurable limits
    - Input validation via Pydantic with clear error messages

Endpoints:
    GET  /health          → Health check + model status
    POST /predict         → Single article classification
    POST /predict/batch   → Batch classification (up to N articles)
    GET  /monitoring      → Live monitoring report
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from configs.settings import get_settings
from src.core.exceptions import FNDBaseError, InferenceError, ModelError
from src.inference.service import InferenceService
from src.monitoring.tracker import PredictionRecord, PredictionTracker
from src.utils.logger import generate_request_id, get_logger, request_id_var

logger = get_logger(__name__)

# ── Global state (populated during lifespan) ─────────────────────────
_predictor: InferenceService | None = None
_tracker: PredictionTracker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _predictor, _tracker

    logger.info("Starting application...")
    try:
        _predictor = InferenceService()
        _tracker = PredictionTracker()

        health = _predictor.health_check()
        logger.info("Model health check: %s", health)

        if health["status"] != "healthy":
            raise RuntimeError(f"Model health check failed: {health}")

    except Exception:
        logger.exception("Failed to initialize inference service")
        raise

    logger.info("Application started successfully.")
    yield

    # Shutdown
    if _tracker:
        _tracker.save_report()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="Fake News Detection API",
    version="2.0.0",
    description=(
        "Production-grade fake news classification using TF-IDF + classical ML. "
        "Includes monitoring, batch prediction, and structured error handling."
    ),
    lifespan=lifespan,
)

# ── Middleware stack ──────────────────────────────────────────────────
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Inject request ID and measure request duration."""
    req_id = request.headers.get("X-Request-ID", generate_request_id())
    request_id_var.set(req_id)

    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    response.headers["X-Request-ID"] = req_id
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"

    logger.info(
        "%s %s → %d (%.1fms)",
        request.method, request.url.path, response.status_code, elapsed * 1000,
    )
    return response


# ── Exception handlers ───────────────────────────────────────────────

@app.exception_handler(FNDBaseError)
async def fnd_error_handler(request: Request, exc: FNDBaseError):
    """Translate domain exceptions to structured HTTP responses."""
    if isinstance(exc, ModelError):
        status = 503
    elif isinstance(exc, InferenceError):
        status = 422
    else:
        status = 500

    logger.error("Domain error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc),
            "context": exc.context,
        },
    )


# ── Request / Response schemas ───────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=50_000,
        description="News article text to classify (10-50,000 characters).",
        examples=["Scientists discover new method for carbon capture using algae-based reactors."],
    )


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=settings.api_batch_max,
        description=f"List of articles (1-{settings.api_batch_max}).",
    )


class PredictResponse(BaseModel):
    label: str
    confidence: float
    raw_label: int
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    model_health: dict[str, Any] | None
    version: str


class MonitoringResponse(BaseModel):
    report: dict[str, Any]


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health():
    """Deep health check — validates model can produce predictions."""
    model_health = _predictor.health_check() if _predictor else None
    return HealthResponse(
        status="healthy" if _predictor and model_health and model_health["status"] == "healthy" else "degraded",
        model_loaded=_predictor is not None,
        model_version=_predictor.model_version if _predictor else None,
        model_health=model_health,
        version=app.version,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """Classify a single news article as FAKE or REAL."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    result = _predictor.predict(request.text)

    # Log to monitoring
    if _tracker:
        _tracker.log(PredictionRecord(
            label=result.label,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            text_length=len(request.text),
        ))

    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        raw_label=result.raw_label,
        latency_ms=result.latency_ms,
        model_version=result.model_version,
    )


@app.post("/predict/batch", response_model=list[PredictResponse], tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """Classify multiple articles. Returns results in input order."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for text in request.texts:
        try:
            r = _predictor.predict(text)
            results.append(PredictResponse(
                label=r.label,
                confidence=r.confidence,
                raw_label=r.raw_label,
                latency_ms=r.latency_ms,
                model_version=r.model_version,
            ))

            if _tracker:
                _tracker.log(PredictionRecord(
                    label=r.label,
                    confidence=r.confidence,
                    latency_ms=r.latency_ms,
                    text_length=len(text),
                ))

        except Exception as e:
            if _tracker:
                _tracker.log_error()
            logger.error("Batch item failed: %s", e)
            results.append(PredictResponse(
                label="ERROR",
                confidence=0.0,
                raw_label=-1,
                latency_ms=0.0,
                model_version=_predictor.model_version if _predictor else "unknown",
            ))

    return results


@app.get("/monitoring", response_model=MonitoringResponse, tags=["Operations"])
def monitoring():
    """Live monitoring report from prediction sliding window."""
    if _tracker is None:
        raise HTTPException(status_code=503, detail="Monitoring not initialized.")
    return MonitoringResponse(report=_tracker.get_report())