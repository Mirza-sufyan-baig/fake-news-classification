<p align="center">
  <img src="https://img.shields.io/badge/🛡️-FakeGuard_AI-00f0ff?style=for-the-badge&labelColor=0a0a0f" alt="FakeGuard AI"/>
</p>

<h1 align="center">FakeGuard AI</h1>

<p align="center">
  <strong>Production-Grade Fake News Detection System</strong><br>
  <em>Multi-model classification • Linguistic explainability • Real-time monitoring</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Streamlit-1.36-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/BERT-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="BERT"/>
  <img src="https://img.shields.io/badge/MLflow-2.14-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Tests-60+-4caf50?style=flat-square" alt="Tests"/>
</p>

---

## Overview

FakeGuard AI is an end-to-end fake news detection platform that classifies news articles as **REAL** or **FAKE** using a multi-model approach. It combines classical ML pipelines (TF-IDF + Logistic Regression / SVM / Naive Bayes) with BERT transformer fine-tuning, augmented by a **linguistic analysis engine** that scores articles on sensationalism, clickbait patterns, emotional manipulation, and credibility signals.

The system follows production engineering standards: Pydantic-validated configuration, structured JSON logging with request tracing, a 13-class custom exception hierarchy, thread-safe monitoring with drift alerts, model versioning with metadata sidecars, and a full test suite.

### Key Features

- **Multi-Model Engine** — Switch between Classical ML, BERT Transformer, or Ensemble (weighted vote) at prediction time
- **Linguistic Explainability** — Every prediction shows sensationalism, clickbait, emotional manipulation, and credibility scores with human-readable risk factors
- **Real-Time Monitoring** — Sliding window tracker with automated alerts for confidence drift, label skew, latency spikes, and error rates
- **Production API** — FastAPI with health checks, batch prediction, CORS, request ID tracing, and domain exception handling
- **Experiment Tracking** — MLflow logs parameters, metrics, and artifacts for every training run
- **Cyberpunk Dashboard** — Neon-themed Streamlit UI with animated verdicts, model selector, batch CSV analysis, and prediction history

---

## Architecture

```
fakenewsdetection/
│
├── configs/
│   └── settings.py              # Pydantic Settings — 50+ validated env vars
│
├── src/
│   ├── core/
│   │   └── exceptions.py        # 13-class domain exception hierarchy
│   │
│   ├── api/
│   │   └── main.py              # FastAPI — health, predict, batch, monitoring
│   │
│   ├── data_pipeline/
│   │   └── loader.py            # Load → validate → quality check → clean → encode
│   │
│   ├── features/
│   │   ├── cleaner.py           # Deterministic text cleaning (single source of truth)
│   │   └── linguistic_analyzer.py  # Sensationalism, clickbait, emotion scoring
│   │
│   ├── inference/
│   │   └── service.py           # Multi-model service (Classical / BERT / Ensemble)
│   │
│   ├── models/
│   │   └── bert_classifier.py   # BERT fine-tuning with mixed precision + early stopping
│   │
│   ├── monitoring/
│   │   └── tracker.py           # Thread-safe sliding window with drift alerts
│   │
│   ├── training/
│   │   ├── train_classical.py   # Cross-validated grid search + MLflow logging
│   │   └── evaluate.py          # AUC-ROC, confusion matrix, error analysis
│   │
│   └── utils/
│       ├── logger.py            # JSON + colored logging with request ID context
│       └── versioning.py        # Model registry with metadata JSON sidecars
│
├── streamlit_app/
│   └── dashboard.py             # Neon cyberpunk UI — 3 pages + model selector
│
├── tests/                       # 60+ pytest cases across 6 test modules
│   ├── conftest.py
│   ├── test_cleaner.py
│   ├── test_versioning.py
│   ├── test_api.py
│   ├── test_inference.py
│   ├── test_monitoring.py
│   └── test_exceptions.py
│
├── scripts/
│   └── healthcheck.py           # Docker health check script
│
├── Dockerfile                   # Multi-stage, non-root user, health check
├── docker-compose.yml           # API + Dashboard services
├── Makefile                     # 15 commands
├── pyproject.toml               # pytest, ruff, coverage config
├── requirements.txt             # Direct dependencies only
└── requirements.lock            # Pinned versions for reproducibility
```

---

## Quick Start

### Prerequisites

- Python 3.11+ ([python.org](https://www.python.org/downloads/))
- Dataset at `data/raw/fake_news_dataset.csv`

### Windows

```
1. Double-click  setup.bat        → Creates venv + installs everything
2. Double-click  run_train.bat    → Trains the model
3. Double-click  run_dashboard.bat → Opens dashboard at localhost:8501
```

### Manual Setup

```bash
# Clone
git clone https://github.com/yourusername/fakenewsdetection.git
cd fakenewsdetection

# Environment
python -m venv fnenv
source fnenv/bin/activate        # Linux/Mac
fnenv\Scripts\activate           # Windows

# Install
pip install -r requirements.txt
cp .env.example .env

# Train
python -m src.training.train_classical

# Launch
streamlit run streamlit_app/dashboard.py                           # Dashboard → localhost:8501
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload     # API → localhost:8000/docs
```

### Docker

```bash
docker compose up --build -d
# API  → localhost:8000
# UI   → localhost:8501
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Deep health check — validates model can predict |
| `POST` | `/predict` | Single article classification |
| `POST` | `/predict/batch` | Batch classification (up to 100 articles) |
| `GET` | `/monitoring` | Live prediction stats + drift alerts |

**Example:**

```bash
curl -s http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists discover new method for carbon capture using algae reactors."}'
```

**Response:**

```json
{
    "label": "REAL",
    "confidence": 0.8732,
    "raw_label": 0,
    "latency_ms": 2.41,
    "model_version": "baseline_v3"
}
```

Every response includes `X-Request-ID` and `X-Process-Time` headers for distributed tracing.

---

## Models

### Classical ML Pipeline

Cross-validated evaluation: **3 models × 2 ngram ranges × 2 class weights = 12 configurations**, stratified 5-fold CV. All runs logged to MLflow.

| Model | Best F1 | Notes |
|-------|---------|-------|
| Multinomial Naive Bayes | **0.575** | Best performer on this dataset |
| Linear SVM | 0.512 | With (1,2) ngram range |
| Logistic Regression | 0.510 | With balanced class weights |

TF-IDF config: `sublinear_tf=True`, `min_df=2`, `max_df=0.95`, 10,000 features.

### BERT Transformer

Optional fine-tuning of `bert-base-uncased`:

- Mixed precision training (fp16 on GPU)
- Learning rate warmup (10% of steps)
- Weight decay regularization (0.01)
- Early stopping (patience=3)
- Checkpoint resumption

```bash
python -m src.models.bert_classifier
```

### Ensemble Mode

Weighted voting: Classical (40%) + BERT (60%). Falls back to Classical-only if BERT is not trained. Dashboard shows individual model votes and combined probability.

---

## Linguistic Analysis

Inspired by [Potthast et al. (2018)](https://aclanthology.org/N18-2013/) and [Rashkin et al. (2017)](https://aclanthology.org/D17-1317/), every prediction includes a model-independent linguistic profile:

| Dimension | What It Measures | Signals |
|-----------|-----------------|---------|
| **Sensationalism** | Hype and exaggeration | ALL CAPS ratio, exclamation abuse, superlative claims |
| **Clickbait** | Manipulative engagement | 40+ phrase patterns ("you won't believe", "share before they delete") |
| **Emotional Manipulation** | Fear / anger / urgency triggers | Lexicon-based detection across 3 emotion categories |
| **Credibility** | Professional writing quality | Attribution phrases, hedging language, sentence structure |

Each dimension scores 0–100% and combines into an overall **Linguistic Risk Score** with HIGH / MEDIUM / LOW classification.

### Risk Factors (Explainability)

Every prediction surfaces human-readable explanations:

```
⚠ Clickbait language detected: shocking, exposed, share before
⚠ Emotional manipulation (fear, urgency)
⚠ Excessive capitalization (34% of text)
✓ No attribution phrases found
```

---

## Monitoring

The `/monitoring` endpoint returns real-time stats from a sliding window:

| Metric | Alert Threshold |
|--------|----------------|
| Label distribution | Fake ratio > 80% or < 20% |
| Low confidence ratio | > 30% of predictions below 0.6 |
| Latency p95 | > 100ms |
| Error rate | > 5% |

---

## Configuration

All settings are driven by `FND_` prefixed environment variables. See [`.env.example`](.env.example).

```bash
FND_TFIDF_MAX_FEATURES=20000 FND_CV_FOLDS=10 python -m src.training.train_classical
```

| Variable | Default | Description |
|----------|---------|-------------|
| `FND_ENV` | `dev` | Environment (dev / staging / production) |
| `FND_TFIDF_MAX_FEATURES` | `10000` | TF-IDF vocabulary size |
| `FND_CV_FOLDS` | `5` | Cross-validation folds |
| `FND_BERT_EPOCHS` | `2` | BERT fine-tuning epochs |
| `FND_API_BATCH_MAX` | `100` | Max articles per batch request |
| `FND_LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Testing

```bash
pytest tests/ -v                    # 60+ tests
pytest tests/ -v --cov=src          # With coverage
```

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_cleaner.py` | 25 | Text transforms, edge cases, determinism, config immutability |
| `test_versioning.py` | 10 | Numeric sort, gaps, collision detection, missing vectorizer |
| `test_api.py` | 10 | Health, predict, batch, CORS, request ID propagation |
| `test_inference.py` | 8 | Mock models, confidence calibration, empty text handling |
| `test_monitoring.py` | 11 | Sliding window, alerts, thread safety, report persistence |
| `test_exceptions.py` | 7 | Inheritance chains, context propagation, catch-by-category |

---

## Engineering Highlights

| Aspect | Implementation |
|--------|---------------|
| Type Safety | Full annotations on every function, method, and return value |
| Error Handling | 13-class exception hierarchy with context dicts |
| Single Source of Truth | `TextCleaner` used across training, inference, and evaluation |
| Logging | JSON in production (ELK/Datadog), colored in dev, request ID tracing |
| Configuration | Pydantic Settings with field + model validators |
| Model Registry | Regex-based versioning with JSON metadata sidecars |
| Immutability | Frozen dataclasses for configs, predictions, and profiles |
| Thread Safety | `deque`-backed monitoring with `threading.Lock` |
| Confidence Calibration | Sigmoid scaling for models without `predict_proba` |
| Overfitting Detection | Train vs test F1 gap warning during tuning |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, BERT (HuggingFace Transformers), TF-IDF |
| NLP | Custom linguistic analyzer (sensationalism, clickbait, emotion lexicons) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit (custom neon cyberpunk theme) |
| Tracking | MLflow |
| Testing | pytest, pytest-cov |
| Linting | Ruff |
| Containers | Docker (multi-stage, non-root) |
| Config | Pydantic Settings + dotenv |

---

## Troubleshooting

**Windows users:** Run `doctor.bat` — it checks Python, venv health, packages, model files, and imports.

| Issue | Fix |
|-------|-----|
| `python` not recognized | Install from [python.org](https://www.python.org/downloads/), check "Add to PATH" |
| `ModuleNotFoundError` | Activate venv first, then `pip install -r requirements.txt` |
| `No models found` | Run `python -m src.training.train_classical` |
| Venv linked to Anaconda | Delete `fnenv/`, run `setup.bat` |
| BERT warnings in VS Code | `pip install torch transformers` or ignore (BERT is optional) |

---

## License

MIT

---

<p align="center">
  <sub>Built with ⚡ by <strong>Shbz</strong> — 2026</sub>
</p>
