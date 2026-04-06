"""Streamlit dashboard for Fake News Detection.

Pages:
    1. Predict — classify a news article
    2. Batch — upload CSV for batch classification
    3. History — view recent prediction history

Run: streamlit run streamlit_app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.service import InferenceService, Prediction

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
)


@st.cache_resource
def load_model() -> InferenceService:
    return InferenceService()


def init_history() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def add_to_history(text: str, result: Prediction) -> None:
    st.session_state.history.insert(0, {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "label": result.label,
        "confidence": f"{result.confidence:.1%}",
        "latency": f"{result.latency_ms:.1f}ms",
        "model": result.model_version,
    })
    # Keep last 50
    st.session_state.history = st.session_state.history[:50]


# ── Sidebar ──────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.header("🔍 Fake News Detector")
        page = st.radio("Navigate", ["Predict", "Batch", "History"], label_visibility="collapsed")

        st.divider()

        st.markdown(
            "**Pipeline:** Text Cleaning → TF-IDF → Classification\n\n"
            "**Models:** Logistic Regression, Linear SVM, Multinomial NB\n\n"
            "**Tracking:** MLflow experiment logging"
        )

        st.divider()

        predictor = load_model()
        health = predictor.health_check()
        status_icon = "🟢" if health["status"] == "healthy" else "🔴"
        st.caption(f"{status_icon} Model: {predictor.model_version}")
        st.caption(f"Latency: {health.get('test_latency_ms', '?')}ms")

    return page


# ── Predict page ─────────────────────────────────────────────────────

def render_predict_page() -> None:
    st.title("Classify Article")
    st.markdown("Paste a news article to classify it as **REAL** or **FAKE**.")

    predictor = load_model()

    user_input = st.text_area(
        "News article text",
        height=200,
        placeholder="Paste the full article text here...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        btn = st.button("Classify", type="primary", use_container_width=True)

    if btn:
        if not user_input or len(user_input.strip()) < 10:
            st.warning("Please enter at least 10 characters.")
            return

        with st.spinner("Analyzing..."):
            result = predictor.predict(user_input)

        add_to_history(user_input, result)

        st.divider()

        # Result display
        col_label, col_conf, col_lat = st.columns(3)

        with col_label:
            if result.label == "FAKE":
                st.error(f"⚠️ **{result.label}**")
            elif result.label == "REAL":
                st.success(f"✅ **{result.label}**")
            else:
                st.info(f"❓ **{result.label}**")

        with col_conf:
            st.metric("Confidence", f"{result.confidence:.1%}")

        with col_lat:
            st.metric("Latency", f"{result.latency_ms:.1f}ms")

        st.progress(result.confidence, text=f"Model confidence: {result.confidence:.1%}")


# ── Batch page ───────────────────────────────────────────────────────

def render_batch_page() -> None:
    st.title("Batch Classification")
    st.markdown("Upload a CSV file with a `text` column to classify multiple articles.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "text" not in df.columns:
            st.error("CSV must have a `text` column.")
            return

        st.info(f"Found {len(df)} articles. Processing...")

        predictor = load_model()
        results = []

        progress = st.progress(0)
        for i, text in enumerate(df["text"]):
            result = predictor.predict(str(text))
            results.append({
                "text": str(text)[:100],
                "label": result.label,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
            })
            progress.progress((i + 1) / len(df))

        results_df = pd.DataFrame(results)

        st.divider()

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(results_df))
        with col2:
            fake_pct = (results_df["label"] == "FAKE").mean()
            st.metric("Fake %", f"{fake_pct:.1%}")
        with col3:
            avg_conf = results_df["confidence"].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")

        st.dataframe(results_df, use_container_width=True)

        csv = results_df.to_csv(index=False)
        st.download_button("Download Results", csv, "predictions.csv", "text/csv")


# ── History page ─────────────────────────────────────────────────────

def render_history_page() -> None:
    st.title("Prediction History")

    if not st.session_state.get("history"):
        st.info("No predictions yet. Go to the Predict page to classify articles.")
        return

    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    init_history()
    page = render_sidebar()

    if page == "Predict":
        render_predict_page()
    elif page == "Batch":
        render_batch_page()
    elif page == "History":
        render_history_page()


if __name__ == "__main__":
    main()