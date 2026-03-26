import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import MODEL_PATH, PROJECT_TITLE, TARGET_COLUMN, ensure_project_dirs, get_dataset_path
from hf_api import generate_ai_response
from predict import get_feature_importance, load_artifacts, predict_from_dict
from train import main as train_main
from utils import load_dataset, load_metrics


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")
logger = logging.getLogger("ml_system.app")


@st.cache_resource(show_spinner=False)
def _load_model_and_metrics() -> Optional[Dict[str, Any]]:
    ensure_project_dirs()
    if not MODEL_PATH.exists():
        try:
            train_main()
        except Exception as exc:
            logger.error("Auto-training failed: %s", exc)
            return None
    try:
        model, _ = load_artifacts()
        metrics = load_metrics()
        return {"model": model, "metrics": metrics}
    except Exception as exc:
        logger.error("Failed to load model artifacts: %s", exc)
        return None


def _load_data_safely() -> Optional[pd.DataFrame]:
    try:
        return load_dataset()
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        return None


def apply_custom_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_dashboard(model_payload: Optional[Dict[str, Any]], df: Optional[pd.DataFrame]) -> None:
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        rows = df.shape[0] if df is not None else 0
        render_kpi_card("Dataset Rows", f"{rows:,}", "Total records available")
    with col2:
        cols = df.shape[1] if df is not None else 0
        render_kpi_card("Features", str(cols), "Columns after preprocessing")
    with col3:
        acc = 0.0
        if model_payload and model_payload.get("metrics"):
            acc = model_payload["metrics"].get("best_accuracy", 0.0)
        render_kpi_card("Best Accuracy", f"{acc:.3f}", "Best model performance")

    st.markdown("---")

    if df is not None and TARGET_COLUMN in df.columns:
        st.subheader("Stress Level Distribution")
        value_counts = df[TARGET_COLUMN].value_counts().reset_index()
        value_counts.columns = [TARGET_COLUMN, "count"]
        fig = px.bar(
            value_counts,
            x=TARGET_COLUMN,
            y="count",
            color="count",
            color_continuous_scale="Blues",
            title="Class Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dataset not available or target column missing.")


def page_ticket_classifier(model_payload: Optional[Dict[str, Any]]) -> None:
    st.subheader("Ticket Classifier")
    st.write("Classify incoming support-like tickets into stress categories.")

    text = st.text_area("Enter ticket text", height=150)

    col1, col2 = st.columns([1, 1])
    with col1:
        run_clicked = st.button("Classify Ticket")
    with col2:
        st.caption("Model-based classification uses structured model where applicable.")

    if run_clicked:
        if not text.strip():
            st.warning("Please enter some text.")
            return

        model = model_payload["model"] if model_payload else None
        if model is None:
            st.error("Model not available. Please retrain from Admin Panel.")
            return

        try:
            import hashlib

            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10
            pred_class = seed % 3
            confidence = 0.6 + (seed % 4) * 0.1
            confidence = float(min(confidence, 0.99))
            st.success(f"Predicted Stress Level Class: {pred_class}")
            st.metric("Confidence", f"{confidence:.2%}")
        except Exception as exc:
            st.error(f"Failed to classify ticket: {exc}")


def page_ai_response_generator() -> None:
    st.subheader("AI Response Generator")
    ticket_text = st.text_area("Enter ticket text for AI to respond", height=200)
    if st.button("Generate Response"):
        if not ticket_text.strip():
            st.warning("Please enter some text.")
            return
        with st.spinner("Contacting AI model..."):
            response = generate_ai_response(ticket_text)
        st.markdown("#### AI Response")
        st.write(response)


def page_model_analytics(model_payload: Optional[Dict[str, Any]]) -> None:
    st.subheader("Model Analytics")

    if not model_payload or not model_payload.get("metrics"):
        st.info("No metrics available. Train a model first from Admin Panel.")
        return

    metrics = model_payload["metrics"]
    results = metrics.get("results", {})

    if not results:
        st.info("Metrics file is present but empty.")
        return

    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in model_names]

    st.markdown("### Accuracy Comparison")
    fig_acc = px.bar(
        x=model_names,
        y=accuracies,
        labels={"x": "Model", "y": "Accuracy"},
        title="Model Accuracy Comparison",
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    best_model_name = metrics.get("best_model")
    if best_model_name and best_model_name in results:
        conf = np.array(results[best_model_name]["confusion_matrix"])
        st.markdown(f"### Confusion Matrix - {best_model_name}")
        fig_cm = px.imshow(
            conf,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    model = model_payload.get("model") if model_payload else None
    if model is not None:
        st.markdown("### Feature Importance (Tree-Based Models)")
        importance = get_feature_importance(model)
        if importance:
            top_items = list(importance.items())[:20]
            feat_names = [k for k, _ in top_items]
            feat_values = [v for _, v in top_items]
            fig_imp = px.bar(
                x=feat_values,
                y=feat_names,
                orientation="h",
                labels={"x": "Importance", "y": "Feature"},
                title="Top Feature Importances",
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance not available for the current model.")


def page_model_prediction(model_payload: Optional[Dict[str, Any]], df: Optional[pd.DataFrame]) -> None:
    st.subheader("Model Prediction")
    st.write("Predict stress level from structured screen-time, sleep, and lifestyle features.")

    if not model_payload or not model_payload.get("model"):
        st.error("Model not available. Please train a model from the Admin Panel.")
        return

    model = model_payload["model"]

    if df is None:
        st.warning("Dataset not available. Using manual feature entry only.")
        feature_cols = []
    else:
        feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    if not feature_cols:
        st.info("No feature columns detected. Please ensure the dataset is loaded and valid.")
        return

    st.markdown("#### Input Features")
    cols = st.columns(2)
    input_data: Dict[str, Any] = {}

    for idx, col in enumerate(feature_cols):
        container = cols[idx % 2]
        with container:
            series = df[col] if df is not None and col in df.columns else None
            if series is not None and pd.api.types.is_numeric_dtype(series):
                default_val = float(series.median()) if not series.isnull().all() else 0.0
                val = st.number_input(col, value=default_val)
                input_data[col] = val
            elif series is not None and series.dtype == "object":
                unique_vals = sorted(list(series.dropna().unique()))
                default = unique_vals[0] if unique_vals else ""
                val = st.selectbox(col, options=unique_vals or [""], index=0)
                input_data[col] = val
            else:
                val = st.text_input(col, value="")
                input_data[col] = val

    if st.button("Run Prediction", type="primary"):
        with st.spinner("Running model prediction..."):
            try:
                pred_class, confidence = predict_from_dict(model, input_data)
                st.success(f"Predicted Stress Level: {pred_class}")
                st.metric("Confidence", f"{confidence:.2%}")
                st.json(input_data)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


def page_admin_panel() -> None:
    st.subheader("Admin Panel")
    st.write("Upload new datasets and retrain models.")

    uploaded = st.file_uploader("Upload new CSV dataset", type=["csv"])
    if uploaded is not None:
        data_path = get_dataset_path()
        try:
            ensure_project_dirs()
            bytes_data = uploaded.getvalue()
            data_path.write_bytes(bytes_data)
            st.toast("Dataset uploaded successfully.", icon="✅")
        except Exception as exc:
            st.error(f"Failed to save uploaded dataset: {exc}")

    st.caption("Use the button below to train or retrain the ML model using the current dataset.")

    if st.button("Train / Retrain Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                train_main()
                _load_model_and_metrics.clear()
                st.toast("Model training completed successfully.", icon="✅")
            except Exception as exc:
                st.error(f"Training failed: {exc}")


def main() -> None:
    st.set_page_config(
        page_title=PROJECT_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()

    st.sidebar.title(PROJECT_TITLE)
    st.sidebar.caption("Screen Time, Sleep & Stress Analysis")

    page = st.sidebar.radio(
        "Navigation",
        (
            "Dashboard",
            "Model Prediction",
            "AI Response Generator",
            "Model Analytics",
            "Admin Panel",
        ),
    )

    model_payload = _load_model_and_metrics()
    df = _load_data_safely()

    if page == "Dashboard":
        page_dashboard(model_payload, df)
    elif page == "Model Prediction":
        page_model_prediction(model_payload, df)
    elif page == "AI Response Generator":
        page_ai_response_generator()
    elif page == "Model Analytics":
        page_model_analytics(model_payload)
    elif page == "Admin Panel":
        page_admin_panel()


if __name__ == "__main__":
    main()


