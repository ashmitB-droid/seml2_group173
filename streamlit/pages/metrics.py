import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import ConfusionMatrixDisplay

import streamlit as st
from config import METRICS_URL

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
st.set_page_config(page_title="Model Metrics", page_icon="📊", layout="wide")
st.title("📊 Model Performance Dashboard")
st.markdown("---")

# -------------------------------------------------------
# Get Metrics
# -------------------------------------------------------
try:
    response = requests.get(METRICS_URL)
    if response.status_code != 200:
        st.error("Unable to fetch metrics from FastAPI.")
        st.stop()
    metrics = response.json()
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# -------------------------------------------------------
# Model Information
# -------------------------------------------------------
st.subheader("📌 Model Information")
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Model:** {metrics['model']}")
    st.info(f"**Dataset:** {metrics['dataset']}")
with col2:
    st.info(f"**Training Samples:** {metrics['train_samples']:,}")
    st.info(f"**Testing Samples:** {metrics['test_samples']:,}")
st.markdown("---")

# -------------------------------------------------------
# Performance Metrics
# -------------------------------------------------------
st.subheader("📈 Performance Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
c2.metric("Precision", f"{metrics['precision']:.4f}")
c3.metric("Recall", f"{metrics['recall']:.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("F1 Score", f"{metrics['f1_score']:.4f}")
c5.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
st.markdown("---")

# -------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------
st.subheader("📊 Confusion Matrix")
cm = np.array(metrics["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Interested", "Interested"])
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title("Confusion Matrix", fontsize=12)
plt.tight_layout(pad=0.5)
st.pyplot(fig, use_container_width=False)
st.markdown("---")

# -------------------------------------------------------
# Feature List
# -------------------------------------------------------
st.subheader("🧠 Features Used for Training")
feature_df = pd.DataFrame({"Feature Name": metrics["features"]})
st.dataframe(feature_df, use_container_width=True, hide_index=True)
st.markdown("---")

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
st.subheader("📋 Summary")
summary = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        "Value": [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["roc_auc"],
        ],
    }
)
st.dataframe(summary, use_container_width=True, hide_index=True)
