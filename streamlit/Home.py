import streamlit as st

st.set_page_config(page_title="Health Insurance Cross Sell", page_icon="🚗", layout="wide")

st.title("🚗 Health Insurance Cross Sell Prediction")
st.markdown("---")
st.write(
    """
This application predicts whether a customer is likely to purchase
Vehicle Insurance using an XGBoost model.
### Features
- Single Prediction
- Batch Prediction
- Model Metrics
- FastAPI Backend
- XGBoost
"""
)
