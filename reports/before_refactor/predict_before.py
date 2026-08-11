"""
predict.py
Prediction pipeline for
Health Insurance Cross Sell Prediction
"""

import joblib
import pandas as pd
from src.config import (
    MODEL_PATH,
    FEATURE_NAMES_PATH
)
from src.preprocessing import (
    preprocess_inference_data
)
from src.feature_engineering import (
    feature_engineering
)
class Predictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)
    # ======================================================
    # Prepare Data
    # ======================================================
    def prepare(self, df: pd.DataFrame):
        df = preprocess_inference_data(df)
        df = feature_engineering(df)
        # Ensure correct feature order
        df = df[self.feature_names]
        return df
    # ======================================================
    # Single Prediction
    # ======================================================
    def predict_single(self, data: dict):
        df = pd.DataFrame([data])
        df = self.prepare(df)
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "label": "Interested" if prediction == 1 else "Not Interested"
        }

    # ======================================================
    # Batch Prediction
    # ======================================================
    def predict_batch(self, df: pd.DataFrame):
        df_processed = self.prepare(df)
        predictions = self.model.predict(df_processed)
        probabilities = self.model.predict_proba(df_processed)[:, 1]
        result = df.copy()
        result["Prediction"] = predictions
        result["Probability"] = probabilities
        result["Label"] = result["Prediction"].map({
            0: "Not Interested",
            1: "Interested"
        })
        return result

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    predictor = Predictor()
    sample = {
        "Gender": "Male",
        "Age": 45,
        "Driving_License": 1,
        "Region_Code": 28,
        "Previously_Insured": 0,
        "Vehicle_Age": "> 2 Years",
        "Vehicle_Damage": "Yes",
        "Annual_Premium": 35000,
        "Policy_Sales_Channel": 26,
        "Vintage": 150
    }
    prediction = predictor.predict_single(sample)
    print(prediction)