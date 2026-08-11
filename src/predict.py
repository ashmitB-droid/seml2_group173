"""
predict.py

Inference pipeline for the Health Insurance Cross-Sell model.

The Predictor loads its artefacts once at construction and is then reused
for every request, so a running API does not pay disk cost per prediction.

Note on the decision threshold: this module no longer calls
model.predict(), which silently hard-codes 0.50. It applies the shared
DECISION_THRESHOLD from config so the deployed decision rule matches the
one the reported metrics were computed under.
"""

from typing import Dict

import joblib
import pandas as pd

from src.config import DECISION_THRESHOLD, FEATURE_NAMES_PATH, MODEL_PATH
from src.exceptions import DataValidationError, InferenceError, ModelNotFoundError
from src.feature_engineering import feature_engineering
from src.logging_config import get_logger
from src.preprocessing import preprocess_inference_data

logger = get_logger(__name__)


class Predictor:
    """Serves predictions from the persisted model and feature order."""

    def __init__(self, threshold: float = DECISION_THRESHOLD):
        self.threshold = threshold
        try:
            self.model = joblib.load(MODEL_PATH)
            self.feature_names = joblib.load(FEATURE_NAMES_PATH)
        except FileNotFoundError as exc:
            logger.error("Model artefacts missing (looked in %s)", MODEL_PATH.parent)
            raise ModelNotFoundError(
                f"Model artefacts not found at {MODEL_PATH.parent}. "
                "Run `python -m src.train` first."
            ) from exc

        logger.info(
            "Predictor ready: %d features, decision threshold %.2f",
            len(self.feature_names),
            self.threshold,
        )

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the inference preprocessing chain and align column order.

        Column alignment is not cosmetic: XGBoost consumes positional
        arrays, so a reordered frame produces silently wrong predictions
        rather than an error.
        """
        df = preprocess_inference_data(df)
        df = feature_engineering(df)

        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            logger.error("Engineered frame is missing expected features: %s", missing)
            raise InferenceError(f"Missing engineered features: {missing}")

        return df[self.feature_names]

    def predict_single(self, data: Dict) -> Dict:
        """Score one customer record.

        Raises:
            DataValidationError: the record failed schema/category checks.
            InferenceError: the record was structurally valid but unscoreable.
        """
        try:
            prepared = self.prepare(pd.DataFrame([data]))
        except DataValidationError:
            # Client error - propagate unchanged so the API returns 422.
            raise
        except Exception as exc:
            logger.error("Failed to prepare record for inference: %s", exc)
            raise InferenceError(f"Could not prepare record: {exc}") from exc

        try:
            probability = float(self.model.predict_proba(prepared)[0][1])
        except Exception as exc:
            logger.error("Model scoring failed: %s", exc)
            raise InferenceError(f"Model could not score record: {exc}") from exc

        prediction = int(probability >= self.threshold)
        logger.info(
            "Scored record -> probability=%.4f threshold=%.2f prediction=%d",
            probability,
            self.threshold,
            prediction,
        )

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "label": "Interested" if prediction == 1 else "Not Interested",
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a dataframe of customer records."""
        if df.empty:
            logger.error("predict_batch() called with an empty dataframe")
            raise DataValidationError("Uploaded file contains no rows")

        logger.info("Batch scoring %d records", len(df))
        prepared = self.prepare(df)

        probabilities = self.model.predict_proba(prepared)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        result = df.copy()
        result["Prediction"] = predictions
        result["Probability"] = probabilities.round(4)
        result["Label"] = result["Prediction"].map({0: "Not Interested", 1: "Interested"})

        logger.info(
            "Batch complete: %d of %d scored as Interested",
            int(predictions.sum()),
            len(predictions),
        )
        return result


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
        "Vintage": 150,
    }
    logger.info("Sample prediction: %s", predictor.predict_single(sample))
