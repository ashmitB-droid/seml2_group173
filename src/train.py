"""
train.py

Trains the XGBoost cross-sell classifier.

Refactored into a ModelTrainer class (Objective 1, Req. 1) so that fitting,
evaluation and persistence are separately callable and separately testable.
The previous version was a single 100-line train() function that did all
three and printed as it went, which meant a test could not evaluate a model
without also retraining and re-saving it.
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import (
    DECISION_THRESHOLD,
    FEATURE_NAMES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TRAIN_DATA_PATH,
)
from src.feature_engineering import feature_engineering
from src.logging_config import get_logger
from src.preprocessing import load_data, preprocess_training_data

logger = get_logger(__name__)

# Sanity floor: below this the model is barely better than guessing and
# something upstream (labels, features, encoding) is almost certainly wrong.
MIN_ACCEPTABLE_AUC = 0.60


class ModelTrainer:
    """Fits, evaluates and persists the cross-sell classifier."""

    def __init__(self, random_state: int = RANDOM_STATE, **overrides):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": random_state,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 2,
            "reg_alpha": 0.5,
            "reg_lambda": 8,
            "scale_pos_weight": 3.5,
            "n_jobs": -1,
        }
        params.update(overrides)
        self.params = params
        self.model = XGBClassifier(**params)
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, eval_set=None) -> "ModelTrainer":
        """Fit the classifier.

        Raises:
            ValueError: on an empty training set or a single-class target,
                either of which makes the resulting model meaningless.
        """
        if len(X_train) == 0:
            logger.error("fit() called with an empty training matrix")
            raise ValueError("Cannot train on an empty dataset")

        if y_train.nunique() < 2:
            logger.error("fit() called with only one class present in the target")
            raise ValueError("Training target must contain both classes")

        logger.info("Fitting XGBClassifier on %d rows, %d features", *X_train.shape)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.feature_names = list(X_train.columns)
        logger.info("Model fitted successfully")
        return self

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = DECISION_THRESHOLD
    ) -> dict:
        """Compute model-quality metrics (Objective 2, Req. 8a).

        The threshold defaults to the shared DECISION_THRESHOLD so that the
        metrics we report describe the decision rule the API actually
        applies. Passing a different value here is fine for threshold
        analysis, but the reported figures must use the deployed one.
        """
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        if metrics["roc_auc"] < MIN_ACCEPTABLE_AUC:
            logger.error(
                "ROC-AUC %.4f is below the %.2f sanity floor - inspect features and labels",
                metrics["roc_auc"],
                MIN_ACCEPTABLE_AUC,
            )
        else:
            logger.info(
                "Evaluation at threshold %.2f -> accuracy=%.4f precision=%.4f "
                "recall=%.4f f1=%.4f roc_auc=%.4f",
                threshold,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["roc_auc"],
            )

        logger.debug("Classification report:\n%s", classification_report(y_test, y_pred))
        return metrics

    def overfit_small_batch(self, X: pd.DataFrame, y: pd.Series, n: int = 50) -> float:
        """Train a deliberately unregularised model on a tiny batch.

        A healthy training loop can memorise a handful of rows. If it cannot,
        the fault is structural - features not reaching the model, labels
        misaligned - rather than a matter of tuning. Used by the training
        test in Objective 2, Req. 7a.
        """
        X_small, y_small = X.iloc[:n], y.iloc[:n]
        probe = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.3,
            reg_lambda=0,
            random_state=self.params["random_state"],
        )
        probe.fit(X_small, y_small, verbose=False)
        accuracy = float(probe.score(X_small, y_small))
        logger.info("Small-batch overfit probe on %d rows: accuracy=%.4f", n, accuracy)
        return accuracy

    def save(self, metrics: dict = None) -> None:
        """Persist the model, the feature order and the metrics."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.feature_names, FEATURE_NAMES_PATH)
        logger.info("Model saved to %s", MODEL_PATH)

        if metrics is not None:
            payload = dict(metrics)
            payload.update(
                {
                    "model": "XGBoost",
                    "dataset": "Health Insurance Cross Sell",
                    "features": self.feature_names,
                }
            )
            with open(METRICS_PATH, "w") as handle:
                json.dump(payload, handle, indent=2)
            logger.info("Metrics written to %s", METRICS_PATH)


def run_training() -> dict:
    """End-to-end training pipeline: load -> preprocess -> engineer -> fit -> save."""
    logger.info("=== Training pipeline started ===")

    df = load_data(TRAIN_DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_training_data(df)

    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    trainer = ModelTrainer()
    trainer.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    metrics = trainer.evaluate(X_test, y_test)
    trainer.save(metrics=metrics)

    logger.info("=== Training pipeline finished ===")
    return metrics


if __name__ == "__main__":
    run_training()
