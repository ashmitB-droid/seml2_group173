"""
train.py
Train XGBoost model for
Health Insurance Cross Sell Prediction.
"""
import json
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from src.preprocessing import (
    load_data,
    preprocess_training_data
)
from src.feature_engineering import (
    feature_engineering
)
from src.config import (
    TRAIN_DATA_PATH,
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    METRICS_PATH,
    RANDOM_STATE
)
def train():
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)
    df = load_data(TRAIN_DATA_PATH)
    print(f"Dataset Shape : {df.shape}")
    print("=" * 60)
    print("Preprocessing")
    print("=" * 60)
    X_train, X_test, y_train, y_test = preprocess_training_data(df)
    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)
    # Handle Imbalanced Dataset
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = negative / positive
    print(f"Scale Pos Weight : {scale_pos_weight:.2f}")
    print("=" * 60)
    print("Training XGBoost")
    print("=" * 60)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=7,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=2,
        reg_alpha=0.5,
        reg_lambda=8,
        scale_pos_weight=3.5,
        n_jobs=-1
    )
    model.fit( X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    print("Training Completed")
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)
    #y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.40).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC : {roc_auc:.4f}")
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))
    print("=" * 60)
    print("Saving Model")
    print("=" * 60)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X_train.columns), FEATURE_NAMES_PATH)
    metrics = {
        "model": "XGBoost",
        "dataset": "Health Insurance Cross Sell",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": list(X_train.columns),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": confusion_matrix(
        y_test,
        y_pred
    ).tolist()
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)
    print(f"Model Saved : {MODEL_PATH}")
    print(f"Feature Names Saved : {FEATURE_NAMES_PATH}")
    print(f"Metrics Saved : {METRICS_PATH}")
    print("=" * 60)
    print("Training Finished")
    print("=" * 60)

if __name__ == "__main__":
    train()