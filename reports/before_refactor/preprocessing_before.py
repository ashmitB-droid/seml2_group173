"""
preprocessing.py
Project: Health Insurance Cross Sell Prediction
This module contains all preprocessing functions.
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
# ==========================================================
# CONFIG
# ==========================================================
from src.config import ENCODER_PATH, TARGET, TRAIN_DATA_PATH, TEST_SIZE, RANDOM_STATE

# ==========================================================
# LOAD DATA
# ==========================================================
def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV dataset
    """
    return pd.read_csv(path)

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before-after} duplicate rows")
    return df

# ==========================================================
# MISSING VALUES
# ==========================================================
def check_missing_values(df: pd.DataFrame):
    missing = pd.DataFrame({
        "Missing": df.isnull().sum(),
        "Percentage": (df.isnull().mean()*100).round(2)
    })
    return missing

# ==========================================================
# DROP ID
# ==========================================================
def drop_id(df: pd.DataFrame):
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df

# ==========================================================
# MAP CATEGORICAL FEATURES
# ==========================================================
def map_features(df: pd.DataFrame):
    gender_map = {
        "Male": 1,
        "Female": 0
    }
    vehicle_age_map = {
        "< 1 Year": 0,
        "1-2 Year": 1,
        "> 2 Years": 2
    }
    damage_map = {
        "No": 0,
        "Yes": 1
    }
    df["Gender"] = df["Gender"].map(gender_map)
    df["Vehicle_Age"] = df["Vehicle_Age"].map(vehicle_age_map)
    df["Vehicle_Damage"] = df["Vehicle_Damage"].map(damage_map)
    return df

# ==========================================================
# CREATE AGE GROUP
# ==========================================================
def create_age_group(df):
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[18,25,35,45,55,65,100],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "65+"
        ]
    )
    return df


# ==========================================================
# ORDINAL ENCODER
# ==========================================================
def fit_encoder(df):
    encoder = OrdinalEncoder()
    df[["Age_Group"]] = encoder.fit_transform(
        df[["Age_Group"]]
    )
    joblib.dump(
        encoder,
        ENCODER_PATH
    )
    return df

def transform_encoder(df):
    encoder = joblib.load(
        ENCODER_PATH
    )
    df[["Age_Group"]] = encoder.transform(
        df[["Age_Group"]]
    )
    return df

# ==========================================================
# SPLIT DATA
# ==========================================================
def split_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

# ==========================================================
# COMPLETE TRAINING PIPELINE
# ==========================================================
def preprocess_training_data(df):
    df = remove_duplicates(df)
    df = drop_id(df)
    df = map_features(df)
    df = create_age_group(df)
    df = fit_encoder(df)
    return split_data(df)

# ==========================================================
# COMPLETE INFERENCE PIPELINE
# ==========================================================
def preprocess_inference_data(df):
    df = drop_id(df)
    df = map_features(df)
    df = create_age_group(df)
    df = transform_encoder(df)
    return df

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    train = load_data(TRAIN_DATA_PATH)
    print(train.head())
    print(check_missing_values(train))
    X_train, X_test, y_train, y_test = preprocess_training_data(train)
    print(X_train.shape)
    print(X_train.head())
    print(X_test.shape)