"""
preprocessing.py
Project: Health Insurance Cross Sell Prediction

Loads raw data and turns it into the encoded form the model expects.

Production concerns handled here (Objective 1, Req. 3):
  * every stage logs at a meaningful level rather than printing;
  * category values are validated instead of silently becoming NaN;
  * failures raise typed exceptions the API can map to status codes.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from src.config import (
    AGE_BINS,
    AGE_LABELS,
    ALLOWED_CATEGORIES,
    ENCODER_PATH,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    TARGET,
    TEST_SIZE,
    TRAIN_DATA_PATH,
)
from src.exceptions import DataValidationError, ModelNotFoundError
from src.logging_config import get_logger

logger = get_logger(__name__)

GENDER_MAP = {"Male": 1, "Female": 0}
VEHICLE_AGE_MAP = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
DAMAGE_MAP = {"No": 0, "Yes": 1}


# ==========================================================
# LOAD DATA
# ==========================================================
def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset from disk.

    Raises:
        FileNotFoundError: if the path does not exist.
        DataValidationError: if the file is empty or unparseable.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Dataset not found at %s", path)
        raise
    except pd.errors.EmptyDataError as exc:
        logger.error("Dataset at %s is empty: %s", path, exc)
        raise DataValidationError(f"Dataset at {path} is empty") from exc
    except pd.errors.ParserError as exc:
        logger.error("Dataset at %s could not be parsed: %s", path, exc)
        raise DataValidationError(f"Malformed CSV at {path}") from exc

    logger.info("Loaded %d rows and %d columns from %s", df.shape[0], df.shape[1], path)
    return df


# ==========================================================
# SCHEMA VALIDATION
# ==========================================================
def validate_schema(df: pd.DataFrame, require_target: bool = False) -> None:
    """Check that a raw dataframe satisfies the schema contract.

    This is the first of our two data-quality metrics (Objective 2, Req. 8b):
    a hard pass/fail gate on structure and category membership.

    Raises:
        DataValidationError: on a missing column or an unrecognised category.
    """
    expected = list(REQUIRED_COLUMNS)
    if require_target:
        expected.append(TARGET)

    missing = [col for col in expected if col not in df.columns]
    if missing:
        logger.error("Schema validation failed - missing columns: %s", missing)
        raise DataValidationError(f"Missing required columns: {missing}")

    for column, allowed in ALLOWED_CATEGORIES.items():
        unexpected = set(df[column].dropna().unique()) - set(allowed)
        if unexpected:
            logger.error(
                "Schema validation failed - column %s contains unrecognised values %s "
                "(allowed: %s)",
                column,
                sorted(unexpected),
                allowed,
            )
            raise DataValidationError(
                f"Column '{column}' contains unrecognised values: {sorted(unexpected)}"
            )

    logger.info("Schema validation passed for dataframe with shape %s", df.shape)


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, logging how many were removed."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    if removed:
        logger.warning(
            "Removed %d duplicate rows (%.2f%% of input)", removed, 100 * removed / before
        )
    else:
        logger.info("No duplicate rows found in %d rows", before)
    return df


# ==========================================================
# MISSING VALUES
# ==========================================================
def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missing-value report."""
    missing = pd.DataFrame(
        {
            "Missing": df.isnull().sum(),
            "Percentage": (df.isnull().mean() * 100).round(2),
        }
    )
    worst = missing["Percentage"].max()
    if worst > 0:
        logger.warning("Highest missing-value rate across columns: %.2f%%", worst)
    else:
        logger.info("No missing values detected")
    return missing


# ==========================================================
# DROP ID
# ==========================================================
def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the identifier column; it carries no predictive signal."""
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df


# ==========================================================
# MAP CATEGORICAL FEATURES
# ==========================================================
def map_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map the three categorical columns to integer codes.

    Previously this used a bare .map(), which turns any unrecognised value
    into NaN. XGBoost treats NaN as a legitimate missing value, so a record
    with Gender="Other" was scored and returned a confident probability
    instead of being rejected. We now validate first and fail loudly.
    """
    df = df.copy()
    validate_schema(df)

    df["Gender"] = df["Gender"].map(GENDER_MAP)
    df["Vehicle_Age"] = df["Vehicle_Age"].map(VEHICLE_AGE_MAP)
    df["Vehicle_Damage"] = df["Vehicle_Damage"].map(DAMAGE_MAP)
    return df


# ==========================================================
# CREATE AGE GROUP
# ==========================================================
def create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket Age into ordered bands.

    include_lowest=True is essential: without it pd.cut excludes the left
    edge, so Age == 18 fell into no bin at all and became NaN.
    """
    df = df.copy()
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
    )

    unbinned = int(df["Age_Group"].isna().sum())
    if unbinned:
        logger.warning(
            "%d rows have an Age outside the range %d-%d and could not be binned",
            unbinned,
            AGE_BINS[0],
            AGE_BINS[-1],
        )
    return df


# ==========================================================
# ORDINAL ENCODER
# ==========================================================
def fit_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """Fit the Age_Group encoder on training data and persist it."""
    df = df.copy()
    encoder = OrdinalEncoder()
    df[["Age_Group"]] = encoder.fit_transform(df[["Age_Group"]])

    ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, ENCODER_PATH)
    logger.info("Age_Group encoder fitted and saved to %s", ENCODER_PATH)
    return df


def transform_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the previously fitted encoder at inference time.

    Raises:
        ModelNotFoundError: if the encoder has not been fitted yet.
        DataValidationError: if a value was not seen during training.
    """
    df = df.copy()
    try:
        encoder = joblib.load(ENCODER_PATH)
    except FileNotFoundError as exc:
        logger.error("Encoder not found at %s - has the model been trained?", ENCODER_PATH)
        raise ModelNotFoundError(f"Encoder artefact missing at {ENCODER_PATH}") from exc

    try:
        df[["Age_Group"]] = encoder.transform(df[["Age_Group"]])
    except ValueError as exc:
        logger.error("Age_Group encoding failed: %s", exc)
        raise DataValidationError(f"Could not encode Age_Group: {exc}") from exc

    return df


# ==========================================================
# SPLIT DATA
# ==========================================================
def split_data(df: pd.DataFrame):
    """Stratified train/test split on the target column."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split into %d training and %d test rows (positive rate %.4f)",
        len(X_train),
        len(X_test),
        y.mean(),
    )
    return X_train, X_test, y_train, y_test


# ==========================================================
# COMPLETE TRAINING PIPELINE
# ==========================================================
def preprocess_training_data(df: pd.DataFrame):
    """Full preprocessing chain for training."""
    logger.info("Starting training preprocessing on %d rows", len(df))
    validate_schema(df, require_target=True)

    df = remove_duplicates(df)
    df = drop_id(df)
    df = map_features(df)
    df = create_age_group(df)
    df = fit_encoder(df)
    return split_data(df)


# ==========================================================
# COMPLETE INFERENCE PIPELINE
# ==========================================================
def preprocess_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing chain for inference.

    Deliberately mirrors preprocess_training_data, minus the steps that
    would leak or refit state (no duplicate removal, no encoder refit).
    """
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
    logger.info("Missing value report:\n%s", check_missing_values(train))
    X_train, X_test, y_train, y_test = preprocess_training_data(train)
    logger.info("Training feature matrix shape: %s", X_train.shape)
