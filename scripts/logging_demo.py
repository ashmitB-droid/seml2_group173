"""
logging_demo.py

Exercises the failure paths of the pipeline so that INFO, WARNING and ERROR
records all appear in one run (Objective 1, Req. 3 - evidence).

Run:  python -m scripts.logging_demo
Log:  logs/application.log

Nothing here is part of the served application; it exists to demonstrate
that each severity level is reachable and that failures are typed.
"""

import numpy as np
import pandas as pd

from src.data_quality import missing_value_report, population_stability_index
from src.exceptions import DataValidationError, ModelNotFoundError
from src.logging_config import get_logger
from src.predict import Predictor
from src.preprocessing import create_age_group, load_data, remove_duplicates, validate_schema

logger = get_logger(__name__)

VALID_RECORD = {
    "id": 1,
    "Gender": "Male",
    "Age": 45,
    "Driving_License": 1,
    "Region_Code": 28.0,
    "Previously_Insured": 0,
    "Vehicle_Age": "> 2 Years",
    "Vehicle_Damage": "Yes",
    "Annual_Premium": 35000.0,
    "Policy_Sales_Channel": 26.0,
    "Vintage": 150,
}


def demo_info() -> pd.DataFrame:
    """INFO: the normal, successful path."""
    logger.info("--- INFO: normal operation ---")
    df = load_data("dataset/train.csv").head(5000)
    validate_schema(df, require_target=True)
    missing_value_report(df)
    return df


def demo_warning(df: pd.DataFrame) -> None:
    """WARNING: recoverable anomalies worth an operator's attention."""
    logger.info("--- WARNING: recoverable anomalies ---")

    # Duplicate rows: handled, but the operator should know.
    remove_duplicates(pd.concat([df.head(50), df.head(50)]))

    # An age outside the binning range: the row survives, the value does not.
    create_age_group(pd.DataFrame({"Age": [45, 120]}))

    # Moderate distribution drift.
    population_stability_index(df["Annual_Premium"], df["Annual_Premium"] * 1.15)


def demo_error(df: pd.DataFrame) -> None:
    """ERROR: failures that abort the operation, raised as typed exceptions."""
    logger.info("--- ERROR: failures ---")

    try:
        validate_schema(df.drop(columns=["Age"]))
    except DataValidationError as exc:
        logger.error("Caught expected schema failure: %s", exc)

    try:
        Predictor().predict_single({**VALID_RECORD, "Gender": "Other"})
    except (DataValidationError, ModelNotFoundError) as exc:
        logger.error("Caught expected category failure: %s", exc)

    # Significant drift is an ERROR: it means the model is scoring a
    # population it was not trained on.
    population_stability_index(df["Annual_Premium"], df["Annual_Premium"] * 3 + 50000)

    try:
        validate_schema(pd.DataFrame({"Age": [np.nan]}))
    except DataValidationError as exc:
        logger.error("Caught expected validation failure: %s", exc)


if __name__ == "__main__":
    logger.info("=== Logging demonstration started ===")
    frame = demo_info()
    demo_warning(frame)
    demo_error(frame)
    logger.info("=== Logging demonstration finished ===")
