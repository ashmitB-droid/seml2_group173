"""
conftest.py

Shared fixtures for the test suite.

Tests run against a small deterministic sample drawn from the real dataset
rather than the full 381k rows. Two reasons: the suite finishes in seconds
so it can run on every commit, and a fixed random_state means a failure is
reproducible instead of depending on which rows happened to be sampled.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRAIN_DATA_PATH  # noqa: E402
from src.feature_engineering import feature_engineering  # noqa: E402
from src.preprocessing import (  # noqa: E402
    create_age_group,
    drop_id,
    fit_encoder,
    map_features,
)

SAMPLE_ROWS = 4000
SEED = 42


@pytest.fixture(scope="session")
def raw_dataframe() -> pd.DataFrame:
    """A stratified-ish sample of the real raw data, with both classes present."""
    df = pd.read_csv(TRAIN_DATA_PATH)
    positives = df[df["Response"] == 1].sample(SAMPLE_ROWS // 2, random_state=SEED)
    negatives = df[df["Response"] == 0].sample(SAMPLE_ROWS // 2, random_state=SEED)
    return (
        pd.concat([positives, negatives]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    )


@pytest.fixture(scope="session")
def engineered_features(raw_dataframe):
    """Raw sample carried through the full training preprocessing chain."""
    df = raw_dataframe.copy()
    y = df["Response"]
    df = df.drop(columns=["Response"])

    df = drop_id(df)
    df = map_features(df)
    df = create_age_group(df)
    df = fit_encoder(df)
    df = feature_engineering(df)
    return df, y


@pytest.fixture()
def valid_record() -> dict:
    """A single well-formed customer record."""
    return {
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
