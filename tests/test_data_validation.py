"""
Data-validation tests (Objective 2, Req. 6 - data validation tests;
Req. 8b - data-quality metrics).

These cover the schema gate and the drift metric. They are the tests most
likely to fire in production, because upstream data changes far more often
than the code does.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_quality import drift_report, missing_value_report, population_stability_index
from src.exceptions import DataValidationError
from src.preprocessing import create_age_group, map_features, validate_schema


# ----------------------------------------------------------
# Schema validation
# ----------------------------------------------------------
def test_valid_dataframe_passes_schema_validation(raw_dataframe):
    validate_schema(raw_dataframe, require_target=True)  # must not raise


def test_missing_required_column_is_rejected(raw_dataframe):
    broken = raw_dataframe.drop(columns=["Age"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_schema(broken)


def test_unrecognised_category_is_rejected(raw_dataframe):
    """Regression test.

    Gender="Other" previously passed straight through .map(), became NaN,
    and was scored by XGBoost as a missing value - so a malformed request
    received a confident probability instead of an error.
    """
    broken = raw_dataframe.copy()
    broken.loc[0, "Gender"] = "Other"
    with pytest.raises(DataValidationError, match="unrecognised values"):
        validate_schema(broken)


def test_map_features_rejects_bad_category_instead_of_producing_nan(raw_dataframe):
    broken = raw_dataframe.head(10).copy()
    broken.loc[0, "Vehicle_Damage"] = "maybe"
    with pytest.raises(DataValidationError):
        map_features(broken)


# ----------------------------------------------------------
# Age binning boundary (regression)
# ----------------------------------------------------------
def test_minimum_age_is_binned_not_dropped():
    """Regression test.

    pd.cut excludes the left edge by default, so Age == 18 fell outside
    every bin and became NaN - then OrdinalEncoder raised "Found unknown
    categories [nan]". The API schema accepts Age >= 18, so this was a
    crash reachable from a perfectly valid request.
    """
    df = pd.DataFrame({"Age": [18, 19, 40, 100]})
    binned = create_age_group(df)
    assert binned["Age_Group"].isna().sum() == 0
    assert binned.loc[0, "Age_Group"] == "18-25"


# ----------------------------------------------------------
# Data-quality metric 1: missing values
# ----------------------------------------------------------
def test_missing_value_report_detects_injected_gaps(raw_dataframe):
    df = raw_dataframe.copy()
    df.loc[df.index[:100], "Annual_Premium"] = np.nan

    report = missing_value_report(df)
    assert report["Annual_Premium"] == pytest.approx(100 / len(df), rel=1e-3)
    assert report["Age"] == 0.0


# ----------------------------------------------------------
# Data-quality metric 2: drift (PSI)
# ----------------------------------------------------------
def test_psi_is_near_zero_for_identical_distributions(raw_dataframe):
    psi = population_stability_index(
        raw_dataframe["Annual_Premium"], raw_dataframe["Annual_Premium"]
    )
    assert psi < 0.01


def test_psi_flags_a_shifted_distribution(raw_dataframe):
    """A premium distribution shifted upward must register as drift."""
    shifted = raw_dataframe["Annual_Premium"] * 2 + 20000
    psi = population_stability_index(raw_dataframe["Annual_Premium"], shifted)
    assert psi > 0.25


def test_psi_increases_monotonically_with_shift_size(raw_dataframe):
    """Larger distribution shifts must produce larger PSI values."""
    reference = raw_dataframe["Annual_Premium"]
    small = population_stability_index(reference, reference + 2000)
    large = population_stability_index(reference, reference + 20000)
    assert large > small


def test_drift_report_covers_requested_columns(raw_dataframe):
    report = drift_report(raw_dataframe, raw_dataframe, ["Age", "Annual_Premium", "Vintage"])
    assert set(report) == {"Age", "Annual_Premium", "Vintage"}
    assert all(value < 0.01 for value in report.values())
