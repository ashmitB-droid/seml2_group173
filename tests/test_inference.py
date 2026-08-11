"""
Inference tests (Objective 2, Req. 7b).

Three kinds of check:

  * output shape and range - the response is always well-formed;
  * directional tests - changing a feature moves the prediction the way
    domain knowledge says it should;
  * invariance tests - changing something irrelevant must NOT move it.

Directional and invariance tests are the ML equivalent of assertions.
They hold for any correctly-trained model, so they keep working after a
retrain, unlike a test that pins an exact probability.
"""

import pandas as pd
import pytest

from src.exceptions import DataValidationError
from src.predict import Predictor


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


# ----------------------------------------------------------
# Output shape and range
# ----------------------------------------------------------
def test_prediction_output_shape_and_range(predictor, valid_record):
    result = predictor.predict_single(valid_record)

    assert set(result) == {"prediction", "probability", "label"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("Interested", "Not Interested")


def test_label_is_consistent_with_prediction(predictor, valid_record):
    result = predictor.predict_single(valid_record)
    expected = "Interested" if result["prediction"] == 1 else "Not Interested"
    assert result["label"] == expected


def test_prediction_matches_threshold_rule(predictor, valid_record):
    """The returned class must be exactly probability >= threshold."""
    result = predictor.predict_single(valid_record)
    assert result["prediction"] == int(result["probability"] >= predictor.threshold)


def test_batch_output_shape(predictor, raw_dataframe):
    batch = raw_dataframe.drop(columns=["Response"]).head(25)
    result = predictor.predict_batch(batch)

    assert len(result) == len(batch)
    assert {"Prediction", "Probability", "Label"} <= set(result.columns)
    assert result["Probability"].between(0, 1).all()


# ----------------------------------------------------------
# Directional tests
# ----------------------------------------------------------
def test_previously_insured_lowers_predicted_interest(predictor, valid_record):
    """A customer who already holds vehicle insurance should not be scored
    as more likely to want a new policy than an identical customer who
    does not."""
    not_insured = predictor.predict_single(dict(valid_record, Previously_Insured=0))
    insured = predictor.predict_single(dict(valid_record, Previously_Insured=1))
    assert insured["probability"] <= not_insured["probability"]


def test_vehicle_damage_raises_predicted_interest(predictor, valid_record):
    """Prior damage is the strongest signal of interest in the business
    domain, so it must not lower the score."""
    undamaged = predictor.predict_single(dict(valid_record, Vehicle_Damage="No"))
    damaged = predictor.predict_single(dict(valid_record, Vehicle_Damage="Yes"))
    assert damaged["probability"] >= undamaged["probability"]


# ----------------------------------------------------------
# Invariance test
# ----------------------------------------------------------
def test_prediction_is_invariant_to_customer_id(predictor, valid_record):
    """The id column is dropped before modelling, so it must have no effect.
    If this fails, an identifier is leaking into the feature matrix - a
    classic source of models that score well offline and fail in production.
    """
    first = predictor.predict_single(dict(valid_record, id=1))
    second = predictor.predict_single(dict(valid_record, id=999_999))
    assert first["probability"] == second["probability"]


def test_single_and_batch_scoring_agree(predictor, valid_record):
    """The same record must score identically whether sent alone or in a
    batch. Encoding that depends on batch composition is a common source of
    train/serve skew."""
    single = predictor.predict_single(valid_record)
    batch = predictor.predict_batch(pd.DataFrame([valid_record]))
    assert batch["Probability"].iloc[0] == pytest.approx(single["probability"], abs=1e-4)


# ----------------------------------------------------------
# Error handling
# ----------------------------------------------------------
def test_invalid_category_raises_validation_error(predictor, valid_record):
    with pytest.raises(DataValidationError):
        predictor.predict_single(dict(valid_record, Gender="Other"))


def test_minimum_age_does_not_crash(predictor, valid_record):
    """Regression test for the Age == 18 binning crash."""
    result = predictor.predict_single(dict(valid_record, Age=18))
    assert 0.0 <= result["probability"] <= 1.0


def test_empty_batch_is_rejected(predictor, raw_dataframe):
    empty = raw_dataframe.drop(columns=["Response"]).head(0)
    with pytest.raises(DataValidationError):
        predictor.predict_batch(empty)
