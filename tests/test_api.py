"""
API integration tests (Objective 2, Req. 6 - integration tests).

Unlike the unit tests, these exercise the whole stack together: routing,
pydantic validation, the preprocessing chain, the model, the exception
handlers and status-code mapping. FastAPI's TestClient runs the real
application in-process, so no server needs to be started.
"""

import io

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ----------------------------------------------------------
# Operational endpoints
# ----------------------------------------------------------
def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_metrics_endpoint_returns_recorded_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert {"accuracy", "roc_auc", "threshold"} <= set(response.json())


def test_metrics_payload_satisfies_dashboard_contract(client):
    """Regression test.

    streamlit/pages/metrics.py indexes these keys directly, so removing any
    one of them crashes the dashboard with a KeyError rather than failing a
    test. metrics.json is a published contract, not an internal file.
    """
    required = {
        "model",
        "dataset",
        "train_samples",
        "test_samples",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
    }
    assert required <= set(client.get("/metrics").json())


# ----------------------------------------------------------
# Happy path
# ----------------------------------------------------------
def test_predict_returns_well_formed_response(client, valid_record):
    response = client.post("/predict", json=valid_record)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["threshold"] == pytest.approx(0.40)


# ----------------------------------------------------------
# Input validation (Req. 9 - security)
# ----------------------------------------------------------
@pytest.mark.parametrize(
    "field,value",
    [
        ("Gender", "Other"),
        ("Vehicle_Age", "3 Years"),
        ("Vehicle_Damage", "maybe"),
        ("Age", 5),
        ("Age", 150),
        ("Annual_Premium", -100),
        ("Previously_Insured", 7),
    ],
)
def test_invalid_field_values_are_rejected(client, valid_record, field, value):
    """Every one of these previously reached the model. Gender="Other"
    returned a confident 0.72 "Interested" because .map() turned it into
    NaN and XGBoost scored NaN as missing."""
    response = client.post("/predict", json={**valid_record, field: value})
    assert response.status_code == 422


def test_missing_required_field_is_rejected(client, valid_record):
    incomplete = {k: v for k, v in valid_record.items() if k != "Age"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_boundary_age_is_accepted(client, valid_record):
    """Age == 18 is the documented minimum and must be scoreable."""
    assert client.post("/predict", json={**valid_record, "Age": 18}).status_code == 200


# ----------------------------------------------------------
# Batch endpoint
# ----------------------------------------------------------
def test_batch_predict_scores_uploaded_csv(client, raw_dataframe):
    csv_bytes = raw_dataframe.drop(columns=["Response"]).head(20).to_csv(index=False).encode()
    response = client.post(
        "/batch-predict",
        files={"file": ("customers.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body, list), "Streamlit passes this straight to st.dataframe()"
    assert len(body) == 20
    assert response.headers["X-Rows-Scored"] == "20"


def test_batch_predict_rejects_non_csv_upload(client):
    response = client.post(
        "/batch-predict",
        files={"file": ("payload.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")},
    )
    assert response.status_code == 422