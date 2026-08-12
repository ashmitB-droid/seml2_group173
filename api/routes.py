"""
routes.py

HTTP endpoints for the cross-sell inference service.

API design decisions (Objective 1, Req. 5):
  * POST for scoring (it creates a prediction), GET for status and metrics;
  * every route declares an explicit response_model, so the OpenAPI schema
    is generated from the code rather than maintained by hand;
  * failures raise typed exceptions handled centrally in main.py, so route
    bodies stay free of try/except noise;
  * the model is loaded once at import, not per request.
"""

import json
from typing import List

import pandas as pd
from fastapi import APIRouter, File, Response, UploadFile, status

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.config import API_VERSION, DECISION_THRESHOLD, METRICS_PATH
from src.exceptions import DataValidationError, ModelNotFoundError
from src.logging_config import get_logger
from src.predict import Predictor

logger = get_logger(__name__)
router = APIRouter()

# Reject oversized uploads before reading them into memory - an unbounded
# read is a denial-of-service vector on a small container.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 10_000

try:
    predictor = Predictor()
except ModelNotFoundError:
    # Let the service start so /health can report the problem, rather than
    # crash-looping and giving operators nothing to inspect.
    logger.error("Starting API without a usable model - /predict will return 503")
    predictor = None


@router.get("/", tags=["ops"])
def home():
    """Service banner."""
    return {
        "project": "Health Insurance Cross-Sell Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness and readiness probe.

    Used by the load balancer and by the canary/shadow rollout described in
    the report: a new revision that cannot load its model reports
    model_loaded=false and never receives traffic.
    """
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_version=API_VERSION,
    )


@router.get("/metrics", tags=["ops"])
def metrics():
    """Return the metrics recorded at training time."""
    try:
        with open(METRICS_PATH) as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        logger.error("Metrics file not found at %s", METRICS_PATH)
        raise ModelNotFoundError(
            "Metrics are unavailable - the model has not been trained"
        ) from exc


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
    tags=["inference"],
)
def predict(request: PredictionRequest):
    """Score a single customer record."""
    if predictor is None:
        raise ModelNotFoundError("No model is loaded")

    result = predictor.predict_single(request.model_dump())
    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        label=result["label"],
        threshold=DECISION_THRESHOLD,
        model_version=API_VERSION,
    )


@router.post(
    "/batch-predict",
    response_model=List[dict],
    responses={
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
    },
    tags=["inference"],
)
async def batch_predict(response: Response, file: UploadFile = File(...)):
    """Score a CSV of customer records.

    Returns a JSON array of records, one per input row, so the response can
    be rendered directly by a dataframe consumer. Summary counts are exposed
    as X-Rows-Scored and X-Interested-Count headers rather than wrapped
    around the array: changing the body shape would silently break the
    Streamlit dashboard, which passes this response straight to
    st.dataframe().
    """
    if predictor is None:
        raise ModelNotFoundError("No model is loaded")

    if not file.filename.lower().endswith(".csv"):
        raise DataValidationError("Only .csv uploads are accepted")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        logger.warning("Rejected upload of %d bytes (limit %d)", len(contents), MAX_UPLOAD_BYTES)
        raise DataValidationError(f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")

    try:
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as exc:
        logger.error("Uploaded file could not be parsed as CSV: %s", exc)
        raise DataValidationError(f"Could not parse uploaded CSV: {exc}") from exc

    if len(df) > MAX_BATCH_ROWS:
        raise DataValidationError(f"Batch exceeds the {MAX_BATCH_ROWS}-row limit")

    result = predictor.predict_batch(df)

    response.headers["X-Rows-Scored"] = str(len(result))
    response.headers["X-Interested-Count"] = str(int((result["Prediction"] == 1).sum()))

    return result.to_dict(orient="records")