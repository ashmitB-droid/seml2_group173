"""
main.py

FastAPI application entrypoint.

Exception handling is centralised here: each typed exception from
src/exceptions.py maps to exactly one HTTP status code, in one place. Route
handlers therefore contain no try/except boilerplate, and adding a new
failure mode means adding one handler rather than editing every route.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import router
from src.config import API_DESCRIPTION, API_TITLE, API_VERSION
from src.exceptions import DataValidationError, InferenceError, ModelNotFoundError
from src.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)


@app.exception_handler(DataValidationError)
def handle_data_validation_error(request: Request, exc: DataValidationError) -> JSONResponse:
    """Client sent something we cannot accept -> 422."""
    logger.warning("Validation error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ModelNotFoundError)
def handle_model_not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    """Server cannot serve right now, but the request was fine -> 503."""
    logger.error("Model unavailable on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(InferenceError)
def handle_inference_error(request: Request, exc: InferenceError) -> JSONResponse:
    """Valid input the model still could not score -> 422."""
    logger.error("Inference error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


app.include_router(router)
