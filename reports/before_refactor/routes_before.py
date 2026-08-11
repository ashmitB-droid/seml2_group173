import json
import pandas as pd
from fastapi import APIRouter, UploadFile, File
from api.schemas import (
    PredictionRequest,
    PredictionResponse
)
from src.predict import Predictor

predictor = Predictor()
router = APIRouter()

@router.get("/")
def home():
    return {"Project": "Health Insurance Prediction API running. Use /docs for API documentation.",  "status":"Running"}

@router.get("/ping")
def ping():
    return {"Project": "Health Insurance Prediction API working", "status": "success", "code": 200, "version": "1.0.0"}

@router.get("/metrics")
def metrics():
    with open("models/metrics.json") as f:
        return json.load(f)

@router.post( "/predict", response_model=PredictionResponse)
def predict(PredictionRequest: PredictionRequest):
    result = predictor.predict_single(
        PredictionRequest.model_dump()
    )
    return result

@router.post("/batch-predict")
async def batch_predict(
    file: UploadFile = File(...)
):
    df = pd.read_csv(file.file)
    result = predictor.predict_batch(df)
    return result.to_dict(
        orient="records")
