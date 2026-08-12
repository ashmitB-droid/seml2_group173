"""
schemas.py

Request and response contracts for the inference API.

The categorical fields are typed as Literal rather than str. This is the
API's primary input-validation control (Objective 2, Req. 9 - security).
With a plain `str` annotation, a request carrying Gender="Other" passed
validation, reached .map(), became NaN, and was scored by XGBoost as a
missing value - so the caller received a confident-looking probability for
input the model had never been trained on. Literal rejects it at the edge
with a 422 and the malformed value never reaches the model.
"""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """One raw customer record."""

    id: int = Field(..., description="Customer identifier", json_schema_extra={"example": 1})
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=100, description="Customer age in years")
    Driving_License: Literal[0, 1]
    Region_Code: float = Field(..., ge=0, le=100)
    Previously_Insured: Literal[0, 1]
    Vehicle_Age: Literal["< 1 Year", "1-2 Year", "> 2 Years"]
    Vehicle_Damage: Literal["Yes", "No"]
    Annual_Premium: float = Field(..., gt=0, le=1_000_000)
    Policy_Sales_Channel: float = Field(..., ge=0, le=200)
    Vintage: int = Field(..., ge=0, le=400)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = Interested, 0 = Not Interested")
    probability: float = Field(..., ge=0.0, le=1.0)
    label: str
    threshold: float = Field(..., description="Decision threshold applied to the probability")
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ErrorResponse(BaseModel):
    detail: str