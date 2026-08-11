from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    id: int = Field(..., example=1)
    Gender: str = Field(..., example="Male")
    Age: int = Field(..., ge=18, le=100, example=35)
    Driving_License: int = Field(..., example=1)
    Region_Code: float = Field(..., example=28.0)
    Previously_Insured: int = Field(..., example=0)
    Vehicle_Age: str = Field(..., example="1-2 Year")
    Vehicle_Damage: str = Field(..., example="Yes")
    Annual_Premium: float = Field(..., example=40454.0)
    Policy_Sales_Channel: float = Field(..., example=26.0)
    Vintage: int = Field(..., example=217)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str