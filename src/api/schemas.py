# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import List

# =============================================================================
#  API Request Schemas
# =============================================================================

class RentPredictionRequest(BaseModel):
    """
    Defines the shape of a single prediction request.
    Pydantic performs automatic data validation based on these type hints.
    """
    size_sq_ft: float = Field(
        ..., 
        gt=0, 
        description="Total square footage of the property."
    )
    bedrooms: int = Field(
        ..., 
        ge=1, 
        description="Number of bedrooms (BHK)."
    )
    localityName: str = Field(
        ..., 
        min_length=1,
        description="The name of the locality or neighborhood in Delhi."
    )
    propertyType: str = Field(
        ...,
        min_length=1,
        description="Type of the property (e.g., 'Apartment', 'Independent Floor')."
    )

    class Config:
        """Example to show in API docs"""
        json_schema_extra = {
            "example": {
                "size_sq_ft": 1000,
                "bedrooms": 2,
                "localityName": "Safdarjung Enclave",
                "propertyType": "Apartment"
            }
        }


# =============================================================================
#  API Response Schemas
# =============================================================================

class PredictionResponse(BaseModel):
    """
    Defines the shape of the prediction response.
    """
    predicted_rent: float
    confidence_interval: List[float]
    prediction_time: str