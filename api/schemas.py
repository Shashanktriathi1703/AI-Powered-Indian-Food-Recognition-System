"""
Nutriiya AI - FastAPI Pydantic Request & Response Schemas
Strictly conforms to Nutriiya Candidate Practical Assignment specification.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field


class NutritionBreakdown(BaseModel):
    calories: Union[float, int] = Field(..., description="Estimated energy in kilocalories (kcal)")
    protein: Union[float, int] = Field(..., description="Estimated protein in grams (g)")
    carbs: Union[float, int] = Field(..., description="Estimated carbohydrates in grams (g)")
    fat: Union[float, int] = Field(..., description="Estimated fat in grams (g)")


class MealItem(BaseModel):
    food_name: str = Field(..., description="Predicted name of detected Indian dish")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0)")
    estimated_quantity_g: int = Field(..., description="Estimated portion weight in grams")
    nutrition: NutritionBreakdown = Field(..., description="Portion-scaled nutritional breakdown")


class MealAnalysisResponse(BaseModel):
    success: bool = Field(True, description="Indicates whether meal analysis succeeded")
    meal_items: List[MealItem] = Field(..., description="List of recognized dishes and their portions")
    total_nutrition: NutritionBreakdown = Field(..., description="Aggregated nutrition for the complete meal")
    annotated_image_base64: Optional[str] = Field(None, description="Optional Base64 JPEG of annotated meal")


class Base64MealRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG/PNG meal image")
    include_visualization: bool = Field(False, description="Whether to include visualized base64 image")
    confidence_threshold: float = Field(0.25, ge=0.05, le=0.95, description="Minimum detection confidence")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    supported_classes: List[str]
