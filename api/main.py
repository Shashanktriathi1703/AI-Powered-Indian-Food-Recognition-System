"""
Nutriiya AI & Nutrition Intelligence - REST API Server
Built with FastAPI. Exposes the Indian Food Recognition & Nutrition Estimation System.
"""

import io
import base64
import os
import sys
from typing import Optional
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.pipeline import MealAnalyzerPipeline
from api.schemas import (
    MealAnalysisResponse, 
    Base64MealRequest, 
    HealthResponse
)

app = FastAPI(
    title="Nutriiya - AI Powered Indian Food Recognition API",
    description="Analyzes Indian meal images, detects multiple food items, estimates portion quantities, and calculates nutritional composition.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web and mobile frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline as singleton
pipeline: Optional[MealAnalyzerPipeline] = None

@app.on_event("startup")
def startup_event():
    global pipeline
    try:
        pipeline = MealAnalyzerPipeline()
        print("MealAnalyzerPipeline successfully initialized on startup.")
    except Exception as e:
        print(f"Failed to initialize MealAnalyzerPipeline: {e}")
        pipeline = None


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Nutriiya AI Indian Food Recognition System",
        "documentation": "/docs",
        "health_check": "/health",
        "primary_endpoint": "POST /analyze-meal"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    from models.food_detector import CLASSES
    return {
        "status": "healthy" if pipeline is not None else "degraded",
        "service": "Nutriiya AI Food & Nutrition Intelligence",
        "version": "1.0.0",
        "model_loaded": pipeline is not None and pipeline.detector.yolo_model is not None,
        "supported_classes": CLASSES
    }


@app.post("/analyze-meal", response_model=MealAnalysisResponse, tags=["Inference"])
async def analyze_meal_endpoint(
    file: UploadFile = File(..., description="JPEG or PNG image of an Indian meal plate"),
    include_visualization: bool = Query(False, description="Return annotated image as Base64 string"),
    conf_threshold: float = Query(0.45, ge=0.05, le=0.95, description="Confidence threshold for detection")
):
    """
    Main endpoint required by Nutriiya practical assignment.
    Accepts an uploaded image file, identifies food items, estimates portions in grams,
    and returns calorie, protein, carbohydrate, and fat values.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="AI inference engine is not ready."
        )

    # 1. Validate content type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"]
    if file.content_type and file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Please upload a JPEG or PNG image."
        )

    # 2. Read and decode image
    try:
        image_bytes = await file.read()
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file provided. Please provide a valid meal image."
            )
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image file. Could not decode image."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {str(e)}"
        )

    # 3. Execute AI pipeline
    try:
        result = pipeline.analyze_meal(
            image=image,
            include_visualization=include_visualization,
            conf_threshold=conf_threshold
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during meal analysis: {str(e)}"
        )

    # 4. Handle optional visualization Base64 serialization
    annotated_b64 = None
    if include_visualization and "annotated_image" in result:
        buf = io.BytesIO()
        result["annotated_image"].save(buf, format="JPEG", quality=90)
        annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "success": result["success"],
        "meal_items": result["meal_items"],
        "total_nutrition": result["total_nutrition"],
        "annotated_image_base64": annotated_b64
    }


@app.post("/analyze-meal-base64", response_model=MealAnalysisResponse, tags=["Inference"])
async def analyze_meal_base64_endpoint(payload: Base64MealRequest):
    """
    Alternative endpoint accepting Base64 encoded image strings in JSON body.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="AI inference engine is not ready."
        )

    try:
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        decoded = base64.b64decode(raw_b64)
        image = Image.open(io.BytesIO(decoded)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Base64 image data: {str(e)}"
        )

    result = pipeline.analyze_meal(
        image=image,
        include_visualization=payload.include_visualization,
        conf_threshold=payload.confidence_threshold
    )

    annotated_b64 = None
    if payload.include_visualization and "annotated_image" in result:
        buf = io.BytesIO()
        result["annotated_image"].save(buf, format="JPEG", quality=90)
        annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "success": result["success"],
        "meal_items": result["meal_items"],
        "total_nutrition": result["total_nutrition"],
        "annotated_image_base64": annotated_b64
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
