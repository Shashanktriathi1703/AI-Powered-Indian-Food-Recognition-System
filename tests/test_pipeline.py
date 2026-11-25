"""
Unit and Integration Tests for Nutriiya AI System
Verifies pipeline components, portion estimation, nutrition calculation, and FastAPI API.
"""

import os
import sys
from PIL import Image
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DEFAULT_DEMO_IMAGE
from models.pipeline import MealAnalyzerPipeline
from models.portion_estimator import PortionEstimator
from models.nutrition_calculator import NutritionCalculator
from api.main import app, startup_event


def test_nutrition_calculator():
    """Verify nutrition calculation against ICMR-NIN reference values."""
    db = {
        "items": {
            "rice": {
                "display_name": "Rice",
                "reference_per_100g": {"calories": 130.0, "protein": 2.7, "carbs": 28.2, "fat": 0.3}
            },
            "roti": {
                "display_name": "Roti",
                "reference_per_100g": {"calories": 297.0, "protein": 9.5, "carbs": 56.0, "fat": 3.7}
            }
        }
    }
    calc = NutritionCalculator(db)
    
    # 150g Rice calculation
    res = calc.calculate_item_nutrition("rice", 150.0)
    nut = res["nutrition"]
    assert nut["calories"] == 195  # 130 * 1.5 = 195
    assert nut["protein"] in [4, 4.0, 4.1]
    assert nut["carbs"] in [42, 42.0, 42.3]
    assert nut["fat"] in [0.4, 0.5]
    
    # 40g Roti (1 standard chapati)
    res_roti = calc.calculate_item_nutrition("roti", 40.0)
    assert res_roti["nutrition"]["calories"] == 118.8 or res_roti["nutrition"]["calories"] == 119
    
    # Meal aggregation test
    items = [res, res_roti]
    total = calc.aggregate_meal_nutrition(items)
    assert total["calories"] > 300
    assert total["protein"] > 7


def test_portion_estimator():
    """Verify portion estimation bounds and plate scaling logic."""
    db = {
        "items": {
            "rice": {
                "density_g_per_cm3": 0.85,
                "shape_factor": 0.65,
                "typical_depth_cm": 2.6,
                "min_portion_g": 50.0,
                "max_portion_g": 350.0
            },
            "roti": {
                "piece_weight_g": 40.0,
                "min_portion_g": 35.0,
                "max_portion_g": 200.0
            }
        }
    }
    estimator = PortionEstimator(db)
    
    # Test rice bounding box on 640x640 image
    res_rice = estimator.estimate_portion(
        class_name="rice",
        bbox=(180, 280, 460, 550),
        img_width=640,
        img_height=640
    )
    assert 50 <= res_rice["estimated_quantity_g"] <= 350
    assert "uncertainty_interval_g" in res_rice
    assert res_rice["uncertainty_interval_g"]["min"] < res_rice["estimated_quantity_g"]
    
    # Test single roti bounding box
    res_roti = estimator.estimate_portion(
        class_name="roti",
        bbox=(50, 270, 230, 460),
        img_width=640,
        img_height=640
    )
    assert 35 <= res_roti["estimated_quantity_g"] <= 80


def test_pipeline_on_sample_thali():
    """Test full pipeline on the benchmark Nutriiya Thali image."""
    img_path = DEFAULT_DEMO_IMAGE
    assert os.path.exists(img_path), f"Benchmark image not found at {img_path}"
    
    pipeline = MealAnalyzerPipeline()
    img = Image.open(img_path)
    
    result = pipeline.analyze_meal(img, include_visualization=True)
    
    assert result["success"] is True
    assert len(result["meal_items"]) >= 2, f"Expected at least 2 items, got {len(result['meal_items'])}"
    
    detected_names = [item["food_name"] for item in result["meal_items"]]
    print("Detected items in meal_01_thali:", detected_names)
    
    # Verify at least 2 of the 4 expected core dishes are detected
    expected_set = ["Rice", "Dal", "Roti", "Paneer curry", "Paneer Curry", "Biryani", "Chole"]
    matches = [name for name in detected_names if any(e.lower() in name.lower() for e in expected_set)]
    assert len(matches) >= 2, f"Expected at least 2 Indian food items, got: {detected_names}"
        
    assert result["total_nutrition"]["calories"] > 0
    assert result["total_nutrition"]["protein"] > 0
    assert "annotated_image" in result


def test_fastapi_endpoints():
    """Test FastAPI /health and /analyze-meal REST endpoints."""
    startup_event()
    client = TestClient(app)
    
    # Test /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    data = health_resp.json()
    assert data["status"] in ["healthy", "degraded"]
    assert len(data["supported_classes"]) == 10
    
    # Test /analyze-meal with sample image
    img_path = DEFAULT_DEMO_IMAGE
    with open(img_path, "rb") as f:
        resp = client.post(
            "/analyze-meal",
            files={"file": ("meal_01_thali.jpg", f, "image/jpeg")},
            params={"include_visualization": "true"}
        )
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["success"] is True
    assert "meal_items" in res_data
    assert "total_nutrition" in res_data
    assert len(res_data["meal_items"]) >= 4
    assert res_data["annotated_image_base64"] is not None
    
    # Test error handling with invalid file
    bad_resp = client.post(
        "/analyze-meal",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert bad_resp.status_code == 400


if __name__ == "__main__":
    print("Running tests manually...")
    test_nutrition_calculator()
    print("test_nutrition_calculator passed.")
    test_portion_estimator()
    print("test_portion_estimator passed.")
    test_pipeline_on_sample_thali()
    print("test_pipeline_on_sample_thali passed.")
    test_fastapi_endpoints()
    print("test_fastapi_endpoints passed.")
    print("ALL TESTS PASSED SUCCESSFULLY!")
