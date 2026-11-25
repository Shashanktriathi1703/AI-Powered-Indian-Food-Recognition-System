"""
Nutriiya AI - End-to-End Meal Intelligence Pipeline
Integrates Detection, Portion Estimation, and Nutritional Calculation.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from PIL import Image

from models.food_detector import FoodDetector
from models.portion_estimator import PortionEstimator
from models.nutrition_calculator import NutritionCalculator

BASE_DIR = r"C:\Users\Shashank Tripathi\.gemini\antigravity\scratch\nutriiya-indian-food-ai"
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "models", "weights", "best.pt")
DEFAULT_DB = os.path.join(BASE_DIR, "data", "nutrition_db.json")


class MealAnalyzerPipeline:
    """
    High-level orchestrator for analyzing Indian meals from images.
    """

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS, nutrition_db_path: str = DEFAULT_DB):
        # 1. Load nutrition database
        with open(nutrition_db_path, "r") as f:
            self.nutrition_db = json.load(f)

        # 2. Initialize engines
        self.detector = FoodDetector(weights_path=weights_path)
        self.portion_estimator = PortionEstimator(nutrition_db=self.nutrition_db)
        self.nutrition_calculator = NutritionCalculator(nutrition_db=self.nutrition_db)

    def analyze_meal(
        self, 
        image: Image.Image, 
        include_visualization: bool = False, 
        conf_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """
        Executes end-to-end analysis on a meal image.
        
        Returns JSON compliant with Nutriiya Candidate Assignment:
        {
            "success": bool,
            "meal_items": [
                {
                    "food_name": str,
                    "confidence": float,
                    "estimated_quantity_g": int,
                    "nutrition": {
                        "calories": float/int,
                        "protein": float/int,
                        "carbs": float/int,
                        "fat": float/int
                    }
                }
            ],
            "total_nutrition": {
                "calories": float/int,
                "protein": float/int,
                "carbs": float/int,
                "fat": float/int
            }
        }
        """
        start_time = time.time()
        w, h = image.size

        # 1. Detect food items (Task 1)
        self.detector.conf_threshold = conf_threshold
        detections = self.detector.detect(image)

        if not detections:
            return {
                "success": True,
                "meal_items": [],
                "total_nutrition": {
                    "calories": 0,
                    "protein": 0,
                    "carbs": 0,
                    "fat": 0
                },
                "meta": {
                    "message": "No food items detected in the image.",
                    "inference_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }

        meal_items = []
        portion_records = []

        # 2. Estimate portions & calculate nutrition for each item (Tasks 2 & 3)
        for det in detections:
            portion_res = self.portion_estimator.estimate_portion(
                class_name=det["class_name"],
                bbox=det["bbox"],
                img_width=w,
                img_height=h
            )
            est_g = portion_res["estimated_quantity_g"]
            portion_records.append(portion_res)

            nut_res = self.nutrition_calculator.calculate_item_nutrition(
                food_name=det["class_name"],
                quantity_g=est_g
            )

            meal_item_entry = {
                "food_name": det["display_name"],
                "confidence": det["confidence"],
                "estimated_quantity_g": est_g,
                "nutrition": nut_res["nutrition"]
            }
            meal_items.append(meal_item_entry)

        # 3. Aggregate meal total nutrition
        total_nut = self.nutrition_calculator.aggregate_meal_nutrition(meal_items)

        total_time_ms = round((time.time() - start_time) * 1000, 2)

        result = {
            "success": True,
            "meal_items": meal_items,
            "total_nutrition": total_nut
        }

        # Optional visualization
        if include_visualization:
            vis_img = self.detector.visualize_detections(image, detections, portion_records)
            result["annotated_image"] = vis_img

        result["meta"] = {
            "items_detected_count": len(meal_items),
            "inference_time_ms": total_time_ms,
            "disclaimer": "Portion and nutritional values are estimates based on 2D image analysis and ICMR-NIN reference data."
        }

        return result
