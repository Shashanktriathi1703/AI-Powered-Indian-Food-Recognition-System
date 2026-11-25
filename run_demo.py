"""
Nutriiya AI - Interactive Live Demonstration Script
Runs meal analysis on any input image and displays the results.
Usage:
    python run_demo.py sample_predictions/meal_01_thali.jpg
"""

import os
import sys
import json
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.pipeline import MealAnalyzerPipeline


def main():
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = os.path.join(PROJECT_ROOT, "sample_predictions", "meal_01_thali.jpg")

    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    print("=================================================================")
    print("NUTRIIYA | AI & NUTRITION INTELLIGENCE")
    print("Indian Food Recognition, Portion Estimation & Nutrition Analysis")
    print("=================================================================")
    print(f"Analyzing meal image: {img_path}")

    pipeline = MealAnalyzerPipeline()
    img = Image.open(img_path)
    result = pipeline.analyze_meal(img, include_visualization=True, conf_threshold=0.20)

    annotated_img = result.pop("annotated_image", None)
    if annotated_img:
        out_vis_path = os.path.splitext(img_path)[0] + "_demo_annotated.jpg"
        annotated_img.save(out_vis_path)
        print(f"Visualization saved to: {out_vis_path}")

    print("\n---------------------- MEAL BREAKDOWN ----------------------")
    print(f"{'Food Item':<18} | {'Conf':<6} | {'Portion (g)':<12} | {'Calories':<10} | {'Protein':<8} | {'Carbs':<8} | {'Fat':<8}")
    print("-" * 82)
    for item in result["meal_items"]:
        n = item["nutrition"]
        print(f"{item['food_name']:<18} | {int(item['confidence']*100)}%    | {item['estimated_quantity_g']:<12} | {n['calories']:<10} | {n['protein']}g{'':<5} | {n['carbs']}g{'':<5} | {n['fat']}g")
    
    print("-" * 82)
    tot = result["total_nutrition"]
    print(f"{'TOTAL MEAL':<18} | {'--':<6} | {'--':<12} | {tot['calories']} kcal{'':<4} | {tot['protein']}g{'':<5} | {tot['carbs']}g{'':<5} | {tot['fat']}g")
    print("=================================================================")
    print("\nNutriiya Standard JSON Output:")
    print(json.dumps({
        "success": result["success"],
        "meal_items": result["meal_items"],
        "total_nutrition": result["total_nutrition"]
    }, indent=2))


if __name__ == "__main__":
    main()
