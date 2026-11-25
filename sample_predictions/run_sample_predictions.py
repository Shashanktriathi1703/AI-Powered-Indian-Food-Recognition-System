"""
Nutriiya AI - Benchmark Sample Predictions Generator
Generates visualized predictions and compliant JSON responses for at least 5 sample meal images.
"""

import os
import sys
import json
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.pipeline import MealAnalyzerPipeline

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "sample_predictions")


def generate_benchmark_predictions():
    print("=================================================================")
    print("NUTRIIYA AI - GENERATING 5 SAMPLE BENCHMARK PREDICTIONS")
    print("=================================================================")
    
    pipeline = MealAnalyzerPipeline()
    
    sample_files = [
        ("meal_01_thali.jpg", "meal_01_thali_annotated.jpg", "meal_01_thali_response.json", "Classic Thali"),
        ("meal_02_punjabi.jpg", "meal_02_punjabi_annotated.jpg", "meal_02_punjabi_response.json", "Punjabi Feast"),
        ("meal_03_south_indian.jpg", "meal_03_south_indian_annotated.jpg", "meal_03_south_indian_response.json", "South Indian Platter"),
        ("meal_04_biryani_feast.jpg", "meal_04_biryani_feast_annotated.jpg", "meal_04_biryani_feast_response.json", "Biryani Feast"),
        ("meal_05_evening_snack.jpg", "meal_05_evening_snack_annotated.jpg", "meal_05_evening_snack_response.json", "Snack Platter")
    ]
    
    all_summaries = []

    for src_file, out_img_file, out_json_file, label in sample_files:
        src_path = os.path.join(SAMPLES_DIR, src_file)
        if not os.path.exists(src_path):
            print(f"Skipping {src_file}: file does not exist.")
            continue
            
        img = Image.open(src_path)
        result = pipeline.analyze_meal(img, include_visualization=True, conf_threshold=0.45)
        
        # Save annotated image
        annotated_img = result.pop("annotated_image", None)
        if annotated_img:
            annotated_path = os.path.join(SAMPLES_DIR, out_img_file)
            annotated_img.save(annotated_path, quality=95)
            print(f"Saved annotated image: {annotated_path}")
            
        # Clean response strictly matching Nutriiya required JSON format
        compliant_response = {
            "success": result["success"],
            "meal_items": result["meal_items"],
            "total_nutrition": result["total_nutrition"]
        }
        
        # Save JSON output
        json_path = os.path.join(SAMPLES_DIR, out_json_file)
        with open(json_path, "w") as f:
            json.dump(compliant_response, f, indent=2)
        print(f"Saved JSON response: {json_path}")
        
        # Print summary
        items_desc = ", ".join([f"{item['food_name']} ({item['estimated_quantity_g']}g)" for item in result["meal_items"]])
        print(f"[{label}] -> Detected: {items_desc}")
        print(f"    Total Calories: {result['total_nutrition']['calories']} kcal | "
              f"Protein: {result['total_nutrition']['protein']}g | "
              f"Carbs: {result['total_nutrition']['carbs']}g | "
              f"Fat: {result['total_nutrition']['fat']}g\n")
              
        all_summaries.append({
            "meal_id": src_file,
            "title": label,
            "items_count": len(result["meal_items"]),
            "items": [item["food_name"] for item in result["meal_items"]],
            "total_nutrition": result["total_nutrition"],
            "annotated_image": out_img_file,
            "json_response": out_json_file
        })

    # Save summary report
    summary_report_path = os.path.join(SAMPLES_DIR, "samples_index.json")
    with open(summary_report_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Sample summary index saved to: {summary_report_path}")
    print("All 5 benchmark sample predictions successfully generated!")


if __name__ == "__main__":
    generate_benchmark_predictions()
