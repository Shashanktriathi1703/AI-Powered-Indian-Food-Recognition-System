"""
Nutriiya AI - Real-World Image Evaluation Engine
Tests the Indian Food Recognition pipeline on 10 genuine photographic images of Indian dishes,
saves annotated visualizations, and exports compliant Nutriiya JSON responses.
"""

import os
import sys
import json
import time
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.pipeline import MealAnalyzerPipeline

REAL_DIR = os.path.join(PROJECT_ROOT, "real_images")


def evaluate_real_world_images():
    print("=================================================================")
    print("NUTRIIYA AI - REAL-WORLD IMAGE EVALUATION BENCHMARK")
    print(f"Testing directory: {REAL_DIR}")
    print("=================================================================\n")

    pipeline = MealAnalyzerPipeline()
    
    # List real photos (exclude already annotated ones)
    real_images = sorted([
        f for f in os.listdir(REAL_DIR)
        if f.endswith((".jpg", ".png", ".jpeg")) and not f.endswith("_annotated.jpg")
    ])

    results_summary = []

    for img_name in real_images:
        img_path = os.path.join(REAL_DIR, img_name)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        t0 = time.time()
        # Analyze meal with visualization
        result = pipeline.analyze_meal(img, include_visualization=True, conf_threshold=0.25)
        inference_time_ms = round((time.time() - t0) * 1000, 2)

        # Save annotated image
        annotated_img = result.pop("annotated_image", None)
        out_img_name = f"{os.path.splitext(img_name)[0]}_annotated.jpg"
        out_img_path = os.path.join(REAL_DIR, out_img_name)
        if annotated_img:
            annotated_img.save(out_img_path, quality=92)

        # Save compliant JSON response
        out_json_name = f"{os.path.splitext(img_name)[0]}_response.json"
        out_json_path = os.path.join(REAL_DIR, out_json_name)
        compliant_resp = {
            "success": result["success"],
            "meal_items": result["meal_items"],
            "total_nutrition": result["total_nutrition"]
        }
        with open(out_json_path, "w") as f:
            json.dump(compliant_resp, f, indent=2)

        # Format items string
        items_desc = ", ".join([
            f"{it['food_name']} ({int(it['confidence']*100)}%, {it['estimated_quantity_g']}g)"
            for it in result["meal_items"]
        ]) if result["meal_items"] else "None (below threshold)"

        tot = result["total_nutrition"]
        print(f"[*] {img_name:<26} ({w}x{h}) in {inference_time_ms:>6.1f}ms")
        print(f"    Detected: {items_desc}")
        print(f"    Nutrition: {tot['calories']} kcal | P: {tot['protein']}g | C: {tot['carbs']}g | F: {tot['fat']}g\n")

        results_summary.append({
            "image_filename": img_name,
            "dimensions": f"{w}x{h}",
            "inference_time_ms": inference_time_ms,
            "items_count": len(result["meal_items"]),
            "items": result["meal_items"],
            "total_nutrition": tot,
            "annotated_image": out_img_name,
            "json_response": out_json_name
        })

    # Save summary report
    summary_path = os.path.join(REAL_DIR, "real_images_benchmark_report.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("=================================================================")
    print(f"Evaluated {len(real_images)} real images successfully.")
    print(f"Full benchmark report saved to: {summary_path}")
    print("=================================================================")
    return results_summary


if __name__ == "__main__":
    evaluate_real_world_images()
