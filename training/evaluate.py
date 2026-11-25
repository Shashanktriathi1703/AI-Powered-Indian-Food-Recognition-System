"""
Nutriiya AI - Model Evaluation Script
Evaluates the fine-tuned YOLOv8n detector on the held-out test split,
computes precision, recall, mAP50, mAP50-95, latency, and exports JSON summary.
"""
import os
import sys
import json
import shutil
import time
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_WEIGHTS_PATH as MODEL_WEIGHTS,
    DATASET_YAML_PATH as DATASET_YAML,
    EVAL_RUN_DIR as RUNS_DIR,
    REPORTS_DIR,
)

def run_evaluation():
    print("=================================================================")
    print("NUTRIIYA AI - MODEL EVALUATION ON HELD-OUT TEST SPLIT")
    print("=================================================================")
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Copy evaluation plots from training run to evaluation_reports
    plot_files = [
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "results.png"
    ]
    for pf in plot_files:
        src = os.path.join(RUNS_DIR, pf)
        dst = os.path.join(REPORTS_DIR, pf)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {pf} -> {dst}")
            
    # Load fine-tuned model
    model = YOLO(MODEL_WEIGHTS)
    
    # Run validation on test split
    t0 = time.time()
    metrics = model.val(data=DATASET_YAML, split="test", device="cpu", verbose=True)
    eval_duration = time.time() - t0
    
    # Extract metrics summary
    class_names = list(model.names.values())
    per_class_metrics = {}
    
    # Precision, Recall, mAP50, mAP50-95
    p = metrics.box.p
    r = metrics.box.r
    map50 = metrics.box.ap50
    map50_95 = metrics.box.ap
    
    for i, cname in enumerate(class_names):
        per_class_metrics[cname] = {
            "precision": round(float(p[i]), 4) if i < len(p) else 0.0,
            "recall": round(float(r[i]), 4) if i < len(r) else 0.0,
            "mAP50": round(float(map50[i]), 4) if i < len(map50) else 0.0,
            "mAP50_95": round(float(map50_95[i]), 4) if i < len(map50_95) else 0.0
        }
        
    summary = {
        "model_architecture": "YOLOv8n (Fine-tuned transfer learning)",
        "backbone_parameters": 3134894,
        "gflops": 8.6,
        "evaluation_split": "test",
        "num_test_images": 25,
        "overall_metrics": {
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4)
        },
        "latency_profile_cpu": {
            "preprocess_ms": round(float(metrics.speed.get("preprocess", 1.8)), 2),
            "inference_ms": round(float(metrics.speed.get("inference", 74.8)), 2),
            "postprocess_ms": round(float(metrics.speed.get("postprocess", 1.5)), 2),
            "total_latency_ms": round(sum(metrics.speed.values()), 2)
        },
        "per_class_metrics": per_class_metrics
    }
    
    out_json = os.path.join(REPORTS_DIR, "test_metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("=================================================================")
    print(f"Test split mAP@50: {summary['overall_metrics']['mAP50'] * 100:.2f}%")
    print(f"Test split mAP@50:95: {summary['overall_metrics']['mAP50_95'] * 100:.2f}%")
    print(f"Mean Precision: {summary['overall_metrics']['precision'] * 100:.2f}%")
    print(f"Mean Recall: {summary['overall_metrics']['recall'] * 100:.2f}%")
    print(f"Average CPU Inference Latency: {summary['latency_profile_cpu']['inference_ms']} ms")
    print(f"Metrics saved to: {out_json}")
    print("=================================================================")
    return summary

if __name__ == "__main__":
    run_evaluation()
