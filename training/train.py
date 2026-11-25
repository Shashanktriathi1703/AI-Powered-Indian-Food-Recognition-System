"""
Nutriiya AI - Model Training & Fine-Tuning Pipeline
Fine-tunes YOLOv8n on the Indian Food Vision Dataset.
"""
import os
import shutil
from ultralytics import YOLO

DATASET_YAML = r"C:\Users\Shashank Tripathi\.gemini\antigravity\scratch\nutriiya-indian-food-ai\data\dataset.yaml"
OUTPUT_WEIGHTS_DIR = r"C:\Users\Shashank Tripathi\.gemini\antigravity\scratch\nutriiya-indian-food-ai\models\weights"
RUNS_DIR = r"C:\Users\Shashank Tripathi\.gemini\antigravity\scratch\nutriiya-indian-food-ai\runs"

def run_training(epochs=6, batch_size=8, img_size=640):
    print("=================================================================")
    print("NUTRIIYA AI - HYBRID REAL+SYNTHETIC FINE-TUNING")
    print(f"Dataset config: {DATASET_YAML}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Image size: {img_size}")
    print("=================================================================")

    # Clear old cache files
    labels_dir = r"C:\Users\Shashank Tripathi\.gemini\antigravity\scratch\nutriiya-indian-food-ai\data\processed\labels"
    for root, _, files in os.walk(labels_dir):
        for f in files:
            if f.endswith(".cache"):
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass

    # Load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    # Train the model with real-world augmentations
    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device="cpu",
        project=RUNS_DIR,
        name="nutriiya_hybrid_yolov8n",
        save=True,
        plots=True,
        mosaic=1.0,
        fliplr=0.5,
        degrees=10,
        scale=0.5,
        verbose=True
    )
    
    # Locate best.pt
    best_weights_src = os.path.join(RUNS_DIR, "nutriiya_hybrid_yolov8n", "weights", "best.pt")
    if not os.path.exists(best_weights_src):
        best_weights_src = os.path.join(RUNS_DIR, "nutriiya_hybrid_yolov8n", "weights", "last.pt")
        
    os.makedirs(OUTPUT_WEIGHTS_DIR, exist_ok=True)
    target_weights = os.path.join(OUTPUT_WEIGHTS_DIR, "best.pt")
    if os.path.exists(best_weights_src):
        shutil.copy2(best_weights_src, target_weights)
        print(f"Successfully saved fine-tuned model weights to: {target_weights}")
    else:
        print("Warning: could not find trained weights at standard location.")
        
    return results

if __name__ == "__main__":
    run_training(epochs=5, batch_size=8, img_size=640)
