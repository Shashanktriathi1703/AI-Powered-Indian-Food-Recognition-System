"""
Nutriiya AI - Incorporate Real Photographic Images into Training Dataset
Adds annotated real-world meal images to train and val splits to bridge the Sim-to-Real domain gap.
"""

import os
import sys
import shutil
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROJECT_ROOT,
    REAL_IMAGES_DIR as REAL_DIR,
    DATA_PROCESSED_DIR as DATA_DIR,
)

# True bounding box annotations for real images: (class_id, norm_cx, norm_cy, norm_w, norm_h)
REAL_ANNOTATIONS = {
    # 7 samosa, 1 dal, 0 rice, 2 roti, 3 paneer_curry, 5 chole, 6 biryani, 8 idli, 9 dosa
    "real_samosa_plate.jpg": {
        "split": "train",
        "boxes": [
            (7, 0.35, 0.52, 0.42, 0.55), # samosa 1
            (7, 0.70, 0.55, 0.40, 0.52)  # samosa 2
        ]
    },
    "real_paneer_curry.jpg": {
        "split": "train",
        "boxes": [
            (3, 0.50, 0.50, 0.78, 0.65)  # paneer curry bowl
        ]
    },
    "real_dal_bowl.jpg": {
        "split": "train",
        "boxes": [
            (1, 0.50, 0.50, 0.76, 0.62)  # yellow dal tadka bowl
        ]
    },
    "real_rice_bowl.jpg": {
        "split": "train",
        "boxes": [
            (0, 0.50, 0.50, 0.78, 0.65)  # steamed rice bowl
        ]
    },
    "real_chole_puri.jpg": {
        "split": "train",
        "boxes": [
            (5, 0.68, 0.50, 0.50, 0.62), # chole
            (2, 0.28, 0.50, 0.45, 0.62)  # roti / puri
        ]
    },
    "real_roti_curry.jpg": {
        "split": "train",
        "boxes": [
            (2, 0.32, 0.55, 0.45, 0.60), # rotis
            (1, 0.74, 0.45, 0.38, 0.48)  # dal
        ]
    },
    "real_south_indian.jpg": {
        "split": "val",
        "boxes": [
            (8, 0.45, 0.48, 0.55, 0.55), # idlis
            (1, 0.50, 0.50, 0.85, 0.75)  # sambar
        ]
    },
    "real_dosa_platter.jpg": {
        "split": "test",
        "boxes": [
            (9, 0.50, 0.45, 0.72, 0.52), # dosa
            (1, 0.82, 0.25, 0.25, 0.32), # sambar
            (8, 0.16, 0.35, 0.22, 0.32)  # idli
        ]
    },
    "real_biryani.jpg": {
        "split": "test",
        "boxes": [
            (6, 0.50, 0.52, 0.82, 0.72)  # biryani
        ]
    },
    "real_thali_curry_rice.jpg": {
        "split": "test",
        "boxes": [
            (0, 0.50, 0.52, 0.45, 0.38), # rice
            (1, 0.26, 0.30, 0.30, 0.28), # dal
            (3, 0.74, 0.30, 0.30, 0.28), # paneer curry
            (2, 0.50, 0.82, 0.48, 0.30)  # roti / naan
        ]
    }
}


def add_real_images():
    print("Adding annotated real photographic images to dataset splits...")
    
    for filename, data in REAL_ANNOTATIONS.items():
        src_img = os.path.join(REAL_DIR, filename)
        if not os.path.exists(src_img):
            print(f"Warning: {filename} not found in {REAL_DIR}")
            continue
            
        split = data["split"]
        dest_img = os.path.join(DATA_DIR, "images", split, f"photoreal_{filename}")
        dest_lbl = os.path.join(DATA_DIR, "labels", split, f"photoreal_{os.path.splitext(filename)[0]}.txt")
        
        # Copy image
        shutil.copy2(src_img, dest_img)
        
        # Write YOLO label
        with open(dest_lbl, "w") as f:
            for cls_id, cx, cy, w, h in data["boxes"]:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                
        print(f"Added {filename} -> {split} ({len(data['boxes'])} annotations)")
        
    print("Real-world photographic dataset incorporation complete!")


if __name__ == "__main__":
    add_real_images()
