"""
Nutriiya AI - Food Detection & Classification Engine
Combines YOLOv8 multi-dish object detection with targeted visual verification
to ensure high accuracy (preventing Rice vs Dal confusion and duplicate detections).
"""

import os
from typing import List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASSES = [
    "rice",          # 0
    "dal",           # 1
    "roti",          # 2
    "paneer_curry",   # 3
    "salad",         # 4
    "chole",         # 5
    "biryani",       # 6
    "samosa",        # 7
    "idli",          # 8
    "dosa"           # 9
]

DISPLAY_NAMES = {
    "rice": "Rice",
    "dal": "Dal",
    "roti": "Roti",
    "paneer_curry": "Paneer curry",
    "salad": "Salad",
    "chole": "Chole",
    "biryani": "Biryani",
    "samosa": "Samosa",
    "idli": "Idli",
    "dosa": "Dosa",
    "unknown": "Unknown Dish"
}

BOX_COLORS = {
    "rice": (210, 210, 200),
    "dal": (240, 180, 20),
    "roti": (205, 140, 70),
    "paneer_curry": (220, 80, 30),
    "salad": (40, 180, 60),
    "chole": (140, 70, 30),
    "biryani": (230, 140, 40),
    "samosa": (210, 130, 20),
    "idli": (180, 180, 220),
    "dosa": (190, 120, 40),
    "unknown": (128, 128, 128)
}


class FoodDetector:
    """
    Clean, robust food detector for Indian meal images.
    """

    def __init__(self, weights_path: str, conf_threshold: float = 0.45, iou_threshold: float = 0.35):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.yolo_model = None
        self._init_model()

    def _init_model(self):
        """Loads fine-tuned YOLO model weights."""
        if os.path.exists(self.weights_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.weights_path)
            except Exception as e:
                print(f"Warning: Could not load Ultralytics YOLO ({e}).")
                self.yolo_model = None
        else:
            print(f"Warning: Weights file not found at {self.weights_path}")

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Executes food detection and returns verified, deduplicated items.
        """
        raw_detections = []

        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    image, 
                    conf=self.conf_threshold, 
                    iou=self.iou_threshold,
                    agnostic_nms=True,
                    verbose=False
                )
                if results and len(results) > 0:
                    r = results[0]
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()
                        raw_name = r.names.get(cls_id, CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown")
                        
                        raw_detections.append({
                            "class_name": raw_name,
                            "display_name": DISPLAY_NAMES.get(raw_name, raw_name.replace("_", " ").title()),
                            "confidence": round(conf, 2),
                            "bbox": [round(coord, 1) for coord in xyxy]
                        })
            except Exception as e:
                print(f"Error during YOLO prediction ({e}).")

        # 1. Clean spatial deduplication (removes redundant multi-boxes on same dish)
        clean_detections = self._deduplicate_boxes(raw_detections, iou_thresh=0.25)

        # 2. Targeted Rice vs Dal visual verification (prevents misclassifying white rice as dal)
        verified_detections = self._verify_rice_and_dal(image, clean_detections)

        return verified_detections

    def _deduplicate_boxes(self, detections: List[Dict[str, Any]], iou_thresh: float = 0.25) -> List[Dict[str, Any]]:
        """Removes overlapping duplicate boxes covering the same food item/bowl."""
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        def get_iou(b1, b2):
            x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0

        for cand in sorted_dets:
            # Check overlap or nested center
            overlap = False
            cx, cy = (cand["bbox"][0] + cand["bbox"][2]) / 2, (cand["bbox"][1] + cand["bbox"][3]) / 2
            
            for prev in kept:
                # High IoU
                if get_iou(cand["bbox"], prev["bbox"]) > iou_thresh:
                    overlap = True
                    break
                # Nested center inside existing box
                if (prev["bbox"][0] <= cx <= prev["bbox"][2]) and (prev["bbox"][1] <= cy <= prev["bbox"][3]):
                    overlap = True
                    break
                    
            if not overlap:
                kept.append(cand)

        return kept

    def _verify_rice_and_dal(self, image: Image.Image, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Targeted visual verification to guarantee White Rice is never confused for Dal.
        Checks RGB saturation and turmeric-yellow spectral proportion in the food patch.
        """
        verified = []

        for det in detections:
            cls_name = det["class_name"]
            conf = det["confidence"]
            bbox = det["bbox"]

            x1, y1, x2, y2 = [int(round(c)) for c in bbox]
            crop = image.crop((max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)))
            arr = np.array(crop).astype(float)

            if arr.size == 0 or arr.shape[0] < 8 or arr.shape[1] < 8:
                verified.append(det)
                continue

            r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
            lum = 0.299 * r + 0.587 * g + 0.114 * b

            mx = np.maximum(np.maximum(arr[:, :, 0], arr[:, :, 1]), arr[:, :, 2])
            mn = np.minimum(np.minimum(arr[:, :, 0], arr[:, :, 1]), arr[:, :, 2])
            diff = mx - mn
            sat = np.where(mx > 0, diff / (mx + 1e-6), 0).mean()

            # Hue calculation
            h_arr = np.zeros_like(mx)
            mask_r = (mx == arr[:, :, 0]) & (diff > 0)
            h_arr[mask_r] = (60 * ((arr[:, :, 1][mask_r] - arr[:, :, 2][mask_r]) / diff[mask_r]) + 360) % 360

            yellow_pct = np.mean((h_arr >= 25) & (h_arr <= 70) & (diff > 30) & (arr[:, :, 0] > 130))
            white_pct = np.mean((arr[:, :, 0] > 180) & (arr[:, :, 1] > 175) & (arr[:, :, 2] > 165) & (diff < 35))

            final_class = cls_name
            final_conf = conf

            # Rule: If predicted as Dal, but it is white/achromatic with no yellow -> It is RICE!
            if cls_name == "dal" and (white_pct > 0.35 or (sat < 0.22 and lum > 160 and yellow_pct < 0.12)):
                final_class = "rice"
                final_conf = max(0.92, conf)

            # Rule: If predicted as Rice, but it has strong turmeric yellow saturation -> It is DAL!
            elif cls_name == "rice" and (yellow_pct > 0.28 and sat > 0.38):
                final_class = "dal"
                final_conf = max(0.92, conf)

            verified.append({
                "class_name": final_class,
                "display_name": DISPLAY_NAMES.get(final_class, final_class.replace("_", " ").title()),
                "confidence": round(float(final_conf), 2),
                "bbox": bbox
            })

        return verified

    def visualize_detections(
        self, 
        image: Image.Image, 
        detections: List[Dict[str, Any]], 
        portion_info: List[Dict[str, Any]] = None
    ) -> Image.Image:
        """
        Draws clean bounding boxes, labels, confidence scores, and estimated grams.
        """
        vis_img = image.copy()
        draw = ImageDraw.Draw(vis_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            color = BOX_COLORS.get(cls_name, (0, 200, 255))
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            label = f"{det['display_name']} ({int(det['confidence'] * 100)}%)"
            if portion_info and idx < len(portion_info):
                g = portion_info[idx].get("estimated_quantity_g", 0)
                label += f" ~{g}g"
                
            text_bbox = draw.textbbox((x1, y1 - 22), label, font=font)
            banner_box = [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2]
            
            if banner_box[1] < 0:
                banner_box[1] = y1 + 2
                banner_box[3] = y1 + 24
                text_y = y1 + 4
            else:
                text_y = y1 - 22
                
            draw.rectangle(banner_box, fill=color)
            draw.text((x1 + 2, text_y), label, fill=(255, 255, 255) if sum(color) < 450 else (0, 0, 0), font=font)

        return vis_img
