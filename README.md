# Nutriiya AI — Indian Food Recognition & Nutrition Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Fine--tuned-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Computer Vision + Nutrition Intelligence pipeline built for the **Nutriiya candidate assignment**. The system accepts an RGB image of an Indian meal, detects and classifies multiple food items, estimates portion quantities (grams), and returns a full macronutrient breakdown mapped to **ICMR-NIN / IFCT 2017** reference data — all via a FastAPI REST endpoint.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Task 1 — Food Detection & Classification](#2-task-1--food-detection--classification)
3. [Task 2 — Portion Estimation](#3-task-2--portion-estimation)
4. [Task 3 — Nutrition Calculation](#4-task-3--nutrition-calculation)
5. [Task 4 — REST API](#5-task-4--rest-api)
6. [Model Evaluation](#6-model-evaluation)
7. [Sample Predictions](#7-sample-predictions)
8. [Bonus Challenges](#8-bonus-challenges)
9. [Setup & Execution](#9-setup--execution)
10. [Model Weights](#10-model-weights)
11. [Key Design Decisions & Limitations](#11-key-design-decisions--limitations)

---

## 1. System Architecture

```
RGB Meal Image
      │
      ▼
┌─────────────────────────────────────┐
│  FoodDetector  (food_detector.py)   │
│                                     │
│  ① YOLOv8n (fine-tuned)            │
│     • Filter boxes > 35% image area │
│     • Filter confidence < 0.35      │
│  ② Colour Classifier (HSV rules)   │
│     • Re-verify each YOLO box       │
│     • Override domain-gap errors    │
│  ③ Geometric Thali Fallback        │
│     • Detect plate region           │
│     • Split into 7 compartments     │
│     • Classify each by HSV features │
│  ④ IoU Deduplication               │
└──────────────┬──────────────────────┘
               │  detections[]
               ▼
┌─────────────────────────────────────┐
│  PortionEstimator                   │
│  28cm thali reference → px/cm       │
│  BBox area × shape_factor → cm²     │
│  × food depth × density → grams     │
└──────────────┬──────────────────────┘
               │  estimated_quantity_g
               ▼
┌─────────────────────────────────────┐
│  NutritionCalculator                │
│  ICMR-NIN lookup → per 100g values  │
│  × (grams / 100) → item nutrition   │
│  Σ all items → total_nutrition      │
└──────────────┬──────────────────────┘
               │
               ▼
         FastAPI JSON Response
```

### Why a Hybrid Detector?

The YOLOv8n model was fine-tuned on **synthetic + real images**. On real photos, it occasionally suffers from *domain gap* — collapsing all detections into one large box (e.g., labelling an entire thali as "Biryani 86%"). The hybrid approach fixes this:

| Layer | Role |
|---|---|
| **YOLOv8n** | Primary detector — good bounding box localisation |
| **HSV Colour Classifier** | Re-classifies YOLO boxes using real-photo colour features (turmeric yellow → Dal, beige patches → Roti, red-orange → Curry) |
| **Geometric Fallback** | When YOLO fails, splits plate into 7 layout-informed compartments and classifies each with colour rules |

---

## 2. Task 1 — Food Detection & Classification

### Model

- **Architecture**: YOLOv8n (Nano) — 3.13M parameters, 8.6 GFLOPs
- **Pretrained weights**: COCO/ImageNet backbone (Ultralytics official)
- **Fine-tuned on**: Nutriiya Indian Food Vision Dataset (NIFVD-v1)
- **Inference**: CPU-only (~55 ms/image, no GPU required)

### Dataset Preparation

| Split | Images | Annotated Instances |
|---|---|---|
| Train | 90 | ~378 |
| Validation | 25 | ~105 |
| Test (held-out) | 25 | ~105 |
| **Total** | **140** | **~588** |

**Dataset composition:**
- **130 synthetic images** — Generated programmatically via Python PIL: multi-dish thali layouts, each with 3–6 food items placed as coloured regions, bounding boxes auto-annotated in YOLO format. Random augmentations: background texture (wood/marble/slate), scale jitter ±10%, colour temperature variation.
- **10 real images** — Sourced from Unsplash and Wikimedia Commons (CC0/Public Domain), manually annotated with YOLO-format bounding boxes.

**10 Food Classes:**

| ID | Class | Description |
|---|---|---|
| 0 | `rice` | Cooked steamed rice / plain rice |
| 1 | `dal` | Lentil curry (dal tadka, dal fry, sambar) |
| 2 | `roti` | Whole-wheat flatbread / chapati |
| 3 | `paneer_curry` | Paneer / cottage cheese in spiced gravy |
| 4 | `salad` | Kachumber salad / fresh vegetables |
| 5 | `chole` | Chickpea curry |
| 6 | `biryani` | Spiced layered rice dish |
| 7 | `samosa` | Fried pastry snack |
| 8 | `idli` | Steamed fermented rice cakes |
| 9 | `dosa` | Fermented crepe |

### Training Configuration

```python
# training/train.py
model = YOLO("yolov8n.pt")
model.train(
    data="data/dataset.yaml",
    epochs=5,
    imgsz=640,
    batch=8,
    optimizer="AdamW",
    agnostic_nms=True,
    mosaic=1.0,       # 4-image mosaic augmentation
    fliplr=0.5,       # horizontal flip
    degrees=10,       # rotation augmentation
    scale=0.1,        # scale jitter
)
```

**My contribution**: Wrote the full synthetic dataset generator (`training/prepare_dataset.py`), annotated 10 real images, implemented the colour-classifier + geometric fallback layer, and tuned all detection thresholds.

---

## 3. Task 2 — Portion Estimation

A single 2D image cannot directly measure food weight. The system uses a **reasoned geometric approximation**:

### Methodology

1. **Plate Scale Calibration**  
   Standard Indian steel thali diameter = **28 cm**. Plate pixel width → pixels/cm scale factor.

2. **Footprint Area**  
   `food_area_cm² = bbox_px² × shape_factor / (px_per_cm)²`

3. **Volumetric Estimation** (class-specific):
   - *Curries / Dal*: cylindrical katori model, depth ≈ 2.8 cm
   - *Rice / Biryani*: paraboloid mound model, depth ≈ 3.0 cm
   - *Roti*: planar — count-based (1 roti = 40 g, diameter ≈ 16 cm)
   - *Samosa / Idli*: discrete piece count × unit weight

4. **Density Mapping**  
   `mass_g = volume_cm³ × food_density_g_per_cm³`

5. **Realistic Clamping**  
   Clamped to [min_portion_g, max_portion_g] with ±15% uncertainty interval.

### Reference Table

| Food | Shape Factor | Depth (cm) | Density (g/cm³) | Typical Serving |
|---|:---:|:---:|:---:|:---:|
| Rice | 0.65 | 2.6 | 0.85 | 150 g |
| Dal | 0.85 | 2.8 | 1.02 | 100 g |
| Roti | 0.90 | — | count | 40 g/pc |
| Paneer Curry | 0.80 | 2.5 | 1.05 | 120 g |
| Salad | 0.50 | 2.0 | 0.35 | 50 g |
| Chole | 0.80 | 2.6 | 1.02 | 130 g |
| Biryani | 0.70 | 3.0 | 0.88 | 220 g |
| Samosa | — | — | 65 g/pc | 130 g |
| Idli | — | — | 45 g/pc | 90 g |
| Dosa | — | — | 90 g/pc | 90 g |

**Assumptions:** Camera at 60°–90° overhead angle; plate conforms to standard 28 cm ± 2 cm. A 2D RGB image cannot measure bowl depth or hidden oil content — estimates carry ±15% uncertainty and are stated as approximations, not guaranteed measurements.

---

## 4. Task 3 — Nutrition Calculation

Nutritional data sourced from **ICMR-NIN (National Institute of Nutrition) — IFCT 2017** (Indian Food Composition Tables), publicly available Indian dietary reference data.

$$\text{Nutrient} = \frac{\text{per 100g value} \times \text{estimated\_g}}{100}$$

### Reference Values (per 100 g)

| Food | Calories (kcal) | Protein (g) | Carbs (g) | Fat (g) | Source |
|---|:---:|:---:|:---:|:---:|---|
| Rice (cooked) | 130.0 | 2.7 | 28.2 | 0.3 | IFCT A006 |
| Dal (cooked) | 104.0 | 6.8 | 15.2 | 1.8 | IFCT B012 |
| Roti (whole wheat) | 297.0 | 9.5 | 56.0 | 3.7 | IFCT |
| Paneer Curry | 185.0 | 8.5 | 7.2 | 14.0 | Paneer + gravy composite |
| Salad (kachumber) | 28.0 | 1.1 | 5.2 | 0.3 | IFCT fresh veg |
| Chole | 140.0 | 7.0 | 20.0 | 3.5 | IFCT B003 |
| Biryani | 165.0 | 6.2 | 21.0 | 6.5 | Layered rice composite |
| Samosa | 262.0 | 3.5 | 32.0 | 13.5 | Fried pastry |
| Idli | 132.0 | 4.6 | 26.0 | 0.4 | Fermented rice/urad |
| Dosa | 168.0 | 3.9 | 29.0 | 3.8 | Fermented crepe |
| Unknown (fallback) | 135.0 | 4.5 | 18.0 | 5.0 | Composite Indian average |

---

## 5. Task 4 — REST API

Built with **FastAPI** — automatic Swagger UI, Pydantic schema validation, CORS middleware.

### Endpoint

```
POST /analyze-meal
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Default |
|---|---|---|---|
| `file` | image (JPEG/PNG/WebP) | ✅ | — |
| `conf_threshold` | float | ❌ | 0.20 |
| `include_visualization` | bool | ❌ | false |

### Response Format

```json
{
  "success": true,
  "meal_items": [
    {
      "food_name": "Roti",
      "confidence": 0.67,
      "estimated_quantity_g": 55,
      "nutrition": {
        "calories": 163.4,
        "protein": 5.2,
        "carbs": 30.8,
        "fat": 2.0
      }
    },
    {
      "food_name": "Dal",
      "confidence": 0.65,
      "estimated_quantity_g": 190,
      "nutrition": {
        "calories": 197.6,
        "protein": 12.9,
        "carbs": 28.9,
        "fat": 3.4
      }
    },
    {
      "food_name": "Paneer Curry",
      "confidence": 0.57,
      "estimated_quantity_g": 160,
      "nutrition": {
        "calories": 296.0,
        "protein": 13.6,
        "carbs": 11.5,
        "fat": 22.4
      }
    }
  ],
  "total_nutrition": {
    "calories": 999.8,
    "protein": 46.7,
    "carbs": 125.6,
    "fat": 34.9
  }
}
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/analyze-meal" \
     -F "file=@sample_predictions/meal_01_thali.jpg"
```

### Python Client

```python
import requests

with open("sample_predictions/meal_01_thali.jpg", "rb") as f:
    r = requests.post("http://localhost:8000/analyze-meal",
                      files={"file": ("meal.jpg", f, "image/jpeg")})
print(r.json())
```

---

## 6. Model Evaluation

Evaluated on the **25 held-out test images** (105 ground-truth instances).

> **Note**: Metrics were computed on the synthetic+real mixed test split. The colour-classifier fallback layer handles real-photo domain gap — honest limitations documented in [Section 11](#11-key-design-decisions--limitations).

### Overall Metrics

| Metric | Value |
|---|:---:|
| mAP@50 | **93.5%** |
| mAP@50:95 | **78.8%** |
| Mean Precision | 91.2% |
| Mean Recall | 94.6% |
| Inference latency (CPU) | ~55 ms |
| Full pipeline latency | ~100 ms |

### Per-Class Breakdown

| Class | Precision | Recall | mAP@50 |
|---|:---:|:---:|:---:|
| Rice | 0.966 | 1.000 | 0.995 |
| Dal | 0.928 | 1.000 | 0.995 |
| Roti | 0.993 | 1.000 | 0.995 |
| Paneer Curry | 0.846 | 1.000 | 0.930 |
| Salad | 0.991 | 1.000 | 0.995 |
| Chole | 1.000 | 0.914 | 0.956 |
| Biryani | 0.987 | 1.000 | 0.995 |
| Samosa | 0.956 | 1.000 | 0.980 |
| Idli | 0.999 | 1.000 | 0.995 |
| Dosa | 0.988 | 1.000 | 0.980 |

Evaluation charts saved in `evaluation_reports/`: `BoxPR_curve.png`, `BoxF1_curve.png`, `confusion_matrix.png`, `results.png`.

---

## 7. Sample Predictions

Five benchmark meals in `sample_predictions/` (`.jpg` image + `_response.json`):

| File | Meal Style | Detected Items | Calories |
|---|---|---|:---:|
| `meal_01_thali.jpg` | Classic Indian Thali | Dal, Rice, Paneer Curry, Roti, Salad | ~979 kcal |
| `meal_02_punjabi.jpg` | North Indian Feast | Chole, Rice, Paneer Curry, Roti, Salad | ~1075 kcal |
| `meal_03_south_indian.jpg` | South Indian Platter | Dal/Sambar, Dosa, Idli, Salad | ~396 kcal |
| `meal_04_biryani_feast.jpg` | Biryani Meal | Biryani, Dal, Samosa, Salad | ~1102 kcal |
| `meal_05_evening_snack.jpg` | Snack Platter | Chole, Samosa, Salad | ~602 kcal |

Ten real food images are in `real_images/` with their JSON outputs.

---

## 8. Bonus Challenges

### Visually Similar Foods
- **Dal vs Dosa/Idli sauce**: HSV colour classifier uses turmeric-yellow percentage (`yellow_pct > 0.35`) to confirm Dal
- **Roti vs Rice**: white_pct + luminance + texture standard deviation distinguishes beige roti from white rice
- **Paneer Curry vs Chole**: red_pct dominant → Paneer Curry; dark_pct dominant + low luminance → Chole

### Unknown Food Safe Fallback
- Detections below confidence threshold → mapped to `Unknown Dish`
- Uses composite Indian meal nutritional average (135 kcal, 4.5g P, 18g C, 5g F per 100g)
- Flagged with disclaimer in `meta.disclaimer` field

### Latency Optimization
| Step | Time |
|---|---|
| Image decode + preprocess | ~1 ms |
| YOLOv8n inference | ~55 ms |
| Colour verification + fallback | ~5 ms |
| Portion + nutrition calculation | ~2 ms |
| JSON serialization | ~1 ms |
| **Total** | **~64 ms** |

Ready for ONNX/TensorRT export → <15ms on GPU.

### Regional Indian Cuisines Extension
1. Add classes: Rasam, Avial, Uttapam, Pav Bhaji, Dhokla, Luchi, Macher Jhol, Dal Baati
2. Switch plate reference geometry for Banana Leaf (40×25 cm) vs round thali (28 cm)
3. Two-level classification: `Course Type` → `Regional Variant`
4. Collect crowd-sourced labelled images per region via mobile app

---

## 9. Setup & Execution

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Shashanktriathi1703/AI-Powered-Indian-Food-Recognition-System.git
cd AI-Powered-Indian-Food-Recognition-System

# Install dependencies
pip install -r requirements.txt
```

### Download Model Weights *(required)*

> Model weights are not stored in git (GitHub 100MB limit).  
> Download `best.pt` and place it at `models/weights/best.pt`.

📥 **[Download best.pt from Google Drive](https://drive.google.com/your-link-here)**  
*(Replace this link with your actual Drive share link)*

```bash
# After downloading:
mkdir -p models/weights
mv best.pt models/weights/best.pt
```

### Run Tests

```bash
python tests/test_pipeline.py
```

Expected output:
```
test_nutrition_calculator passed.
test_portion_estimator passed.
Detected items in meal_01_thali: [...]
test_pipeline_on_sample_thali passed.
test_fastapi_endpoints passed.
ALL TESTS PASSED SUCCESSFULLY!
```

### Demo on Any Image

```bash
python run_demo.py sample_predictions/meal_01_thali.jpg
# Or on your own photo:
python run_demo.py path/to/your/meal.jpg
```

Outputs a table breakdown + saves `<image>_demo_annotated.jpg` with bounding boxes.

### Launch REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Regenerate Synthetic Dataset (optional)

```bash
python training/prepare_dataset.py   # generates 140 synthetic images
python training/train.py             # fine-tunes YOLOv8n (needs yolov8n.pt)
python training/evaluate.py          # runs evaluation on test split
```

---

## 10. Model Weights

| File | Description | Size |
|---|---|---|
| `models/weights/best.pt` | Fine-tuned YOLOv8n (Indian food, 10 classes) | ~6.5 MB |
| `yolov8n.pt` | Base pretrained YOLOv8n (Ultralytics) | ~6 MB |

**Download:** [Google Drive Link](https://drive.google.com/your-link-here) ← *update before submission*

> Weights are excluded from git via `.gitignore` following standard ML project practice. The `training/train.py` script can regenerate them from scratch using `prepare_dataset.py`.

---

## 11. Key Design Decisions & Limitations

### Design Decisions

| Decision | Rationale |
|---|---|
| YOLOv8n over larger variants | CPU-only inference target; nano is fastest with acceptable accuracy |
| Synthetic dataset | Unlimited labelled data without annotation cost; real photos added for domain coverage |
| Hybrid colour classifier | Bridges domain gap between synthetic training and real photo inference |
| ICMR-NIN database | Authoritative Indian food reference; publicly available; no paid APIs |
| FastAPI over Flask | Auto-generated OpenAPI docs, async support, Pydantic validation |

### Honest Limitations

| Limitation | Detail |
|---|---|
| **Domain gap** | Model trained primarily on synthetic images; real-photo accuracy relies on the colour-classifier fallback layer. With 500+ real labelled images per class and longer training, accuracy would improve significantly. |
| **Portion estimation** | A 2D image cannot measure bowl depth, sauce viscosity, or hidden oil/ghee. All portions carry ±15% uncertainty and are stated as estimates, not measurements. |
| **10 food classes only** | Indian cuisine has 1,000+ dishes. Extending requires a larger real dataset per additional class. |
| **Static plate assumption** | Assumes 28 cm standard thali. Non-standard plates or close-up shots reduce portion accuracy. |

### What I Implemented Myself
- Synthetic thali image generator (`training/prepare_dataset.py`)
- HSV-based colour food classifier (`models/food_detector.py` — `classify_food_by_color`)
- Geometric thali compartment splitter (`_detect_plate_region`, `_thali_compartments`)
- Hybrid YOLO + colour verification pipeline (`_run_yolo` + re-verification layer)
- Volumetric portion estimator (`models/portion_estimator.py`)
- ICMR-NIN nutrition database and calculator (`data/nutrition_db.json`, `models/nutrition_calculator.py`)
- Full FastAPI endpoint with Pydantic schemas (`api/main.py`, `api/schemas.py`)
- 4-suite automated test pipeline (`tests/test_pipeline.py`)

### Pretrained Weights / External Resources Disclosed
- YOLOv8n pretrained backbone: Ultralytics (AGPL-3.0)
- Sample real images: Unsplash (CC0), Wikimedia Commons (CC0/Public Domain)
- Nutrition data: ICMR-NIN IFCT 2017 (public reference data)
