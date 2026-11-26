# Nutriiya AI: Indian Food Recognition and Nutrition Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Fine--tuned-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision and nutrition pipeline built for the **Nutriiya candidate assignment**. It accepts an RGB image of an Indian meal, detects multiple food items, estimates each portion in grams, and returns calories and macronutrients through a FastAPI REST endpoint.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Food Detection and Classification](#food-detection-and-classification)
- [Portion Estimation](#portion-estimation)
- [Nutrition Calculation](#nutrition-calculation)
- [REST API](#rest-api)
- [Model Evaluation](#model-evaluation)
- [Sample Predictions](#sample-predictions)
- [Bonus Challenges](#bonus-challenges)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Author Contributions](#author-contributions)
- [Credits and License](#credits-and-license)

---

## Features

- **Multi-item detection** across 10 Indian food classes
- **Hybrid recognition** combining fine-tuned YOLOv8n, HSV colour rules and a geometric thali fallback
- **Portion estimation** from plate-referenced geometry and food density
- **Nutrition breakdown** per item and for the whole meal
- **REST API** with Swagger documentation and Pydantic validation
- **CPU-only inference**, with no GPU or paid API required

---

## Quick Start

### Prerequisites

- Python 3.10 or later
- pip

### 1. Clone and install

```bash
git clone https://github.com/Shashanktriathi1703/AI-Powered-Indian-Food-Recognition-System.git
cd AI-Powered-Indian-Food-Recognition-System

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download the model weights (required)

The fine-tuned weights are excluded from Git through `.gitignore`.

**[Download `best.pt` from Google Drive](https://drive.google.com/file/d/1hsCzp6v3fpoxZAE3DcRpuO7de6iALl71/view?usp=sharing)** (about 6.5 MB) and place it at:

```text
models/weights/best.pt
```

### 3. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

| Resource | URL |
|---|---|
| Swagger UI | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

### 4. Run the demo

```bash
python run_demo.py sample_predictions/meal_01_thali.jpg
```

The script prints a nutrition table and saves an annotated `<image>_demo_annotated.jpg`.

Example output for `meal_01_thali.jpg`:

| Food item | Confidence | Portion (g) | Calories (kcal) | Protein (g) | Carbs (g) | Fat (g) |
|---|---:|---:|---:|---:|---:|---:|
| Dal | 100% | 205 | 213.2 | 13.9 | 31.2 | 3.7 |
| Salad | 100% | 25 | 7 | 0.3 | 1.3 | 0.1 |
| Paneer curry | 98% | 115 | 212.7 | 9.8 | 8.3 | 16.1 |
| Roti | 92% | 40 | 118.8 | 3.8 | 22.4 | 1.5 |
| Rice | 92% | 300 | 390 | 8.1 | 84.6 | 0.9 |
| Paneer curry | 44% | 90 | 166.5 | 7.7 | 6.5 | 12.6 |
| Rice | 25% | 135 | 175.5 | 3.6 | 38.1 | 0.4 |
| **Total** | | | **1283.7** | **47.2** | **192.4** | **35.3** |

<details>
<summary>View JSON output</summary>

```json
{
  "success": true,
  "meal_items": [
    {
      "food_name": "Dal",
      "confidence": 1.0,
      "estimated_quantity_g": 205,
      "nutrition": {
        "calories": 213.2,
        "protein": 13.9,
        "carbs": 31.2,
        "fat": 3.7
      }
    },
    {
      "food_name": "Salad",
      "confidence": 1.0,
      "estimated_quantity_g": 25,
      "nutrition": {
        "calories": 7,
        "protein": 0.3,
        "carbs": 1.3,
        "fat": 0.1
      }
    },
    {
      "food_name": "Paneer curry",
      "confidence": 0.98,
      "estimated_quantity_g": 115,
      "nutrition": {
        "calories": 212.7,
        "protein": 9.8,
        "carbs": 8.3,
        "fat": 16.1
      }
    },
    {
      "food_name": "Roti",
      "confidence": 0.92,
      "estimated_quantity_g": 40,
      "nutrition": {
        "calories": 118.8,
        "protein": 3.8,
        "carbs": 22.4,
        "fat": 1.5
      }
    },
    {
      "food_name": "Rice",
      "confidence": 0.92,
      "estimated_quantity_g": 300,
      "nutrition": {
        "calories": 390,
        "protein": 8.1,
        "carbs": 84.6,
        "fat": 0.9
      }
    },
    {
      "food_name": "Paneer curry",
      "confidence": 0.44,
      "estimated_quantity_g": 90,
      "nutrition": {
        "calories": 166.5,
        "protein": 7.7,
        "carbs": 6.5,
        "fat": 12.6
      }
    },
    {
      "food_name": "Rice",
      "confidence": 0.25,
      "estimated_quantity_g": 135,
      "nutrition": {
        "calories": 175.5,
        "protein": 3.6,
        "carbs": 38.1,
        "fat": 0.4
      }
    }
  ],
  "total_nutrition": {
    "calories": 1283.7,
    "protein": 47.2,
    "carbs": 192.4,
    "fat": 35.3
  }
}
```

</details>

### 5. Run the tests

```bash
python tests/test_pipeline.py
```

Expected output:

```text
test_nutrition_calculator passed.
test_portion_estimator passed.
Detected items in meal_01_thali: [...]
test_pipeline_on_sample_thali passed.
test_fastapi_endpoints passed.
ALL TESTS PASSED SUCCESSFULLY!
```

---

## System Architecture

```text
RGB Meal Image
      │
      ▼
┌──────────────────────────────────────┐
│  FoodDetector (food_detector.py)     │
│  1. Fine-tuned YOLOv8n               │
│     - drop boxes > 35% of image area │
│     - drop confidence < 0.35         │
│  2. HSV colour verification          │
│  3. Geometric thali fallback         │
│  4. IoU de-duplication               │
└──────────────────┬───────────────────┘
                   │ detections
                   ▼
┌──────────────────────────────────────┐
│  PortionEstimator                    │
│  Plate scale → area → volume → grams │
└──────────────────┬───────────────────┘
                   │ estimated grams
                   ▼
┌──────────────────────────────────────┐
│  NutritionCalculator                 │
│  Per-100 g lookup → item and total   │
└──────────────────┬───────────────────┘
                   │
                   ▼
          FastAPI JSON Response
```

### Why a hybrid detector?

<img width="757" height="762" alt="image" src="https://github.com/user-attachments/assets/c41db26d-9ea4-47cc-a16f-cf38ac7ad6b2" />

On real photos, the YOLOv8n model can suffer from a **domain gap**, for example by labelling an entire thali as a single "Biryani" box. Additional layers reduce this failure mode.

| Layer | Role |
|---|---|
| **YOLOv8n** | Primary detector and bounding-box localisation |
| **HSV colour classifier** | Re-checks each box using colour features and overrides likely errors |
| **Geometric fallback** | If YOLO fails, splits the plate into 7 compartments and classifies each by colour |

---

## Food Detection and Classification

<img width="993" height="747" alt="image" src="https://github.com/user-attachments/assets/3e8c0d71-5b06-4bfa-b260-4eed838045ca" />

### Model

| Property | Value |
|---|---|
| Architecture | YOLOv8n (3.13M parameters, 8.6 GFLOPs) |
| Pretrained weights | Ultralytics YOLOv8n (COCO) |
| Fine-tuning data | Nutriiya Indian Food Vision Dataset (NIFVD-v1) |
| Inference | CPU, about 55 ms per image |

### Dataset

| Split | Images | Annotated instances |
|---|---:|---:|
| Train | 90 | ~378 |
| Validation | 25 | ~105 |
| Test (held out) | 25 | ~105 |
| **Total** | **140** | **~588** |

- **130 synthetic images:** multi-dish thali layouts generated with Python PIL, each containing 3 to 6 food regions with automatic YOLO annotations. Variations include background texture (wood, marble, slate), ±10% scale jitter and colour temperature.
- **10 real images:** sourced from Unsplash and Wikimedia Commons, manually annotated in YOLO format.

### Classes

| ID | Class | Description |
|:---:|---|---|
| 0 | `rice` | Plain steamed rice |
| 1 | `dal` | Lentil curry (dal tadka, dal fry, sambar) |
| 2 | `roti` | Whole-wheat flatbread or chapati |
| 3 | `paneer_curry` | Paneer in spiced gravy |
| 4 | `salad` | Kachumber or fresh vegetables |
| 5 | `chole` | Chickpea curry |
| 6 | `biryani` | Spiced layered rice |
| 7 | `samosa` | Fried stuffed pastry |
| 8 | `idli` | Steamed fermented rice cake |
| 9 | `dosa` | Fermented crepe |

### Training configuration

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
    mosaic=1.0,    # 4-image mosaic
    fliplr=0.5,    # horizontal flip
    degrees=10,    # rotation
    scale=0.1,     # scale jitter
)
```

### Retraining (optional)

```bash
python training/prepare_dataset.py   # generate the dataset
python training/train.py             # fine-tune YOLOv8n
python training/evaluate.py          # evaluate on the test split
```

---

## Portion Estimation

<img width="1267" height="961" alt="image" src="https://github.com/user-attachments/assets/01d8905b-f44b-45bc-88b7-08a742165afe" />

A single 2D image cannot measure food weight directly, so the system uses a geometric approximation.

1. **Plate calibration:** a standard thali diameter of 28 cm gives the pixels-per-centimetre scale.
2. **Footprint area:**

```math
\text{Area (cm}^2\text{)} = \frac{\text{box area (px}^2\text{)} \times \text{shape factor}}{(\text{pixels per cm})^2}
```

3. **Volume:** curries and dal use a cylindrical bowl (katori) model; rice and biryani use a mound model. Depth is class-specific.
4. **Mass:**

```math
\text{Mass (g)} = \text{volume (cm}^3\text{)} \times \text{density (g/cm}^3\text{)}
```

5. **Clamping:** results are limited to a realistic range per class and reported with ±15% uncertainty.

Roti, samosa, idli and dosa are estimated from piece counts and unit weights.

### Volume-based foods

| Food | Shape factor | Depth (cm) | Density (g/cm³) | Typical serving |
|---|:---:|:---:|:---:|:---:|
| Rice | 0.65 | 2.6 | 0.85 | 150 g |
| Dal | 0.85 | 2.8 | 1.02 | 100 g |
| Paneer Curry | 0.80 | 2.5 | 1.05 | 120 g |
| Salad | 0.50 | 2.0 | 0.35 | 50 g |
| Chole | 0.80 | 2.6 | 1.02 | 130 g |
| Biryani | 0.70 | 3.0 | 0.88 | 220 g |

### Count-based foods

| Food | Unit weight | Typical serving |
|---|:---:|:---:|
| Roti (about 16 cm diameter) | 40 g | 40 g per piece |
| Samosa | 65 g | 130 g |
| Idli | 45 g | 90 g |
| Dosa | 90 g | 90 g |

**Assumptions:** the camera angle is 60° to 90° (90° is directly overhead), and the plate is 28 ± 2 cm wide.

---

## Nutrition Calculation

Each nutrient is scaled from its per-100 g value:

```math
\text{Nutrient} = \frac{\text{value per 100 g} \times \text{estimated weight (g)}}{100}
```

### Reference values (per 100 g)

| Food | Energy (kcal) | Protein (g) | Carbs (g) | Fat (g) | Basis |
|---|---:|---:|---:|---:|---|
| Rice (cooked) | 130.0 | 2.7 | 28.2 | 0.3 | Cooked staple |
| Dal (cooked) | 104.0 | 6.8 | 15.2 | 1.8 | Cooked dish |
| Roti (whole wheat) | 297.0 | 9.5 | 56.0 | 3.7 | Prepared flatbread |
| Paneer Curry | 185.0 | 8.5 | 7.2 | 14.0 | Paneer and gravy composite |
| Salad (kachumber) | 28.0 | 1.1 | 5.2 | 0.3 | Fresh vegetables |
| Chole | 140.0 | 7.0 | 20.0 | 3.5 | Cooked dish |
| Biryani | 165.0 | 6.2 | 21.0 | 6.5 | Layered rice composite |
| Samosa | 262.0 | 3.5 | 32.0 | 13.5 | Fried pastry |
| Idli | 132.0 | 4.6 | 26.0 | 0.4 | Fermented rice and urad |
| Dosa | 168.0 | 3.9 | 29.0 | 3.8 | Fermented crepe |
| Unknown (fallback) | 135.0 | 4.5 | 18.0 | 5.0 | Composite average |

> **Data note:** these are approximate values for prepared foods. [ICMR-NIN IFCT 2017](https://nin.res.in/ebooks/IFCT2017.pdf) is used as a reference, but it mainly reports raw foods, so the values above are not direct IFCT entries for cooked dishes.

---

## REST API

### `POST /analyze-meal`

Content type: `multipart/form-data`

| Parameter | Type | Required | Default |
|---|---|:---:|:---:|
| `file` | Image (JPEG, PNG, WebP) | Yes | none |
| `conf_threshold` | float | No | `0.20` |
| `include_visualization` | bool | No | `false` |

### cURL

```bash
curl -X POST "http://localhost:8000/analyze-meal" \
     -F "file=@sample_predictions/meal_01_thali.jpg"
```

### Python

```python
import requests

with open("sample_predictions/meal_01_thali.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze-meal",
        files={"file": ("meal.jpg", f, "image/jpeg")},
    )

print(response.json())
```

### Example response

```json
{
  "success": true,
  "meal_items": [
    {
      "food_name": "Roti",
      "confidence": 0.67,
      "estimated_quantity_g": 55,
      "nutrition": { "calories": 163.4, "protein": 5.2, "carbs": 30.8, "fat": 2.0 }
    },
    {
      "food_name": "Dal",
      "confidence": 0.65,
      "estimated_quantity_g": 190,
      "nutrition": { "calories": 197.6, "protein": 12.9, "carbs": 28.9, "fat": 3.4 }
    },
    {
      "food_name": "Paneer Curry",
      "confidence": 0.57,
      "estimated_quantity_g": 160,
      "nutrition": { "calories": 296.0, "protein": 13.6, "carbs": 11.5, "fat": 22.4 }
    }
  ],
  "total_nutrition": { "calories": 657.0, "protein": 31.7, "carbs": 71.2, "fat": 27.8 }
}
```

---

## Model Evaluation

Evaluated on 25 held-out test images (105 ground-truth instances) from the mixed synthetic and real dataset. These scores should not be read as real-world accuracy (see [Limitations](#limitations)).

### Overall metrics

| Metric | Value |
|---|:---:|
| mAP@50 | **93.5%** |
| mAP@50:95 | **78.8%** |
| Mean precision | 91.2% |
| Mean recall | 94.6% |

### Per-class metrics

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

Charts are saved in `evaluation_reports/`: `BoxPR_curve.png`, `BoxF1_curve.png`, `confusion_matrix.png` and `results.png`.

### Latency (CPU)

| Step | Time |
|---|---:|
| Image decoding and preprocessing | ~1 ms |
| YOLOv8n inference | ~55 ms |
| Colour verification and fallback | ~5 ms |
| Portion and nutrition calculation | ~2 ms |
| JSON serialisation | ~1 ms |
| **Total** | **~64 ms** |

---

## Sample Predictions

Five benchmark meals are provided in `sample_predictions/`, each as a `.jpg` image with a matching `_response.json`.

| File | Meal style | Detected items | Energy |
|---|---|---|---:|
| `meal_01_thali.jpg` | Classic thali | Dal, Rice, Paneer Curry, Roti, Salad | ~979 kcal |
| `meal_02_punjabi.jpg` | North Indian feast | Chole, Rice, Paneer Curry, Roti, Salad | ~1075 kcal |
| `meal_03_south_indian.jpg` | South Indian platter | Dal/Sambar, Dosa, Idli, Salad | ~396 kcal |
| `meal_04_biryani_feast.jpg` | Biryani meal | Biryani, Dal, Samosa, Salad | ~1102 kcal |
| `meal_05_evening_snack.jpg` | Snack platter | Chole, Samosa, Salad | ~602 kcal |

Ten real food images and their JSON outputs are in `real_images/`.

---

## Bonus Challenges

### Visually similar foods

| Confusion | Distinguishing signal |
|---|---|
| Dal vs dosa/idli accompaniments | Turmeric-yellow ratio (`yellow_pct > 0.35`) confirms dal |
| Roti vs rice | White ratio, luminance and texture variation |
| Paneer curry vs chole | Dominant red suggests paneer curry; dark ratio with low luminance suggests chole |

### Unknown-food fallback

Low-confidence detections are labelled `Unknown Dish`, use the composite average values above, and are flagged in `meta.disclaimer`.

### Latency optimisation

ONNX or TensorRT export is a planned next step to reduce inference time on GPU hardware.

### Regional cuisine extension

1. Add classes such as rasam, avial, uttapam, pav bhaji, dhokla, luchi, macher jhol and dal baati.
2. Support other plate geometries, such as a banana leaf (40 × 25 cm).
3. Classify in two levels: course type, then regional variant.
4. Collect labelled images for each region through a mobile app.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| YOLOv8n over larger variants | Fast enough for CPU-only inference |
| Synthetic training data | Labelled data without manual annotation cost |
| Hybrid colour classifier | Reduces the synthetic-to-real domain gap |
| Local nutrition database | Works offline with no paid API |
| FastAPI | Automatic OpenAPI docs and Pydantic validation |

---

## Limitations

| Limitation | Detail |
|---|---|
| **Domain gap** | Training data is mostly synthetic, and real photos depend heavily on the colour fallback. A larger labelled real-image set (500+ per class) and longer training are expected to help. |
| **Portion accuracy** | Depth, viscosity and hidden oil or ghee cannot be seen in a 2D image. The ±15% range is an assumption, not a measured error. |
| **Class coverage** | Only 10 of the many Indian dishes are supported. |
| **Plate assumption** | Non-standard plates and close-up shots reduce portion accuracy. |

---

## Author Contributions

- Synthetic thali generator (`training/prepare_dataset.py`)
- Manual annotation of 10 real images
- HSV colour classifier (`classify_food_by_color` in `models/food_detector.py`)
- Plate detection and compartment splitting (`_detect_plate_region`, `_thali_compartments`)
- Hybrid YOLO and colour verification pipeline, including threshold tuning
- Portion estimator (`models/portion_estimator.py`)
- Nutrition database and calculator (`data/nutrition_db.json`, `models/nutrition_calculator.py`)
- FastAPI service and schemas (`api/main.py`, `api/schemas.py`)
- Automated test pipeline with 4 tests (`tests/test_pipeline.py`)

---

## Credits and License

| Resource | Source |
|---|---|
| Pretrained YOLOv8n | [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) |
| Real sample images | Unsplash (Unsplash License) and Wikimedia Commons (CC0/Public Domain) |
| Nutrition reference | ICMR-NIN IFCT 2017 |

This project is released under the MIT License (see [LICENSE](LICENSE)). Third-party components remain under their own licenses.
