# Nutriiya AI | AI Powered Indian Food Recognition & Nutrition Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade Computer Vision and Nutrition Intelligence pipeline designed for **Nutriiya**. The system accepts an RGB image of an Indian meal, detects and classifies multi-dish components, calculates reasoned volumetric portion quantities (grams), and maps each item to reference Indian nutritional composition tables (**ICMR - National Institute of Nutrition / IFCT 2017**) to return comprehensive caloric and macronutrient breakdowns.

---

## Table of Contents
1. [Executive Summary & Objectives](#1-executive-summary--objectives)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Task 1 — Food Detection & Classification](#3-task-1--food-detection--classification)
4. [Task 2 — Portion Estimation Engine](#4-task-2--portion-estimation-engine)
5. [Task 3 — Nutrition Calculation & ICMR-NIN Mapping](#5-task-3--nutrition-calculation--icmr-nin-mapping)
6. [Task 4 — REST API (`POST /analyze-meal`)](#6-task-4--rest-api-post-analyze-meal)
7. [Model Evaluation & Benchmarking](#7-model-evaluation--benchmarking)
8. [Benchmark Sample Predictions (5 Diverse Meals)](#8-benchmark-sample-predictions-5-diverse-meals)
9. [Bonus Challenges & Solutions](#9-bonus-challenges--solutions)
10. [Local Environment Setup & Execution](#10-local-environment-setup--execution)
11. [Walkthrough & Live Demonstration Guide](#11-walkthrough--live-demonstration-guide)

---

## 1. Executive Summary & Objectives

Indian meals present unique challenges for traditional computer vision:
- **Multi-dish composite arrangements**: Meals are typically served on a single large thali, banana leaf, or individual katoris (small bowls).
- **Visually similar textures & colors**: Yellow lentils (Dal Tadka vs. Sambar), reddish-brown curries (Paneer Makhani vs. Chole), and white grains (steamed rice vs. idli).
- **Non-rigid geometry**: Semi-solid gravies and piled mounds lack strict polygonal outlines.
- **2D-to-3D portion ambiguity**: Standard RGB cameras lack depth sensors ($Z$-axis); portions vary widely based on bowl depth, packing density, and fluffiness.

**What This System Delivers:**
1. **Accurate Object Detection**: Fine-tuned **YOLOv8n** detector capable of simultaneously localizing and classifying up to 10 distinct Indian culinary categories with **99.5% mAP@50** and **98.98% mAP@50:95**.
2. **Reasoned Portion Estimation**: A physics-and-culinary-grounded volumetric model utilizing reference plate scaling (calibrated to standard $28\text{ cm}$ Indian thalis), serving shape factors, and food bulk densities.
3. **Nutritional Mapping**: Authoritative macronutrient mapping from ICMR-NIN (National Institute of Nutrition) and IFCT 2017.
4. **Production REST API**: High-throughput FastAPI application strictly implementing the candidate specification format.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    A[RGB Meal Image] --> B[FastAPI: POST /analyze-meal]
    B --> C[FoodDetector Engine]
    C -->|YOLOv8n Backbone| D[Multi-Dish Bounding Boxes + Confidence]
    D --> E[Spatial NMS Deduplication & Context Disambiguation]
    
    E --> F[PortionEstimator Engine]
    subgraph Portion Modeling
        F1[Plate Reference Scaling: 28cm Thali Calibrator]
        F2[Projected Food Footprint: Shape Factor Adjusted]
        F3[Volumetric Depth Formulation: Katori vs Mound vs Flatbread]
        F4[Food Bulk Density: ICMR g/cm3 Mapping]
        F1 --> F2 --> F3 --> F4
    end
    F --> Portion Modeling
    Portion Modeling --> G[Estimated Grams per Item ± Uncertainty]

    G --> H[NutritionCalculator Engine]
    subgraph Nutrition Database
        H1[(ICMR-NIN / IFCT 2017 Reference DB)]
        H2[Portion Scaling Factor: g / 100g]
        H3[Macronutrient Computation: Calories, Protein, Carbs, Fat]
        H4[Meal Total Aggregation & Unknown Safe Fallback]
        H1 --> H2 --> H3 --> H4
    end
    H --> Nutrition Database
    Nutrition Database --> I[Standard Nutriiya JSON Response + Visual Annotation]
```

---

## 3. Task 1 — Food Detection & Classification

### 3.1 Approach & Model Selection
- **Backbone**: Ultralytics **YOLOv8n** (Nano architecture).
  - *Parameters*: 3,134,894 (3.13M)
  - *Compute*: 8.6 GFLOPs
  - *Inference Latency*: **~51.7 ms on standard Intel Core CPU** (zero GPU dependency required for production inference).
- **What Was Implemented**:
  1. Fine-tuned the pretrained feature extractor on an Indian food dataset covering: `rice`, `dal`, `roti`, `paneer_curry`, `salad`, `chole`, `biryani`, `samosa`, `idli`, `dosa`.
  2. Implemented spatial IoU deduplication to eliminate spurious multi-box overlaps around liquid katoris.
  3. Built contextual disambiguation heuristics (e.g. resolving yellow dal vs. sambar using co-occurring meal elements).
  4. Created a pure-Pillow annotation engine for drawing bounding boxes and labels without external GUI system library dependencies.

### 3.2 Dataset Preparation & Splits
- **Dataset**: Nutriiya Indian Food Vision Dataset (NIFVD-v1) containing 140 multi-item meal images and 588 annotated bounding boxes.
- **Split Distribution**:
  - **Train**: 70% (90 images, 378 instances)
  - **Validation**: 15% (25 images, 105 instances)
  - **Test (Held-Out)**: 15% (25 images, 105 instances)
- **Data Augmentations**:
  - Plate background rotation (wood, marble, slate, linen)
  - Random scale jittering ($\pm 10\%$)
  - Specular reflection simulation on stainless steel thali rims
  - Ambient color temperature variation (warm dining vs. fluorescent lighting)

---

## 4. Task 2 — Portion Estimation Engine

### 4.1 Methodology
A single 2D RGB image cannot directly weigh food. To solve this practically, our engine implements a **multi-stage reasoned geometric estimation**:

1. **Plate Scale Calibration**:
   - In Indian dining, thalis and dinner plates follow standardized industrial dimensions.
   - Standard Indian steel thali diameter: $D_{\text{plate}} = 28.0\text{ cm}$ (Surface Area $\approx 615.75\text{ cm}^2$).
   - Pixel scale factor: $S = \frac{\text{Plate Diameter (px)}}{28.0\text{ cm}}$.
   - Pixel area conversion: $A_{\text{cm}^2} = \frac{\text{BBox Area (px)} \times \text{Shape Factor}}{S^2}$.

2. **Culinary Volumetric Profiles**:
   - **Planar Flatbreads (Roti/Chapati)**: Flat item with standardized dough mass. 1 standard $16\text{ cm}$ roti weighs $\approx 40\text{ g}$. Count is computed via surface ratio $\frac{\text{Detected Area}}{\text{Single Roti Area}}$.
   - **Katori Curries & Soups (Dal, Paneer Curry, Chole)**: Liquids level out inside a cylindrical katori bowl. Effective depth $h \approx 2.5\text{ to }2.8\text{ cm}$. $V = A_{\text{surface}} \times h \times \eta_{\text{meniscus}}$.
   - **Heaped Grains (Rice, Biryani)**: Basmati rice is served as a heaped dome/paraboloid. Effective depth scales with mound radius ($h_{\text{eff}} \approx 2.6\text{ to }3.2\text{ cm}$).
   - **Loose Vegetables (Salad)**: Kachumber / sliced salad has porous air gaps; modeled with low packing factor ($0.50$).

3. **Food Bulk Density ($\rho$)**:
   Densities were empirically calibrated from food science references:
   | Food Class | Typology | Shape Factor | Typical Depth ($h$) | Bulk Density ($\rho$) | Standard Serving |
   | :--- | :--- | :---: | :---: | :---: | :---: |
   | **Rice** | Heaped Paraboloid | 0.65 | 2.6 cm | $0.85\text{ g/cm}^3$ | 150 g |
   | **Dal** | Katori Cylinder | 0.85 | 2.8 cm | $1.02\text{ g/cm}^3$ | 100 g |
   | **Roti** | Planar Disc | 0.90 | 0.3 cm | Count-calibrated | 40 g / pc |
   | **Paneer Curry** | Gravy + Chunks | 0.80 | 2.5 cm | $1.05\text{ g/cm}^3$ | 120 g |
   | **Salad** | Porous Slices | 0.50 | 2.0 cm | $0.35\text{ g/cm}^3$ | 50 g |
   | **Chole** | Dense Legumes | 0.80 | 2.6 cm | $1.02\text{ g/cm}^3$ | 130 g |
   | **Biryani** | Layered Rice | 0.70 | 3.0 cm | $0.88\text{ g/cm}^3$ | 220 g |
   | **Samosa** | Discrete Snack | 0.75 | 3.5 cm | $65\text{ g / pc}$ | 130 g (2 pcs) |
   | **Idli** | Steamed Cake | 0.70 | 2.0 cm | $45\text{ g / pc}$ | 90 g (2 pcs) |
   | **Dosa** | Rolled Cylinder | 0.85 | 0.5 cm | $90\text{ g / pc}$ | 90 g |

4. **Realistic Bounds Clamping**:
   Estimates are clamped to plausible single-meal minimum and maximum servings ($[\text{min\_portion}, \text{max\_portion}]$) and rounded to 5g increments.

### 4.2 Assumptions & Uncertainty Disclosure
- **Assumptions**: Plate is photographed from an overhead angle ($60^\circ - 90^\circ$ relative to horizontal); plate diameter conforms to standard $28\text{ cm} \pm 2\text{ cm}$.
- **Limitations**: A 2D RGB image cannot measure bowl depth below the rim, sauce viscosity, or hidden oil/ghee content.
- **Uncertainty Interval**: The system computes a confidence range of $\pm 15\%$ for every estimated item rather than claiming scientific certitude.

---

## 5. Task 3 — Nutrition Calculation & ICMR-NIN Mapping

Nutritional data is derived from the **ICMR-NIN (Indian Council of Medical Research - National Institute of Nutrition) Indian Food Composition Tables (IFCT)**:

$$\text{Nutrient Value} = \frac{\text{Nutrient per } 100\text{g} \times \text{Estimated Quantity (g)}}{100}$$

### Reference Nutritional Values (per 100g edible portion):
| Food Item | Energy (kcal) | Protein (g) | Carbohydrates (g) | Fat (g) | Source / Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Rice (Cooked)** | 130.0 | 2.7 | 28.2 | 0.3 | IFCT 2017 Code A006 |
| **Dal (Tadka / Cooked)**| 104.0 | 6.8 | 15.2 | 1.8 | IFCT 2017 Code B012 |
| **Roti (Whole Wheat)** | 297.0 | 9.5 | 56.0 | 3.7 | 1 pc (40g) $\approx 119\text{ kcal}$ |
| **Paneer Curry** | 185.0 | 8.5 | 7.2 | 14.0 | Cottage cheese + spiced gravy |
| **Salad (Kachumber)** | 28.0 | 1.1 | 5.2 | 0.3 | Fresh cucumber, tomato, onion |
| **Chole (Chickpeas)** | 140.0 | 7.0 | 20.0 | 3.5 | IFCT 2017 Code B003 |
| **Biryani (Spiced Rice)**| 165.0 | 6.2 | 21.0 | 6.5 | Layered basmati + aromatics |
| **Samosa** | 262.0 | 3.5 | 32.0 | 13.5 | Fried pastry with potato filling |
| **Idli** | 132.0 | 4.6 | 26.0 | 0.4 | Steamed fermented rice/urad |
| **Dosa** | 168.0 | 3.9 | 29.0 | 3.8 | Crispy fermented crepe |
| **Unknown Dish** | 135.0 | 4.5 | 18.0 | 5.0 | Composite Indian meal average |

---

## 6. Task 4 — REST API (`POST /analyze-meal`)

The service is exposed using **FastAPI** with automatic Swagger OpenAPI documentation, CORS middleware, and input sanitization.

### 6.1 Required Endpoint Specification
- **Endpoint**: `POST /analyze-meal`
- **Request Format**: `multipart/form-data`
  - `file`: Image binary (JPEG, PNG, WebP)
  - `include_visualization`: Boolean (optional, default `false`)
  - `conf_threshold`: Float (optional, default `0.25`)

### 6.2 Compliant Response Example
```json
{
  "success": true,
  "meal_items": [
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
      "food_name": "Dal",
      "confidence": 1.0,
      "estimated_quantity_g": 175,
      "nutrition": {
        "calories": 182,
        "protein": 11.9,
        "carbs": 26.6,
        "fat": 3.1
      }
    },
    {
      "food_name": "Rice",
      "confidence": 0.99,
      "estimated_quantity_g": 260,
      "nutrition": {
        "calories": 338,
        "protein": 7,
        "carbs": 73.3,
        "fat": 0.8
      }
    },
    {
      "food_name": "Paneer curry",
      "confidence": 0.98,
      "estimated_quantity_g": 180,
      "nutrition": {
        "calories": 333,
        "protein": 15.3,
        "carbs": 13,
        "fat": 25.2
      }
    },
    {
      "food_name": "Roti",
      "confidence": 0.96,
      "estimated_quantity_g": 40,
      "nutrition": {
        "calories": 118.8,
        "protein": 3.8,
        "carbs": 22.4,
        "fat": 1.5
      }
    }
  ],
  "total_nutrition": {
    "calories": 978.8,
    "protein": 38.3,
    "carbs": 136.6,
    "fat": 30.7
  }
}
```

### 6.3 cURL Usage Example
```bash
curl -X POST "http://localhost:8000/analyze-meal" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_predictions/meal_01_thali.jpg"
```

### 6.4 Python Client Example
```python
import requests

url = "http://localhost:8000/analyze-meal"
with open("sample_predictions/meal_01_thali.jpg", "rb") as f:
    files = {"file": ("meal.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)
    print(response.json())
```

---

## 7. Model Evaluation & Benchmarking

The model was evaluated on the **25 held-out test images** (105 ground-truth dish instances).

### 7.1 Quantitative Metrics
| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **mAP@50** | **99.50%** | Exceptional precision at standard 50% IoU overlap |
| **mAP@50:95** | **98.98%** | Highly precise bounding box localization across all IoU scales |
| **Mean Precision** | **97.98%** | Minimal false positive detections |
| **Mean Recall** | **99.14%** | Captures virtually all food items on the plate |
| **Inference Latency** | **51.7 ms** | Real-time capable on standard consumer CPU |
| **Total Pipeline Latency**| **~98 ms** | Detection + Portion Estimation + Nutrition + JSON serialization |

### 7.2 Per-Class Evaluation Breakdown
| Class | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **Rice** | 0.966 | 1.000 | 0.995 | 0.978 |
| **Dal** | 0.928 | 1.000 | 0.995 | 0.995 |
| **Roti** | 0.993 | 1.000 | 0.995 | 0.995 |
| **Paneer Curry** | 0.992 | 1.000 | 0.995 | 0.995 |
| **Salad** | 0.991 | 1.000 | 0.995 | 0.980 |
| **Chole** | 1.000 | 0.914 | 0.995 | 0.995 |
| **Biryani** | 0.987 | 1.000 | 0.995 | 0.995 |
| **Samosa** | 0.956 | 1.000 | 0.995 | 0.995 |
| **Idli** | 0.999 | 1.000 | 0.995 | 0.975 |
| **Dosa** | 0.988 | 1.000 | 0.995 | 0.995 |

*Evaluation plots are saved in `evaluation_reports/`: `BoxPR_curve.png`, `BoxF1_curve.png`, `confusion_matrix.png`, and `results.png`.*

---

## 8. Benchmark Sample Predictions (5 Diverse Meals)

The system was executed across 5 meal compositions. Both annotated images and compliant JSON responses are stored in `sample_predictions/`.

| Sample File | Meal Style | Detected Items & Portions | Total Calories | Total Protein | Total Carbs | Total Fat |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `meal_01_thali.jpg` | **Classic Indian Thali** | Salad (25g), Dal (175g), Rice (260g), Paneer curry (180g), Roti (40g) | **978.8 kcal** | 38.3 g | 136.6 g | 30.7 g |
| `meal_02_punjabi.jpg`| **North Indian Feast** | Salad (25g), Chole (205g), Rice (225g), Paneer curry (200g), Roti (40g) | **1075.3 kcal** | 41.5 g | 142.5 g | 37.5 g |
| `meal_03_south_indian.jpg`| **South Indian Platter**| Salad (25g), Dal/Sambar (220g), Dosa (60g), Idli (45g) | **396.0 kcal** | 19.7 g | 63.8 g | 6.6 g |
| `meal_04_biryani_feast.jpg`| **Royal Biryani Meal** | Dal (175g), Salad (25g), Samosa (65g), Biryani (450g) | **1101.8 kcal** | 42.4 g | 143.2 g | 41.2 g |
| `meal_05_evening_snack.jpg`| **Snack Platter** | Salad (35g), Chole (180g), Samosa (130g) | **602.4 kcal** | 17.5 g | 79.4 g | 24.0 g |

---

## 9. Bonus Challenges & Solutions

### 9.1 Handling Visually Similar Food Items
- **Challenge**: Dal vs. Sambar vs. Rasam; Paneer Curry vs. Chole / Tofu curry.
- **Solution**:
  - *Contextual Co-occurrence Matrix*: Yellow lentil curry in the presence of Idli or Dosa is contextually disambiguated as South Indian Sambar. Yellow lentil curry beside Roti/Steamed Rice is classified as Dal Tadka.
  - *Granularity Texture Analysis*: Paneer curry exhibits rectangular white geometric contours; Chole displays spherical chickpea clusters. The model leverages these fine spatial features to maintain $>99\%$ discrimination precision.

### 9.2 Unknown or Unsupported Food Items (Safe Fallback)
- **Challenge**: An uncatalogued regional item (e.g., Baingan Bharta, Tindora, fish curry) appears on the plate.
- **Solution**:
  - If a detection produces an ambiguous classification or low confidence ($0.20 \le \text{conf} < 0.35$), the system maps it to `Unknown Food Item`.
  - Instead of throwing an error or omitting the item, it applies an empirical composite Indian meal fallback density ($0.80\text{ g/cm}^3$) and nutritional profile ($135\text{ kcal}$, $4.5\text{g P}$, $18.0\text{g C}$, $5.0\text{g F}$ per 100g) with an uncertainty flag in metadata.

### 9.3 Inference Latency Optimization
- **Measured CPU Latency**:
  - Preprocess: $0.7\text{ ms}$
  - Inference: $51.7\text{ ms}$
  - Postprocess & NMS: $45.6\text{ ms}$
  - **Total**: $\approx 98\text{ ms}$ per meal image.
- **Optimization Strategy**: The model is ready for ONNX Runtime / OpenVINO quantization (`INT8`) which provides a further $3\times$ speedup on commodity server hardware ($<20\text{ ms}$).

### 9.4 Extension to Regional Indian Cuisines
The architecture is structured to scale cleanly across regional taxonomies:
1. **South Indian (Tamil, Kerala, Andhra, Karnataka)**: Add classes for Avial, Poriyal, Rasam, Uttapam, Appam, Upma, Pesarattu. Plate calibration can dynamically switch between circular thali ($28\text{ cm}$) and rectangular Banana Leaf ($40\text{ cm} \times 25\text{ cm}$).
2. **East Indian (Bengali, Odia, Assamese)**: Add classes for Macher Jhol, Luchi, Chholar Dal, Shukto, Mishti Doi, Pakhala Bhat.
3. **West Indian (Gujarati, Maharashtrian, Rajasthani)**: Add classes for Dhokla, Thepla, Pav Bhaji, Misal, Puran Poli, Dal Baati Churma.
4. **Hierarchical Taxonomy**: A two-level classification head: `Level 1: Course Type` (Grain, Lentil, Curry, Dry Veg, Bread, Sweet) $\to$ `Level 2: Specific Regional Variant`.

---

## 10. Local Environment Setup & Execution

### 10.1 Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14
- Standard terminal (PowerShell, Bash, or Command Prompt)

### 10.2 Installation
```bash
# Clone the repository
git clone https://github.com/nutriiya-candidate/nutriiya-indian-food-ai.git
cd nutriiya-indian-food-ai

# Install dependencies
pip install -r requirements.txt
```

### 10.3 Run Unit & Integration Tests
```bash
python tests/test_pipeline.py
```
*Runs 4 automated test suites validating nutrition calculations, portion bounds, benchmark thali inference, and FastAPI endpoints.*

### 10.4 Run Interactive Meal Demonstration
```bash
python run_demo.py sample_predictions/meal_01_thali.jpg
```
*Outputs formatted ASCII table breakdown and saves annotated image with bounding boxes.*

### 10.5 Launch REST API Server
```bash
python api/main.py
```
*Or with Uvicorn:*
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 11. Walkthrough & Live Demonstration Guide

For the 20–30 minute technical walkthrough with the Nutriiya engineering panel:

1. **Architecture Walkthrough (5 min)**:
   - Walk through `models/food_detector.py`, `models/portion_estimator.py`, and `models/nutrition_calculator.py`.
   - Explain the 3-stage pipeline separation and defense-in-depth design.
2. **Portion Estimation Reasoning (5 min)**:
   - Walk through the plate reference scale equation ($D=28\text{ cm}$), geometric shape factors, and volumetric bulk density assumptions.
   - Explain why direct 2D regression fails without physical reference scaling and how our approach provides reasoned bounds.
3. **Live API Demonstration (10 min)**:
   - Open `http://localhost:8000/docs`.
   - Execute `POST /analyze-meal` with `meal_01_thali.jpg` (the prompt's exact example: Rice, Dal, Roti, Paneer curry, Salad).
   - Verify that the returned JSON matches the candidate prompt's exact structure and review the annotated bounding box image.
4. **Evaluation & Edge Cases (5 min)**:
   - Review `evaluation_reports/test_metrics.json` and confusion matrix.
   - Demonstrate bonus features: yellow dal vs. sambar disambiguation, unknown food safe fallback, and latency profile.
