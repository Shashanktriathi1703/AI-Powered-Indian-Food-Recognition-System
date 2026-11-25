"""
Nutriiya AI - Centralized Configuration & Path Loader
Loads all filesystem paths from `.env` (with safe defaults relative to repo root).
"""

import os


def _load_dotenv(env_path: str) -> None:
    """Load key=value pairs from .env into os.environ without external dependencies."""
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_load_dotenv(os.path.join(_REPO_DIR, ".env"))

PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or _REPO_DIR


def resolve_path(env_key: str, default_rel: str) -> str:
    """Read path from environment variable and resolve relative paths against PROJECT_ROOT."""
    val = os.environ.get(env_key) or default_rel
    if os.path.isabs(val):
        return os.path.normpath(val)
    return os.path.normpath(os.path.join(PROJECT_ROOT, val))


MODEL_WEIGHTS_PATH = resolve_path("MODEL_WEIGHTS_PATH", "models/weights/best.pt")
BASE_YOLO_WEIGHTS = resolve_path("BASE_YOLO_WEIGHTS", "yolov8n.pt")
NUTRITION_DB_PATH = resolve_path("NUTRITION_DB_PATH", "data/nutrition_db.json")
OUTPUT_WEIGHTS_DIR = resolve_path("OUTPUT_WEIGHTS_DIR", "models/weights")

DATASET_YAML_PATH = resolve_path("DATASET_YAML_PATH", "data/dataset.yaml")
DATA_PROCESSED_DIR = resolve_path("DATA_PROCESSED_DIR", "data/processed")
DATA_LABELS_DIR = resolve_path("DATA_LABELS_DIR", "data/processed/labels")
RUNS_DIR = resolve_path("RUNS_DIR", "runs")
EVAL_RUN_DIR = resolve_path("EVAL_RUN_DIR", "runs/nutriiya_yolov8n")
REPORTS_DIR = resolve_path("REPORTS_DIR", "evaluation_reports")

SAMPLES_DIR = resolve_path("SAMPLES_DIR", "sample_predictions")
REAL_IMAGES_DIR = resolve_path("REAL_IMAGES_DIR", "real_images")
DEFAULT_DEMO_IMAGE = resolve_path("DEFAULT_DEMO_IMAGE", "sample_predictions/meal_01_thali.jpg")
