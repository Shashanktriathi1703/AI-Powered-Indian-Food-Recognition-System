"""
Generate 5 benchmark meal images for Nutriiya AI testing and visual demonstration.
"""
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLES_DIR
from training.prepare_dataset import generate_single_meal_image

os.makedirs(SAMPLES_DIR, exist_ok=True)

sample_configs = [
    ("meal_01_thali.jpg", "standard_nutriiya_thali", 42, "Classic Indian Thali (Rice, Dal, Roti, Paneer curry, Salad)"),
    ("meal_02_punjabi.jpg", "punjabi_meal", 108, "North Indian Feast (Rice, Chole, Paneer curry, Roti, Salad)"),
    ("meal_03_south_indian.jpg", "south_indian_platter", 256, "South Indian Platter (Idli, Dosa, Dal/Sambar, Salad)"),
    ("meal_04_biryani_feast.jpg", "biryani_feast", 512, "Biryani Meal (Biryani, Samosa, Dal, Salad)"),
    ("meal_05_evening_snack.jpg", "snack_platter", 777, "Indian Snack Plate (Samosa, Chole, Salad)")
]

for filename, pattern, seed, desc in sample_configs:
    img, labels = generate_single_meal_image(img_size=640, meal_pattern=pattern, seed=seed)
    save_path = os.path.join(SAMPLES_DIR, filename)
    img.save(save_path, quality=95)
    print(f"Generated {filename} ({desc}) with {len(labels)} items.")
