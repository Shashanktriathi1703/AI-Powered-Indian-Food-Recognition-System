"""
Nutriiya AI & Nutrition Intelligence
Dataset Preparation and Multi-Dish Thali Generator
Generates realistic multi-item Indian meals with YOLO bounding box annotations.
"""

import os
import sys
import math
import random
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PROCESSED_DIR as OUTPUT_DIR

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

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def draw_rice(draw, box, rng):
    """Render cooked white/basmati rice mound with texture grains."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    # Base rice mound
    base_color = (245 + rng.randint(-5, 5), 244 + rng.randint(-5, 5), 238 + rng.randint(-5, 5))
    draw.ellipse([x1, y1, x2, y2], fill=base_color, outline=(220, 218, 205), width=2)
    
    # Individual rice grains & shadow variations
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = w / 2 * 0.9, h / 2 * 0.9
    for _ in range(int(w * h * 0.08)):
        # Random point inside ellipse
        ang = rng.uniform(0, 2 * math.pi)
        rad = math.sqrt(rng.uniform(0, 1))
        gx = cx + rx * rad * math.cos(ang)
        gy = cy + ry * rad * math.sin(ang)
        glen = rng.uniform(3, 7)
        gang = rng.uniform(0, math.pi)
        dx = glen * math.cos(gang)
        dy = glen * math.sin(gang)
        shade = rng.randint(220, 255)
        draw.line([gx - dx, gy - dy, gx + dx, gy + dy], fill=(shade, shade, shade - 8), width=rng.choice([1, 2]))
        
    # Optional garnish: coriander leaves or fried onions
    if rng.random() > 0.4:
        for _ in range(rng.randint(3, 8)):
            gx = cx + rng.uniform(-rx * 0.4, rx * 0.4)
            gy = cy + rng.uniform(-ry * 0.4, ry * 0.4)
            draw.ellipse([gx - 3, gy - 2, gx + 3, gy + 2], fill=(34, 120 + rng.randint(-15, 15), 34))


def draw_dal(draw, box, rng):
    """Render yellow/toor dal in katori/bowl with tadka (mustard seeds, cumin, chili)."""
    x1, y1, x2, y2 = box
    # Katori steel/silver rim
    draw.ellipse([x1 - 3, y1 - 3, x2 + 3, y2 + 3], fill=(195, 198, 205), outline=(150, 155, 165), width=3)
    
    # Dal soup base (golden turmeric yellow)
    dal_base = (225 + rng.randint(-15, 15), 175 + rng.randint(-15, 15), 25 + rng.randint(-10, 10))
    draw.ellipse([x1, y1, x2, y2], fill=dal_base, outline=(190, 140, 20), width=2)
    
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) / 2 * 0.85, (y2 - y1) / 2 * 0.85
    
    # Tadka (oil swirls, cumin seeds, mustard seeds)
    for _ in range(rng.randint(20, 45)):
        ang = rng.uniform(0, 2 * math.pi)
        rad = math.sqrt(rng.uniform(0, 1))
        tx = cx + rx * rad * math.cos(ang)
        ty = cy + ry * rad * math.sin(ang)
        c_type = rng.choice(["cumin", "mustard", "chili_oil", "coriander"])
        if c_type == "cumin":
            draw.line([tx - 2, ty - 1, tx + 2, ty + 1], fill=(70, 45, 20), width=1)
        elif c_type == "mustard":
            draw.ellipse([tx - 1, ty - 1, tx + 1, ty + 1], fill=(25, 20, 15))
        elif c_type == "chili_oil":
            draw.ellipse([tx - 4, ty - 3, tx + 4, ty + 3], fill=(200 + rng.randint(0, 40), 70, 20))
        elif c_type == "coriander":
            draw.ellipse([tx - 2, ty - 2, tx + 2, ty + 2], fill=(40, 140, 40))


def draw_roti(draw, box, rng):
    """Render round whole wheat roti/chapati with puffed golden spots and ghee shine."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    
    # Base roti color (warm golden wheat)
    base_color = (228 + rng.randint(-10, 10), 195 + rng.randint(-10, 10), 145 + rng.randint(-10, 10))
    draw.ellipse([x1, y1, x2, y2], fill=base_color, outline=(180, 140, 95), width=2)
    
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = w / 2 * 0.85, h / 2 * 0.85
    
    # Brown blister spots from tawa
    for _ in range(rng.randint(30, 60)):
        ang = rng.uniform(0, 2 * math.pi)
        rad = math.sqrt(rng.uniform(0, 1))
        bx = cx + rx * rad * math.cos(ang)
        by = cy + ry * rad * math.sin(ang)
        spot_r = rng.uniform(1.5, 4.5)
        spot_shade = rng.randint(110, 160)
        draw.ellipse([bx - spot_r, by - spot_r, bx + spot_r, by + spot_r], 
                     fill=(spot_shade, int(spot_shade * 0.7), int(spot_shade * 0.4)))
        
    # Fold line or slight puffed surface highlight
    draw.arc([cx - rx * 0.5, cy - ry * 0.3, cx + rx * 0.5, cy + ry * 0.3], 
             start=20, end=160, fill=(245, 220, 175), width=2)


def draw_paneer_curry(draw, box, rng):
    """Render paneer curry (reddish-orange rich gravy with white paneer cubes and cream swirl)."""
    x1, y1, x2, y2 = box
    # Katori rim
    draw.ellipse([x1 - 3, y1 - 3, x2 + 3, y2 + 3], fill=(195, 198, 205), outline=(150, 155, 165), width=3)
    
    # Rich orange-red tomato-cashew gravy
    gravy_color = (215 + rng.randint(-15, 15), 90 + rng.randint(-15, 15), 25 + rng.randint(-10, 10))
    draw.ellipse([x1, y1, x2, y2], fill=gravy_color, outline=(170, 60, 20), width=2)
    
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) / 2 * 0.75, (y2 - y1) / 2 * 0.75
    
    # Paneer cubes (square-ish white chunks)
    num_cubes = rng.randint(5, 10)
    for _ in range(num_cubes):
        ang = rng.uniform(0, 2 * math.pi)
        rad = math.sqrt(rng.uniform(0, 0.85))
        px = cx + rx * rad * math.cos(ang)
        py = cy + ry * rad * math.sin(ang)
        cube_sz = rng.randint(8, 14)
        c_fill = (248 + rng.randint(-5, 5), 245 + rng.randint(-5, 5), 235 + rng.randint(-5, 5))
        draw.rectangle([px - cube_sz/2, py - cube_sz/2, px + cube_sz/2, py + cube_sz/2], 
                       fill=c_fill, outline=(210, 180, 150))
        
    # Cream drizzle and coriander garnish
    draw.arc([cx - rx * 0.4, cy - ry * 0.4, cx + rx * 0.4, cy + ry * 0.4], 
             start=40, end=240, fill=(255, 252, 240), width=2)
    for _ in range(5):
        gx = cx + rng.uniform(-rx * 0.5, rx * 0.5)
        gy = cy + rng.uniform(-ry * 0.5, ry * 0.5)
        draw.ellipse([gx - 2, gy - 2, gx + 2, gy + 2], fill=(30, 130, 30))


def draw_salad(draw, box, rng):
    """Render fresh Indian salad (sliced cucumbers, red tomatoes, purple onion rings, green chili)."""
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], fill=(240, 240, 235), outline=(210, 210, 205))
    
    w, h = x2 - x1, y2 - y1
    num_slices = rng.randint(12, 24)
    for _ in range(num_slices):
        sx = x1 + rng.uniform(10, w - 10)
        sy = y1 + rng.uniform(10, h - 10)
        stype = rng.choice(["cucumber", "tomato", "onion", "lemon", "chili"])
        if stype == "cucumber":
            cr = rng.uniform(8, 14)
            draw.ellipse([sx - cr, sy - cr, sx + cr, sy + cr], fill=(225, 242, 220), outline=(50, 130, 50), width=2)
            draw.ellipse([sx - cr*0.6, sy - cr*0.6, sx + cr*0.6, sy + cr*0.6], fill=(210, 235, 200))
        elif stype == "tomato":
            tr = rng.uniform(7, 13)
            draw.ellipse([sx - tr, sy - tr, sx + tr, sy + tr], fill=(220, 45, 35), outline=(170, 25, 20), width=1)
        elif stype == "onion":
            orad = rng.uniform(9, 15)
            draw.ellipse([sx - orad, sy - orad, sx + orad, sy + orad], outline=(150, 50, 120), width=2)
        elif stype == "lemon":
            draw.pieslice([sx - 8, sy - 8, sx + 8, sy + 8], start=30, end=150, fill=(245, 225, 30), outline=(210, 190, 15))
        elif stype == "chili":
            draw.line([sx - 8, sy - 4, sx + 8, sy + 4], fill=(30, 140, 30), width=3)


def draw_chole(draw, box, rng):
    """Render Punjabi Chole (dark brown spiced chickpea curry)."""
    x1, y1, x2, y2 = box
    draw.ellipse([x1 - 3, y1 - 3, x2 + 3, y2 + 3], fill=(195, 198, 205), outline=(150, 155, 165), width=3)
    chole_base = (110 + rng.randint(-10, 10), 65 + rng.randint(-10, 10), 30 + rng.randint(-5, 5))
    draw.ellipse([x1, y1, x2, y2], fill=chole_base, outline=(80, 45, 20), width=2)
    
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) / 2 * 0.8, (y2 - y1) / 2 * 0.8
    # Chickpeas (round spheres)
    for _ in range(rng.randint(15, 25)):
        ang = rng.uniform(0, 2 * math.pi)
        rad = math.sqrt(rng.uniform(0, 0.9))
        cpx = cx + rx * rad * math.cos(ang)
        cpy = cy + ry * rad * math.sin(ang)
        cr = rng.uniform(4, 7)
        draw.ellipse([cpx - cr, cpy - cr, cpx + cr, cpy + cr], 
                     fill=(175 + rng.randint(-10, 10), 125 + rng.randint(-10, 10), 65 + rng.randint(-5, 5)), 
                     outline=(100, 60, 25))


def draw_biryani(draw, box, rng):
    """Render fragrant Biryani (multicolored spiced rice, fried onions, mint)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    draw.ellipse([x1, y1, x2, y2], fill=(230, 185, 110), outline=(180, 130, 60), width=2)
    
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = w / 2 * 0.9, h / 2 * 0.9
    for _ in range(int(w * h * 0.07)):
        ang = rng.uniform(0, 2 * math.pi)
        rad = math.sqrt(rng.uniform(0, 1))
        gx = cx + rx * rad * math.cos(ang)
        gy = cy + ry * rad * math.sin(ang)
        # Mix of yellow, orange, white rice grains
        grain_color = rng.choice([
            (250, 245, 235), # white
            (235, 175, 45),  # saffron yellow
            (215, 100, 35),  # spiced orange
            (95, 45, 15)     # brown fried onion
        ])
        draw.line([gx - 2, gy - 2, gx + 2, gy + 2], fill=grain_color, width=2)


def draw_samosa(draw, box, rng):
    """Render golden triangular fried samosas."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    
    # 1 or 2 samosas
    tri_points = [
        (cx, y1 + 5),
        (x1 + 5, y2 - 5),
        (x2 - 5, y2 - 5)
    ]
    draw.polygon(tri_points, fill=(215, 155, 60), outline=(160, 100, 30))
    # Crust folds and bubbles
    draw.line([(cx + x1 + 5)/2, (y1 + 5 + y2 - 5)/2, cx, cy], fill=(180, 120, 40), width=2)
    for _ in range(12):
        bx = rng.uniform(x1 + 10, x2 - 10)
        by = rng.uniform(y1 + 15, y2 - 10)
        draw.ellipse([bx - 1.5, by - 1.5, bx + 1.5, by + 1.5], fill=(160, 95, 25))


def draw_idli(draw, box, rng):
    """Render round fluffy white steamed idlis."""
    x1, y1, x2, y2 = box
    draw.ellipse([x1, y1, x2, y2], fill=(250, 250, 248), outline=(215, 215, 210), width=2)
    # Texture dimples
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) / 2 * 0.7, (y2 - y1) / 2 * 0.7
    for _ in range(15):
        ang = rng.uniform(0, 2 * math.pi)
        rad = math.sqrt(rng.uniform(0, 1))
        ix = cx + rx * rad * math.cos(ang)
        iy = cy + ry * rad * math.sin(ang)
        draw.ellipse([ix - 1, iy - 1, ix + 1, iy + 1], fill=(230, 230, 225))


def draw_dosa(draw, box, rng):
    """Render golden-brown rolled or folded crispy dosa."""
    x1, y1, x2, y2 = box
    draw.rounded_rectangle([x1, y1, x2, y2], radius=15, fill=(215, 160, 75), outline=(170, 115, 45), width=2)
    # Crispy roasted streaks
    for sx in range(int(x1 + 10), int(x2 - 10), 12):
        draw.line([sx, y1 + 5, sx + 5, y2 - 5], fill=(150, 85, 25), width=2)


RENDERERS = {
    "rice": draw_rice,
    "dal": draw_dal,
    "roti": draw_roti,
    "paneer_curry": draw_paneer_curry,
    "salad": draw_salad,
    "chole": draw_chole,
    "biryani": draw_biryani,
    "samosa": draw_samosa,
    "idli": draw_idli,
    "dosa": draw_dosa
}


def generate_single_meal_image(img_size=640, meal_pattern=None, seed=None):
    """
    Creates a full thali/meal image with standard 28cm steel plate representation
    and placed food bowls/portions, returning image and YOLO label list.
    """
    rng = random.Random(seed)
    
    # Table background (wood, marble, or restaurant mat)
    bg_style = rng.choice(["wood", "marble", "dark_slate", "warm_cloth"])
    if bg_style == "wood":
        bg_color = (rng.randint(70, 110), rng.randint(45, 75), rng.randint(25, 45))
    elif bg_style == "marble":
        bg_color = (rng.randint(220, 240), rng.randint(220, 240), rng.randint(225, 245))
    elif bg_style == "dark_slate":
        bg_color = (rng.randint(35, 55), rng.randint(38, 58), rng.randint(42, 62))
    else:
        bg_color = (rng.randint(180, 210), rng.randint(160, 190), rng.randint(140, 170))
        
    img = Image.new("RGB", (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw thali/plate (steel thali with rim)
    plate_margin = rng.randint(20, 35)
    plate_box = [plate_margin, plate_margin, img_size - plate_margin, img_size - plate_margin]
    
    # Steel plate outer rim and inner reflection
    plate_rim = (205 + rng.randint(-10, 10), 208 + rng.randint(-10, 10), 215 + rng.randint(-10, 10))
    plate_inner = (220 + rng.randint(-10, 10), 223 + rng.randint(-10, 10), 230 + rng.randint(-10, 10))
    draw.ellipse(plate_box, fill=plate_rim, outline=(170, 175, 185), width=5)
    draw.ellipse([plate_box[0] + 12, plate_box[1] + 12, plate_box[2] - 12, plate_box[3] - 12], 
                 fill=plate_inner, outline=(185, 190, 200), width=2)
    
    # Meal items placement
    items_to_place = []
    
    if meal_pattern == "standard_nutriiya_thali" or meal_pattern is None:
        # Example from prompt: Rice, Dal, Roti, Paneer curry, Salad
        dish_configs = [
            ("rice", [img_size * 0.28, img_size * 0.45, img_size * 0.72, img_size * 0.86]),
            ("dal", [img_size * 0.16, img_size * 0.14, img_size * 0.44, img_size * 0.42]),
            ("paneer_curry", [img_size * 0.54, img_size * 0.14, img_size * 0.84, img_size * 0.42]),
            ("roti", [img_size * 0.08, img_size * 0.42, img_size * 0.36, img_size * 0.72]),
            ("salad", [img_size * 0.65, img_size * 0.45, img_size * 0.88, img_size * 0.70])
        ]
        items_to_place = dish_configs
    elif meal_pattern == "punjabi_meal":
        items_to_place = [
            ("rice", [img_size * 0.30, img_size * 0.48, img_size * 0.70, img_size * 0.88]),
            ("chole", [img_size * 0.15, img_size * 0.14, img_size * 0.46, img_size * 0.44]),
            ("paneer_curry", [img_size * 0.54, img_size * 0.14, img_size * 0.85, img_size * 0.44]),
            ("roti", [img_size * 0.08, img_size * 0.44, img_size * 0.38, img_size * 0.74]),
            ("salad", [img_size * 0.66, img_size * 0.46, img_size * 0.89, img_size * 0.69])
        ]
    elif meal_pattern == "south_indian_platter":
        items_to_place = [
            ("idli", [img_size * 0.18, img_size * 0.30, img_size * 0.44, img_size * 0.56]),
            ("dosa", [img_size * 0.35, img_size * 0.58, img_size * 0.88, img_size * 0.88]),
            ("dal", [img_size * 0.52, img_size * 0.20, img_size * 0.82, img_size * 0.48]),
            ("salad", [img_size * 0.18, img_size * 0.62, img_size * 0.38, img_size * 0.82])
        ]
    elif meal_pattern == "biryani_feast":
        items_to_place = [
            ("biryani", [img_size * 0.20, img_size * 0.30, img_size * 0.80, img_size * 0.85]),
            ("salad", [img_size * 0.12, img_size * 0.10, img_size * 0.38, img_size * 0.34]),
            ("dal", [img_size * 0.60, img_size * 0.10, img_size * 0.88, img_size * 0.36]),
            ("samosa", [img_size * 0.38, img_size * 0.08, img_size * 0.60, img_size * 0.28])
        ]
    elif meal_pattern == "snack_platter":
        items_to_place = [
            ("samosa", [img_size * 0.18, img_size * 0.25, img_size * 0.48, img_size * 0.55]),
            ("chole", [img_size * 0.52, img_size * 0.22, img_size * 0.82, img_size * 0.52]),
            ("salad", [img_size * 0.30, img_size * 0.60, img_size * 0.68, img_size * 0.88])
        ]
        
    labels = []
    
    # Render each food item and collect YOLO annotations
    for food_name, box in items_to_place:
        # Slight jitter in placement & scaling
        jit_x = rng.uniform(-10, 10)
        jit_y = rng.uniform(-10, 10)
        jit_s = rng.uniform(0.95, 1.05)
        
        cx = (box[0] + box[2]) / 2 + jit_x
        cy = (box[1] + box[3]) / 2 + jit_y
        w = (box[2] - box[0]) * jit_s
        h = (box[3] - box[1]) * jit_s
        
        x1 = max(10, min(img_size - 10, cx - w/2))
        y1 = max(10, min(img_size - 10, cy - h/2))
        x2 = max(10, min(img_size - 10, cx + w/2))
        y2 = max(10, min(img_size - 10, cy + h/2))
        
        actual_box = [x1, y1, x2, y2]
        renderer = RENDERERS.get(food_name)
        if renderer:
            renderer(draw, actual_box, rng)
            
        # YOLO format: class_id, x_center, y_center, width, height (normalized 0..1)
        cls_id = CLASS_TO_ID[food_name]
        norm_cx = ((x1 + x2) / 2) / img_size
        norm_cy = ((y1 + y2) / 2) / img_size
        norm_w = (x2 - x1) / img_size
        norm_h = (y2 - y1) / img_size
        
        labels.append((cls_id, norm_cx, norm_cy, norm_w, norm_h))
        
    # Apply subtle realistic gaussian smoothing and sensor noise
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return img, labels


def generate_full_dataset(num_train=80, num_val=20, num_test=20):
    """Generates the full dataset with train, val, and test splits."""
    patterns = [
        "standard_nutriiya_thali", 
        "punjabi_meal", 
        "south_indian_platter", 
        "biryani_feast", 
        "snack_platter"
    ]
    
    splits = [
        ("train", num_train),
        ("val", num_val),
        ("test", num_test)
    ]
    
    total_images = 0
    total_instances = {c: 0 for c in CLASSES}
    
    for split_name, count in splits:
        img_dir = os.path.join(OUTPUT_DIR, "images", split_name)
        lbl_dir = os.path.join(OUTPUT_DIR, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        for i in range(count):
            pat = patterns[i % len(patterns)]
            seed = 1000 * len(split_name) + i * 17
            img, labels = generate_single_meal_image(img_size=640, meal_pattern=pat, seed=seed)
            
            img_path = os.path.join(img_dir, f"meal_{split_name}_{i:04d}.jpg")
            lbl_path = os.path.join(lbl_dir, f"meal_{split_name}_{i:04d}.txt")
            
            img.save(img_path, quality=92)
            
            with open(lbl_path, "w") as f:
                for cls_id, cx, cy, w, h in labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    total_instances[CLASSES[cls_id]] += 1
                    
            total_images += 1
            
    print(f"Generated {total_images} total meal images across splits.")
    print("Class instance distribution:", total_instances)


if __name__ == "__main__":
    generate_full_dataset(num_train=90, num_val=25, num_test=25)
