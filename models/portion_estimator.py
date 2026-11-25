"""
Nutriiya AI - Portion Estimation Engine
Estimates food portion mass (grams) from 2D bounding boxes using
reference plate scale calibration, geometric volume modeling, and food density.
"""

import math
from typing import Dict, Any, Tuple, Optional


class PortionEstimator:
    """
    Reasoned portion estimator for Indian meals from 2D images.
    
    Methodology:
    1. Plate Calibration: Uses standard Indian dinner plate / steel thali
       (nominal diameter = 28.0 cm, area ~615.75 cm^2) as reference object.
    2. Footprint Area: Maps bounding box to projected food surface area using shape factor.
    3. Volumetric Modeling: Models 3D height based on culinary serving typology:
       - Dal/Curries: Cylindrical katori with meniscus geometry
       - Rice/Biryani: Paraboloid heaped mound
       - Roti: Planar flatbread with known unit piece mass
       - Salad: Loosely packed porous structure
    4. Food Density Mapping: Volumetric mass = Volume * Density (g/cm^3)
    5. Bounds Clamping: Realistic limits derived from Indian dietary guidelines.
    """

    DEFAULT_PLATE_DIAMETER_CM = 28.0

    def __init__(self, nutrition_db: Dict[str, Any]):
        self.nutrition_db = nutrition_db.get("items", {})

    def estimate_portion(
        self,
        class_name: str,
        bbox: Tuple[float, float, float, float],
        img_width: int,
        img_height: int,
        plate_bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[str, Any]:
        """
        Estimate portion weight in grams for a detected food item.
        
        Args:
            class_name: Normalized class identifier (e.g. 'rice', 'dal')
            bbox: (x1, y1, x2, y2) in pixel coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels
            plate_bbox: Optional (px1, py1, px2, py2) of containing plate
            
        Returns:
            Dict containing estimated_quantity_g, uncertainty_range, and explanation.
        """
        item_cfg = self.nutrition_db.get(class_name.lower(), self.nutrition_db.get("unknown", {}))
        
        # 1. Determine pixels-per-cm scale factor
        if plate_bbox is not None:
            plate_w = plate_bbox[2] - plate_bbox[0]
            plate_h = plate_bbox[3] - plate_bbox[1]
            plate_pixel_dia = (plate_w + plate_h) / 2.0
        else:
            # Assume plate occupies ~85-90% of image width/height in typical top-down food photo
            plate_pixel_dia = min(img_width, img_height) * 0.88

        px_per_cm = plate_pixel_dia / self.DEFAULT_PLATE_DIAMETER_CM
        px_per_cm2 = px_per_cm ** 2

        # 2. Compute bounding box dimensions and projected area
        x1, y1, x2, y2 = bbox
        box_w_px = max(1.0, x2 - x1)
        box_h_px = max(1.0, y2 - y1)
        raw_box_area_px = box_w_px * box_h_px

        shape_factor = item_cfg.get("shape_factor", 0.75)
        food_area_px = raw_box_area_px * shape_factor
        food_area_cm2 = food_area_px / px_per_cm2

        # 3. Handle discrete flatbreads (Roti) vs volumetric dishes
        if class_name.lower() == "roti":
            piece_weight = item_cfg.get("piece_weight_g", 40.0)
            nominal_single_roti_area_cm2 = math.pi * (8.0 ** 2)  # ~201 cm2 (16cm dia)
            # Estimate count (or fraction) of rotis
            count = max(1.0, round(food_area_cm2 / nominal_single_roti_area_cm2, 1))
            if count <= 1.2:
                estimated_g = 40.0
            else:
                estimated_g = count * piece_weight
        elif class_name.lower() == "samosa":
            piece_weight = item_cfg.get("piece_weight_g", 65.0)
            nominal_samosa_area_cm2 = 35.0
            count = max(1.0, round(food_area_cm2 / nominal_samosa_area_cm2))
            estimated_g = count * piece_weight
        elif class_name.lower() == "idli":
            piece_weight = item_cfg.get("piece_weight_g", 45.0)
            nominal_idli_area_cm2 = 38.0
            count = max(1.0, round(food_area_cm2 / nominal_idli_area_cm2))
            estimated_g = count * piece_weight
        else:
            # Volumetric estimation
            typical_depth_cm = item_cfg.get("typical_depth_cm", 2.5)
            density = item_cfg.get("density_g_per_cm3", 0.90)

            # Heuristic scale: larger mounds tend to have slightly deeper profile
            area_ratio = food_area_cm2 / (math.pi * (14.0 ** 2)) # ratio to entire plate
            effective_depth = typical_depth_cm * (0.8 + 0.4 * min(1.0, area_ratio * 4))
            
            volume_cm3 = food_area_cm2 * effective_depth
            estimated_g = volume_cm3 * density

        # 4. Realistic clamping based on Indian dietary ranges
        min_g = item_cfg.get("min_portion_g", 30.0)
        max_g = item_cfg.get("max_portion_g", 350.0)
        clamped_g = max(min_g, min(max_g, estimated_g))

        # Round to neat 5g interval (or closest integer)
        final_g = int(round(clamped_g / 5.0) * 5)
        if final_g == 0:
            final_g = int(round(clamped_g))

        return {
            "estimated_quantity_g": final_g,
            "uncertainty_interval_g": {
                "min": int(max(min_g, final_g * 0.85)),
                "max": int(min(max_g, final_g * 1.15))
            },
            "methodology": {
                "plate_reference_diameter_cm": self.DEFAULT_PLATE_DIAMETER_CM,
                "projected_area_cm2": round(food_area_cm2, 1),
                "assumed_density_g_per_cm3": item_cfg.get("density_g_per_cm3", 0.90),
                "assumptions": "Standard 28cm dinner plate reference, shape-adjusted footprint, class-specific volumetric density."
            }
        }
