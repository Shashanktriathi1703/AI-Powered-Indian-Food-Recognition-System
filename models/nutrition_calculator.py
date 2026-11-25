"""
Nutriiya AI - Nutrition Calculation Engine
Calculates nutritional breakdown (calories, protein, carbs, fat) from
estimated portion weights and ICMR-NIN / IFCT reference composition tables.
"""

from typing import Dict, Any, List, Tuple


class NutritionCalculator:
    """
    Computes macronutrient and caloric values for Indian food items
    based on ICMR-NIN (National Institute of Nutrition) composition data.
    """

    def __init__(self, nutrition_db: Dict[str, Any]):
        self.db = nutrition_db
        self.items_db = nutrition_db.get("items", {})

    def resolve_food_item(self, query_name: str) -> Tuple[str, Dict[str, Any]]:
        """
        Resolves query food name to known database entry using
        exact key, case-insensitive match, or alias lookup.
        """
        clean_query = query_name.strip().lower().replace(" ", "_").replace("-", "_")

        # 1. Direct match
        if clean_query in self.items_db:
            return clean_query, self.items_db[clean_query]

        # 2. Check display names and aliases
        for key, entry in self.items_db.items():
            if clean_query == entry.get("display_name", "").lower().replace(" ", "_"):
                return key, entry
            for alias in entry.get("aliases", []):
                if clean_query == alias.lower().replace(" ", "_"):
                    return key, entry
                if alias.lower() in query_name.lower():
                    return key, entry

        # 3. Fallback to unknown
        return "unknown", self.items_db.get("unknown", {
            "display_name": "Unknown Dish",
            "reference_per_100g": {"calories": 135.0, "protein": 4.5, "carbs": 18.0, "fat": 5.0}
        })

    def calculate_item_nutrition(self, food_name: str, quantity_g: float) -> Dict[str, Any]:
        """
        Calculates calories, protein, carbs, and fat for a given quantity in grams.
        """
        key, entry = self.resolve_food_item(food_name)
        ref = entry.get("reference_per_100g", {})
        
        ratio = quantity_g / 100.0

        cal = round(ref.get("calories", 0.0) * ratio, 1)
        prot = round(ref.get("protein", 0.0) * ratio, 1)
        carbs = round(ref.get("carbs", 0.0) * ratio, 1)
        fat = round(ref.get("fat", 0.0) * ratio, 1)

        # Format integer-like values cleanly (e.g. 195.0 -> 195, 4.0 -> 4)
        def format_num(val):
            return int(val) if val.is_integer() else val

        return {
            "display_name": entry.get("display_name", food_name.capitalize()),
            "resolved_key": key,
            "nutrition": {
                "calories": format_num(cal),
                "protein": format_num(prot),
                "carbs": format_num(carbs),
                "fat": format_num(fat)
            }
        }

    def aggregate_meal_nutrition(self, item_nutritions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sums up calories, protein, carbs, and fat across all detected meal items.
        """
        total_cal = 0.0
        total_prot = 0.0
        total_carbs = 0.0
        total_fat = 0.0

        for item in item_nutritions:
            nut = item.get("nutrition", {})
            total_cal += float(nut.get("calories", 0))
            total_prot += float(nut.get("protein", 0))
            total_carbs += float(nut.get("carbs", 0))
            total_fat += float(nut.get("fat", 0))

        def format_num(val):
            val_rounded = round(val, 1)
            return int(val_rounded) if val_rounded.is_integer() else val_rounded

        return {
            "calories": format_num(total_cal),
            "protein": format_num(total_prot),
            "carbs": format_num(total_carbs),
            "fat": format_num(total_fat)
        }
