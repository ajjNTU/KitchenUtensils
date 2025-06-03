"""
Utensil name aliases and canonicalization for the Kitchen Utensils Chatbot.
"""
from typing import Dict

ALIASES: Dict[str, str] = {
    # blender
    "smoothie maker": "blender",
    "food blender": "blender",
    "blendor": "blender",
    # bowl
    "mixing bowl": "bowl",
    "salad bowl": "bowl",
    "bowls": "bowl",
    # can opener
    "tin opener": "canopener",
    "can opener": "canopener",
    # chopping board
    "cutting board": "choppingboard",
    "board": "choppingboard",
    # colander
    "strainer": "colander",
    "pasta drainer": "colander",
    "colendar": "colander",
    # cup & mug
    "mug": "cup",
    "jug": "cup",
    "cups": "cup",
    # dinner fork
    "fork": "dinnerfork",
    "table fork": "dinnerfork",
    "main fork": "dinnerfork",
    "forks": "dinnerfork",
    # dinner knife
    "table knife": "dinnerknife",
    "main knife": "dinnerknife",
    # fish slice
    "fish spatula": "fishslice",
    "slotted turner": "fishslice",
    "turner": "fishslice",
    # garlic press
    "garlic crusher": "garlicpress",
    # kitchen knife
    "chef knife": "kitchenknife",
    "chef's knife": "kitchenknife",
    "cook's knife": "kitchenknife",
    "knives": "kitchenknife",
    "knife": "kitchenknife",
    # ladle
    "soup ladle": "ladle",
    "ladles": "ladle",
    # pan
    "frying pan": "pan",
    "skillet": "pan",
    "pans": "pan",
    # peeler
    "vegetable peeler": "peeler",
    "potato peeler": "peeler",
    "peelers": "peeler",
    # saucepan
    "saucepan": "saucepan",
    "milk pan": "saucepan",
    "small pot": "saucepan",
    "pot": "saucepan",
    # spoon
    "serving spoon": "spoon",
    "tablespoon": "spoon",
    "spoons": "spoon",
    # teaspoon
    "tea spoon": "teaspoon",
    # tongs
    "kitchen tongs": "tongs",
    "serving tongs": "tongs",
    "tongs": "tongs",
    # tray
    "baking tray": "tray",
    "serving tray": "tray",
    "trays": "tray",
    # whisk
    "balloon whisk": "whisk",
    "egg whisk": "whisk",
    "whisks": "whisk",
    # wooden spoon
    "wooden stirrer": "woodenspoon",
    "wood spoon": "woodenspoon",
    "woodenspoon": "woodenspoon",
}

def canonical_name(name: str) -> str:
    """Return the canonical utensil name for a given alias or misspelling."""
    return ALIASES.get(name.lower(), name.lower()) 