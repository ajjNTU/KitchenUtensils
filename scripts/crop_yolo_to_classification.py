import os
from PIL import Image

# Adjust these paths:
INPUT_ROOT = os.path.join("image_classification", "utensils-wp5hm-yolo8")
OUTPUT_ROOT = os.path.join("image_classification", "cls_data")

# Class mapping from data.yaml
classes = [
    "Blender", "Bowl", "Canopener", "Choppingboard", "Colander", "Cup", "Dinnerfork", "Dinnerknife", "Fishslice", "Garlicpress", "Kitchenknife", "Ladle", "Pan", "Peeler", "Saucepan", "Spoon", "Teaspoon", "Tongs", "Tray", "Whisk", "Woodenspoon"
]

splits = ["train", "valid", "test"]

for split in splits:
    img_dir = os.path.join(INPUT_ROOT, split, "images")
    label_dir = os.path.join(INPUT_ROOT, split, "labels")
    out_root = os.path.join(OUTPUT_ROOT, split)

    os.makedirs(out_root, exist_ok=True)

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(img_dir, fname)
        label_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(label_path):
            continue  # No label file

        img = Image.open(img_path)
        width, height = img.size

        with open(label_path) as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # skip invalid lines
                class_idx, x_center, y_center, w, h = map(float, parts[:5])
                class_idx = int(class_idx)
                if class_idx < 0 or class_idx >= len(classes):
                    continue  # skip invalid class
                class_name = classes[class_idx]
                # YOLO format is normalized [0,1]
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                xmin = int(x_center - w / 2)
                ymin = int(y_center - h / 2)
                xmax = int(x_center + w / 2)
                ymax = int(y_center + h / 2)
                # Clamp to image bounds
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width, xmax)
                ymax = min(height, ymax)
                # Skip empty crops
                if xmax <= xmin or ymax <= ymin:
                    print(f"Warning: Empty crop for {fname} bbox {idx} (class {class_name}) - Skipping.")
                    continue
                crop = img.crop((xmin, ymin, xmax, ymax))
                out_dir = os.path.join(out_root, class_name)
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"{os.path.splitext(fname)[0]}_{idx}.jpg"
                crop.save(os.path.join(out_dir, out_name)) 