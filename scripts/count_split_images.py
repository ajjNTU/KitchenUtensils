import os

# Paths
YOLO_ROOT = os.path.join("image_classification", "utensils-wp5hm-yolo8")
CNN_ROOT = os.path.join("image_classification", "cls_data")

splits = ["train", "valid", "test"]

def count_images_in_folder(folder):
    if not os.path.exists(folder):
        print(f"[WARN] Folder does not exist: {folder}")
        return 0
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if folder.endswith('test/images'):
        print(f"[DEBUG] Files in test/images: {files[:10]} ... total: {len(files)}")
    return len(files)

def count_images_in_subfolders(root):
    total = 0
    if not os.path.exists(root):
        return 0
    for sub in os.listdir(root):
        sub_path = os.path.join(root, sub)
        if os.path.isdir(sub_path):
            total += count_images_in_folder(sub_path)
    return total

print("Split | YOLO images | CNN crops")
print("------|-------------|-----------")
for split in splits:
    yolo_dir = os.path.join(YOLO_ROOT, split, "images")
    cnn_dir = os.path.join(CNN_ROOT, split)
    print(f"Checking YOLO: {yolo_dir}")
    print(f"Checking CNN:  {cnn_dir}")
    yolo_count = count_images_in_folder(yolo_dir)
    cnn_count = count_images_in_subfolders(cnn_dir)
    print(f"{split:5} | {yolo_count:11} | {cnn_count:9}")

print("\nSummary:")
print("YOLO: Number of original images per split (each image may have multiple objects)")
print("CNN:  Number of cropped objects per split (each object = one crop)") 