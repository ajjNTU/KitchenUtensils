import os
import torch
from pathlib import Path

# Temporarily set PyTorch to use legacy loading for compatibility
original_load = torch.load
def legacy_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = legacy_load

from ultralytics import YOLO

def main():
    # Set up paths
    base_dir = Path(__file__).parent
    dataset_path = base_dir / 'utensils-wp5hm-yolo8'
    
    # Change to dataset directory
    os.chdir(dataset_path)
    
    # Load YOLOv8-small model
    model = YOLO('yolov8s.pt')
    
    # Begin training
    print("🚀 Starting YOLOv8 training...")
    results = model.train(
        data='data.yaml',     # Path to data.yaml in the current folder
        epochs=100,           # Increased epochs for better results
        imgsz=640,           # Image size
        device=0             # Use GPU
    )
    
    print("✅ Training complete! Model saved in runs/detect/train/weights/best.pt")
    
    # Test the trained model on test set
    print("\n🧪 Testing trained model on test set...")
    
    # Go back to main directory to find the trained model
    os.chdir(base_dir.parent)
    
    # Load the trained model
    best_model = YOLO('runs/detect/train/weights/best.pt')
    
    # Go back to dataset directory for testing
    os.chdir(dataset_path)
    
    # Run validation on test split
    test_results = best_model.val(data='data.yaml', split='test')
    
    print("📊 Test Results:")
    print(f"   mAP50: {test_results.box.map50:.3f}")
    print(f"   mAP50-95: {test_results.box.map:.3f}")
    print(f"   Precision: {test_results.box.mp:.3f}")
    print(f"   Recall: {test_results.box.mr:.3f}")
    
    print("\n🎉 Training and testing complete!")
    print(f"📁 Best model: runs/detect/train/weights/best.pt")
    print(f"📁 Test results: runs/detect/train/")

if __name__ == '__main__':
    main() 