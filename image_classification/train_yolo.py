"""
YOLOv8 Training Script for Kitchen Utensils Detection

This script trains a YOLOv8 model on the kitchen utensils dataset with
optimized parameters and comprehensive logging.
"""

import os
import sys
import torch
from pathlib import Path
from yolo_detector import YOLODetector
from ultralytics import YOLO
import yaml
from ultralytics.nn.tasks import DetectionModel
import torch.nn as nn

# Add safe globals for PyTorch 2.6
torch.serialization.add_safe_globals([
    DetectionModel,
    nn.modules.container.Sequential,
    nn.modules.conv.Conv2d,
    nn.modules.batchnorm.BatchNorm2d,
    nn.modules.activation.SiLU,
    nn.modules.pooling.AdaptiveAvgPool2d,
    nn.modules.linear.Linear,
    nn.modules.container.ModuleList,
    nn.modules.upsampling.Upsample
])

def create_dataset_yaml(data_dir, train_dir, val_dir, class_names):
    """Create YAML configuration file for YOLOv8 dataset."""
    yaml_data = {
        'path': data_dir,
        'train': train_dir,
        'val': val_dir,
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = os.path.join(data_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    return yaml_path


def train_yolo_model(
    data_yaml,
    model_size='n',  # n, s, m, l, x
    epochs=100,
    batch_size=16,
    img_size=640,
    device='0'  # GPU device
):
    """Train YOLOv8 model with optimized parameters."""
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Training parameters optimized for GTX 1080 Ti
    training_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'patience': 50,  # Early stopping patience
        'save': True,    # Save best model
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': True,   # Cache images for faster training
        'exist_ok': True,  # Overwrite existing experiment
        'pretrained': True,  # Use pretrained weights
        'optimizer': 'Adam',  # Optimizer
        'verbose': True,  # Print training progress
        'seed': 42,      # Random seed for reproducibility
        'deterministic': True,  # Deterministic training
        'single_cls': False,  # Multi-class detection
        'rect': False,   # Rectangular training
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 10,  # Disable mosaic augmentation for last 10 epochs
        'resume': False,  # Resume from last checkpoint
        'amp': True,     # Automatic Mixed Precision
        'lr0': 0.01,     # Initial learning rate
        'lrf': 0.01,     # Final learning rate
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.0005,  # Optimizer weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup momentum
        'warmup_bias_lr': 0.1,  # Warmup bias learning rate
        'box': 7.5,      # Box loss gain
        'cls': 0.5,      # Class loss gain
        'dfl': 1.5,      # Distribution focal loss gain
        'fl_gamma': 0.0,  # Focal loss gamma
        'label_smoothing': 0.0,  # Label smoothing epsilon
        'nbs': 64,       # Nominal batch size
        'overlap_mask': True,  # Masks should overlap during training
        'mask_ratio': 4,  # Mask downsample ratio
        'dropout': 0.0,  # Use dropout regularization
        'val': True,     # Validate training results
    }
    
    # Start training
    results = model.train(**training_args)
    return results


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
    best_model = YOLO('runs/detect/train/weights/best.pt')
    
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


def test_trained_model(model_path: str):
    """Test the trained model on sample images."""
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    print(f"\n" + "=" * 60)
    print("Testing Trained Model")
    print("=" * 60)
    
    # Initialize detector with trained model
    data_yaml = os.path.join('image_classification', 'utensils-wp5hm-yolo8', 'data.yaml')
    detector = YOLODetector(model_path=model_path, data_yaml_path=data_yaml)
    
    # Test images in the root directory
    test_images = [
        "chopboard.jpg",
        "fork.jpg", 
        "garlicpress.jpg",
        "garlicpress2.jpg",
        "peeler.png"
    ]
    
    print(f"Testing on sample images...")
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📸 Testing: {img_path}")
            try:
                detections, annotated_path = detector.predict_and_display(
                    img_path, 
                    conf_threshold=0.25, 
                    show_plot=False,  # Don't show plots in training script
                    save_annotated=True
                )
                
                if detections:
                    print(f"  ✅ Found {len(detections)} objects:")
                    for det in detections:
                        print(f"     - {det['class']}: {det['confidence']:.3f}")
                    print(f"  📁 Annotated image: {annotated_path}")
                else:
                    print(f"  ⚠️  No objects detected")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        else:
            print(f"  ⚠️  Image not found: {img_path}")


if __name__ == "__main__":
    print("Kitchen Utensils YOLOv8 Training Script")
    print("This will train a YOLOv8 model for kitchen utensil detection.\n")
    
    # Check if user wants to proceed
    response = input("Do you want to start training? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        # Train the model
        main()
        
        # Test the trained model
        trained_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        if trained_model_path:
            test_response = input(f"\nDo you want to test the trained model? (y/n): ").lower().strip()
            if test_response in ['y', 'yes']:
                test_trained_model(trained_model_path)
    else:
        print("Training cancelled.") 