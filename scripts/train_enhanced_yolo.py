#!/usr/bin/env python3
"""
Enhanced YOLO Training Script

Trains YOLOv8 on the combined single-object and multi-object dataset
for improved multi-object detection capabilities.
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml
import argparse
from datetime import datetime

# Add comprehensive safe globals for PyTorch 2.6 compatibility
print("🔧 Configuring PyTorch 2.6 safe globals for ultralytics...")

# Import modules individually with error handling
safe_globals_list = []

# Basic PyTorch modules
try:
    import torch.nn as nn
    safe_globals_list.extend([
        nn.modules.container.Sequential,
        nn.modules.conv.Conv2d,
        nn.modules.batchnorm.BatchNorm2d,
        nn.modules.activation.SiLU,
        nn.modules.pooling.AdaptiveAvgPool2d,
        nn.modules.linear.Linear,
        nn.modules.container.ModuleList,
        nn.modules.upsampling.Upsample,
        nn.modules.pooling.MaxPool2d,
        nn.modules.activation.ReLU,
        nn.modules.activation.Hardswish,
        nn.modules.dropout.Dropout,
    ])
    print("   ✅ Basic PyTorch modules added")
except Exception as e:
    print(f"   ⚠️  Basic PyTorch modules failed: {e}")

# Ultralytics modules with individual error handling
ultralytics_modules = [
    ('ultralytics.nn.tasks', 'DetectionModel'),
    ('ultralytics.nn.modules', 'Conv'),
    ('ultralytics.nn.modules', 'C2f'),
    ('ultralytics.nn.modules', 'SPPF'),
    ('ultralytics.nn.modules', 'Detect'),
]

for module_path, class_name in ultralytics_modules:
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        safe_globals_list.append(cls)
        print(f"   ✅ Added {module_path}.{class_name}")
    except Exception as e:
        print(f"   ⚠️  Could not add {module_path}.{class_name}: {e}")

# Try to add the specific module mentioned in the error
try:
    from ultralytics.nn.modules import Conv
    safe_globals_list.append(Conv)
    print("   ✅ Added ultralytics.nn.modules.Conv")
except Exception as e:
    print(f"   ⚠️  Could not add ultralytics.nn.modules.Conv: {e}")

# Add the safe globals
try:
    torch.serialization.add_safe_globals(safe_globals_list)
    print(f"🔧 Successfully configured {len(safe_globals_list)} safe globals for PyTorch 2.6")
except Exception as e:
    print(f"❌ Failed to configure safe globals: {e}")
    print("   Training may fail due to PyTorch 2.6 compatibility issues")

class EnhancedYOLOTrainer:
    def __init__(self, data_path: str, model_size: str = 's'):
        """Initialize the enhanced YOLO trainer."""
        self.data_path = Path(data_path)
        self.model_size = model_size
        
        # Load dataset config
        with open(self.data_path / 'data.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.num_classes = self.config['nc']
        self.class_names = self.config['names']
        
        print(f"🚀 Enhanced YOLO Trainer initialized")
        print(f"   📁 Dataset: {self.data_path}")
        print(f"   🎯 Classes: {self.num_classes}")
        print(f"   📦 Model: YOLOv8{model_size}")
        print(f"   🔧 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    def train_enhanced_model(self, epochs: int = 100, batch_size: int = 16, 
                           img_size: int = 640, patience: int = 50,
                           save_dir: str = None):
        """
        Train YOLOv8 model with enhanced parameters for multi-object detection.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size
            patience: Early stopping patience
            save_dir: Directory to save results
        """
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"runs/detect/enhanced_train_{timestamp}"
        
        print(f"\n🎨 Starting Enhanced YOLO Training...")
        print(f"   📊 Training parameters:")
        print(f"     Epochs: {epochs}")
        print(f"     Batch size: {batch_size}")
        print(f"     Image size: {img_size}")
        print(f"     Patience: {patience}")
        print(f"     Save directory: {save_dir}")
        
        # Initialize model with safe globals context manager for PyTorch 2.6
        print("🔧 Loading YOLO model with PyTorch 2.6 compatibility...")
        
        # Import the Conv module that's causing the issue
        try:
            from ultralytics.nn.modules import Conv
            
            # Use context manager approach recommended in error message
            with torch.serialization.safe_globals([Conv]):
                model = YOLO(f'yolov8{self.model_size}.pt')
            
            print("✅ YOLO model loaded successfully with safe globals context")
            
        except Exception as e:
            print(f"⚠️  Context manager approach failed: {e}")
            print("   Trying alternative approach...")
            
            # Fallback: try loading with weights_only=False (less secure but functional)
            try:
                # Monkey patch torch.load to use weights_only=False temporarily
                original_load = torch.load
                
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = patched_load
                model = YOLO(f'yolov8{self.model_size}.pt')
                torch.load = original_load  # Restore original
                
                print("✅ YOLO model loaded with weights_only=False fallback")
                
            except Exception as fallback_e:
                print(f"❌ All loading approaches failed: {fallback_e}")
                raise fallback_e
        
        # Enhanced training parameters for multi-object detection
        training_args = {
            'data': str(self.data_path / 'data.yaml'),
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'patience': patience,
            'save': True,
            'cache': True,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'project': save_dir.split('/')[0] if '/' in save_dir else 'runs',
            'name': save_dir.split('/')[-1] if '/' in save_dir else save_dir,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',  # Better for complex scenes
            'verbose': True,
            'seed': 42,
            'single_cls': False,
            'cos_lr': True,
            'resume': False,
            'amp': False,  # Disable AMP to avoid PyTorch 2.6 compatibility issues
            
            # Learning rate schedule optimized for multi-object
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss weights optimized for multi-object detection
            'box': 7.5,      # Standard box loss
            'cls': 0.5,      # Class loss
            'dfl': 1.5,      # Distribution focal loss
            'label_smoothing': 0.0,
            
            # Kitchen-specific augmentation settings
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,    # No rotation for kitchen scenes
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,      # No shear for kitchen utensils
            'perspective': 0.0, # No perspective for countertop views
            'flipud': 0.0,     # No vertical flip
            'fliplr': 0.5,     # Horizontal flip only
            'mosaic': 1.0,     # Mosaic augmentation for multi-object training
            'mixup': 0.0,      # No mixup to preserve object integrity
            
            # Validation settings
            'val': True,
            'plots': True,
        }
        
        print(f"\n⏱️  Training started...")
        results = model.train(**training_args)
        
        print(f"\n✅ Training completed!")
        
        # Get best model path
        best_model_path = Path(save_dir) / 'weights' / 'best.pt'
        if not best_model_path.exists():
            # Try alternative path structure
            best_model_path = Path('runs') / 'detect' / save_dir.split('/')[-1] / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            print(f"   🏆 Best model: {best_model_path}")
            
            # Test the model
            self.test_enhanced_model(str(best_model_path))
        else:
            print("⚠️  Could not find best model for testing")
        
        return results
    
    def test_enhanced_model(self, model_path: str):
        """Test the enhanced model on the test set."""
        print(f"\n🧪 Testing enhanced model...")
        print(f"   📍 Model: {model_path}")
        
        # Load trained model
        model = YOLO(model_path)
        
        # Run validation on test split
        test_results = model.val(
            data=str(self.data_path / 'data.yaml'),
            split='test',
            save_json=True,
            plots=True
        )
        
        print(f"\n📊 Enhanced Model Test Results:")
        print(f"   🎯 mAP50: {test_results.box.map50:.3f}")
        print(f"   📈 mAP50-95: {test_results.box.map:.3f}")
        print(f"   🔍 Precision: {test_results.box.mp:.3f}")
        print(f"   📝 Recall: {test_results.box.mr:.3f}")
        
        # Compare with original baseline
        original_map50 = 0.972  # Your original performance
        improvement = test_results.box.map50 - original_map50
        
        print(f"\n📈 Performance Comparison:")
        print(f"   🔹 Original mAP50: {original_map50:.3f}")
        print(f"   🔸 Enhanced mAP50: {test_results.box.map50:.3f}")
        if improvement > 0:
            print(f"   ✅ Improvement: +{improvement:.3f} ({improvement/original_map50*100:.1f}%)")
        else:
            print(f"   📊 Change: {improvement:.3f} ({improvement/original_map50*100:.1f}%)")
        
        return test_results
    
    def analyze_multi_object_performance(self, model_path: str):
        """Analyze performance specifically on multi-object scenes."""
        print(f"\n🔍 Analyzing multi-object performance...")
        
        # This would require loading the test images and separating
        # single-object vs multi-object results
        # For now, we'll provide general guidance
        
        print(f"   💡 To analyze multi-object performance:")
        print(f"     1. Test on images with 2+ objects")
        print(f"     2. Check detection completeness (all objects found)")
        print(f"     3. Verify spatial accuracy (bounding box precision)")
        print(f"     4. Assess class confusion in multi-object scenes")

def main():
    parser = argparse.ArgumentParser(description='Train enhanced YOLO for multi-object detection')
    parser.add_argument('--data', default='image_classification/combined_dataset',
                       help='Path to combined dataset')
    parser.add_argument('--model', default='s', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--save-dir', default=None,
                       help='Save directory for results')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    data_path = Path(args.data)
    if not data_path.exists() or not (data_path / 'data.yaml').exists():
        print(f"❌ Dataset not found: {data_path}")
        print("   Run combine_datasets.py first to create the combined dataset")
        return
    
    # Initialize trainer
    trainer = EnhancedYOLOTrainer(args.data, args.model)
    
    # Start training
    trainer.train_enhanced_model(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        patience=args.patience,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main() 