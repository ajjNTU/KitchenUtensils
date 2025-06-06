#!/usr/bin/env python3
"""
Enhanced Multi-Object Dataset Creator with Albumentations

Creates realistic multi-object training scenes from single-object YOLO data
with kitchen-specific augmentations using albumentations.
"""

import os
import cv2
import numpy as np
import random
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import argparse

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

class EnhancedMultiObjectCreator:
    def __init__(self, source_yolo_path: str, output_path: str):
        """Initialize the enhanced multi-object creator."""
        self.source_path = Path(source_yolo_path)
        self.output_path = Path(output_path)
        
        # Load class configuration
        with open(self.source_path / 'data.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        self.class_names = self.config['names']
        
        print(f"🎨 Enhanced Multi-Object Creator initialized")
        print(f"   📁 Source: {self.source_path}")
        print(f"   📁 Output: {self.output_path}")
        print(f"   🎯 Classes: {len(self.class_names)}")
        print(f"   🔧 Albumentations: {'✅ Available' if ALBUMENTATIONS_AVAILABLE else '❌ Not available'}")
        
        self._setup_output_dirs()
        self._setup_augmentation()
    
    def _setup_output_dirs(self):
        """Create output directory structure."""
        for split in ['train', 'valid', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def _setup_augmentation(self):
        """Setup albumentations pipeline for kitchen-specific augmentations."""
        if not ALBUMENTATIONS_AVAILABLE:
            self.augmentation = None
            return
        
        self.augmentation = A.Compose([
            # Geometric transformations (light)
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.1, 
                rotate_limit=5, 
                border_mode=cv2.BORDER_REFLECT,
                p=0.4
            ),
            
            # Kitchen lighting simulation
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=1.0
                ),
                A.ColorJitter(
                    brightness=0.15, 
                    contrast=0.15, 
                    saturation=0.1, 
                    hue=0.05, 
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=1.0),
            ], p=0.6),
            
            # Realistic effects
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.15),
            
            # Kitchen environment effects
            A.RandomShadow(p=0.1),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.15),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,
            min_area=100,
        ))
        
        print("🔧 Kitchen-specific augmentation pipeline configured")
    
    def extract_object_crops(self, split: str = 'train', max_per_class: int = 50):
        """Extract object crops from YOLO dataset."""
        print(f"\n🔍 Extracting crops from {split} split...")
        
        crops_by_class = defaultdict(list)
        images_dir = self.source_path / split / 'images'
        labels_dir = self.source_path / split / 'labels'
        
        if not images_dir.exists():
            print(f"❌ {images_dir} not found")
            return {}
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        for img_path in image_files[:300]:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Read YOLO annotations
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    if class_id >= len(self.class_names):
                        continue
                    
                    class_name = self.class_names[class_id]
                    
                    # Skip if we have enough crops for this class
                    if len(crops_by_class[class_name]) >= max_per_class:
                        continue
                    
                    # Parse YOLO bbox
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert to pixel coordinates with padding
                    padding = 0.2
                    x_center_px = x_center * w
                    y_center_px = y_center * h
                    width_px = width * w * (1 + padding)
                    height_px = height * h * (1 + padding)
                    
                    x1 = max(0, int(x_center_px - width_px / 2))
                    y1 = max(0, int(y_center_px - height_px / 2))
                    x2 = min(w, int(x_center_px + width_px / 2))
                    y2 = min(h, int(y_center_px + height_px / 2))
                    
                    # Extract crop
                    if x2 > x1 and y2 > y1 and (x2-x1) > 40 and (y2-y1) > 40:
                        crop = image[y1:y2, x1:x2]
                        
                        crops_by_class[class_name].append({
                            'image': crop,
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': (x2-x1) * (y2-y1)
                        })
        
        # Sort by area
        for class_name in crops_by_class:
            crops_by_class[class_name].sort(key=lambda x: x['area'], reverse=True)
        
        print("✅ Extracted crops:")
        for class_name, crops in crops_by_class.items():
            print(f"   {class_name}: {len(crops)}")
        
        return crops_by_class
    
    def create_kitchen_backgrounds(self, size: Tuple[int, int] = (640, 640)):
        """Create realistic kitchen background variations."""
        w, h = size
        backgrounds = []
        
        # Kitchen countertop colors
        kitchen_colors = [
            [245, 245, 240],  # White marble
            [230, 225, 220],  # Light granite
            [210, 200, 190],  # Beige quartz
            [190, 175, 160],  # Wood butcher block
            [200, 195, 190],  # Gray concrete
            [220, 210, 195],  # Cream ceramic
        ]
        
        for color in kitchen_colors:
            # Solid color background
            bg = np.full((h, w, 3), color, dtype=np.uint8)
            backgrounds.append(bg)
            
            # Add subtle texture
            noise = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
            textured_bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            backgrounds.append(textured_bg)
        
        print(f"📸 Created {len(backgrounds)} kitchen-style backgrounds")
        return backgrounds
    
    def compose_scene_enhanced(self, objects: List[Dict], background: np.ndarray):
        """Compose multi-object scene with enhanced placement."""
        scene = background.copy()
        h, w = scene.shape[:2]
        
        placements = []
        occupied_areas = []
        
        # Sort objects by area (largest first)
        objects = sorted(objects, key=lambda x: x['area'], reverse=True)
        
        for obj in objects:
            # Random scale
            scale = random.uniform(0.4, 0.8)
            obj_h, obj_w = obj['image'].shape[:2]
            new_w = int(obj_w * scale)
            new_h = int(obj_h * scale)
            
            # Find position with better placement logic
            attempts = 0
            placed = False
            
            while attempts < 60 and not placed:
                margin = 30
                x = random.randint(margin, max(margin, w - new_w - margin))
                y = random.randint(margin, max(margin, h - new_h - margin))
                
                # Check overlap
                new_area = {'x1': x, 'y1': y, 'x2': x + new_w, 'y2': y + new_h}
                
                overlap = False
                for existing in occupied_areas:
                    if self._areas_overlap(new_area, existing, threshold=0.15):
                        overlap = True
                        break
                
                if not overlap:
                    placements.append({
                        'object': obj,
                        'x': x, 'y': y,
                        'width': new_w, 'height': new_h,
                        'scale': scale
                    })
                    occupied_areas.append(new_area)
                    placed = True
                
                attempts += 1
        
        # Render objects
        annotations = []
        for placement in placements:
            obj = placement['object']
            x, y = placement['x'], placement['y']
            obj_w, obj_h = placement['width'], placement['height']
            
            # Resize object
            resized_obj = cv2.resize(obj['image'], (obj_w, obj_h))
            
            # Simple blending
            scene[y:y+obj_h, x:x+obj_w] = resized_obj
            
            # Create YOLO annotation
            x_center = (x + obj_w / 2) / w
            y_center = (y + obj_h / 2) / h
            width_norm = obj_w / w
            height_norm = obj_h / h
            
            annotations.append([
                obj['class_id'], x_center, y_center, width_norm, height_norm
            ])
        
        return scene, annotations
    
    def _areas_overlap(self, area1: Dict, area2: Dict, threshold: float = 0.2) -> bool:
        """Check if two areas overlap significantly."""
        x_overlap = max(0, min(area1['x2'], area2['x2']) - max(area1['x1'], area2['x1']))
        y_overlap = max(0, min(area1['y2'], area2['y2']) - max(area1['y1'], area2['y1']))
        
        if x_overlap == 0 or y_overlap == 0:
            return False
        
        overlap_area = x_overlap * y_overlap
        area1_size = (area1['x2'] - area1['x1']) * (area1['y2'] - area1['y1'])
        area2_size = (area2['x2'] - area2['x1']) * (area2['y2'] - area2['y1'])
        
        min_area = min(area1_size, area2_size)
        return overlap_area > threshold * min_area
    
    def apply_augmentation(self, scene: np.ndarray, annotations: List):
        """Apply albumentations augmentation if available."""
        if not ALBUMENTATIONS_AVAILABLE or self.augmentation is None:
            return scene, annotations
        
        try:
            # Convert annotations format
            bboxes = []
            class_labels = []
            
            for ann in annotations:
                class_id, x_center, y_center, width, height = ann
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(int(class_id))
            
            # Apply augmentation
            augmented = self.augmentation(
                image=scene,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            # Convert back
            aug_annotations = []
            for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                x_center, y_center, width, height = bbox
                aug_annotations.append([class_id, x_center, y_center, width, height])
            
            return augmented['image'], aug_annotations
            
        except Exception as e:
            print(f"⚠️  Augmentation failed: {e}")
            return scene, annotations
    
    def generate_dataset(self, num_scenes: int = 800, objects_per_scene: Tuple[int, int] = (2, 5)):
        """Generate enhanced multi-object dataset."""
        print(f"\n🎨 Generating {num_scenes} enhanced multi-object scenes...")
        
        # Extract crops
        crops_by_class = self.extract_object_crops('train', max_per_class=40)
        
        if not crops_by_class:
            print("❌ No crops extracted")
            return
        
        # Create backgrounds
        backgrounds = self.create_kitchen_backgrounds()
        
        # Split ratios
        splits = {
            'train': int(num_scenes * 0.7),
            'valid': int(num_scenes * 0.2),
            'test': int(num_scenes * 0.1)
        }
        
        scene_count = 0
        
        for split, target_count in splits.items():
            print(f"\n📦 Creating {target_count} scenes for {split} split...")
            
            for i in range(target_count):
                try:
                    # Select random objects
                    num_objects = random.randint(*objects_per_scene)
                    available_classes = [cls for cls, crops in crops_by_class.items() if crops]
                    
                    if len(available_classes) < 2:
                        continue
                    
                    selected_objects = []
                    used_classes = set()
                    
                    for _ in range(num_objects):
                        # Avoid too many of same class
                        available = [cls for cls in available_classes if cls not in used_classes or len(used_classes) < 3]
                        if not available:
                            break
                        
                        class_name = random.choice(available)
                        obj = random.choice(crops_by_class[class_name])
                        selected_objects.append(obj)
                        used_classes.add(class_name)
                    
                    if len(selected_objects) < 2:  # Ensure multi-object
                        continue
                    
                    # Compose scene
                    background = random.choice(backgrounds).copy()
                    scene, annotations = self.compose_scene_enhanced(selected_objects, background)
                    
                    if len(annotations) < 2:
                        continue
                    
                    # Apply augmentation
                    scene_aug, annotations_aug = self.apply_augmentation(scene, annotations)
                    
                    if len(annotations_aug) < 2:
                        continue
                    
                    # Save scene
                    filename = f"multi_{split}_{i:06d}.jpg"
                    self._save_scene(scene_aug, annotations_aug, filename, split)
                    scene_count += 1
                    
                    if (i + 1) % 50 == 0:
                        print(f"   ✅ {i + 1}/{target_count} scenes created")
                
                except Exception as e:
                    print(f"⚠️  Error creating scene {i}: {e}")
                    continue
        
        # Create config file
        self._create_config()
        
        print(f"\n🎉 Enhanced dataset creation complete!")
        print(f"   📊 Total scenes: {scene_count}")
        print(f"   📁 Output: {self.output_path}")
    
    def _save_scene(self, scene: np.ndarray, annotations: List, filename: str, split: str):
        """Save scene and annotations."""
        # Convert RGB back to BGR for OpenCV
        scene_bgr = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
        
        # Save image
        img_path = self.output_path / split / 'images' / filename
        cv2.imwrite(str(img_path), scene_bgr)
        
        # Save labels
        label_path = self.output_path / split / 'labels' / f"{Path(filename).stem}.txt"
        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id, x_center, y_center, width, height = ann
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def _create_config(self):
        """Create data.yaml for the multi-object dataset."""
        config = {
            'train': str(self.output_path / 'train' / 'images'),
            'val': str(self.output_path / 'valid' / 'images'),
            'test': str(self.output_path / 'test' / 'images'),
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        with open(self.output_path / 'data.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📋 Enhanced config saved: {self.output_path / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(description='Create enhanced multi-object YOLO dataset')
    parser.add_argument('--source', default='image_classification/utensils-wp5hm-yolo8',
                       help='Source YOLO dataset path')
    parser.add_argument('--output', default='image_classification/multi_object_enhanced',
                       help='Output path for multi-object dataset')
    parser.add_argument('--scenes', type=int, default=800,
                       help='Number of scenes to generate')
    parser.add_argument('--objects', type=int, nargs=2, default=[2, 5],
                       help='Min and max objects per scene')
    
    args = parser.parse_args()
    
    creator = EnhancedMultiObjectCreator(args.source, args.output)
    creator.generate_dataset(
        num_scenes=args.scenes,
        objects_per_scene=tuple(args.objects)
    )

if __name__ == "__main__":
    main() 