#!/usr/bin/env python3
"""
Multi-Object Dataset Creator for Kitchen Utensils YOLO Training

Creates realistic multi-object training scenes from single-object YOLO data.
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

class MultiObjectCreator:
    def __init__(self, source_yolo_path: str, output_path: str):
        """Initialize the multi-object creator."""
        self.source_path = Path(source_yolo_path)
        self.output_path = Path(output_path)
        
        # Load class configuration
        with open(self.source_path / 'data.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        self.class_names = self.config['names']
        
        print(f"🎨 Multi-Object Creator initialized")
        print(f"   📁 Source: {self.source_path}")
        print(f"   📁 Output: {self.output_path}")
        print(f"   🎯 Classes: {len(self.class_names)}")
        
        self._setup_output_dirs()
    
    def _setup_output_dirs(self):
        """Create output directory structure."""
        for split in ['train', 'valid', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
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
        
        for img_path in image_files[:200]:  # Limit processing for speed
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
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
                    padding = 0.15  # 15% padding
                    x_center_px = x_center * w
                    y_center_px = y_center * h
                    width_px = width * w * (1 + padding)
                    height_px = height * h * (1 + padding)
                    
                    x1 = max(0, int(x_center_px - width_px / 2))
                    y1 = max(0, int(y_center_px - height_px / 2))
                    x2 = min(w, int(x_center_px + width_px / 2))
                    y2 = min(h, int(y_center_px + height_px / 2))
                    
                    # Extract crop
                    if x2 > x1 and y2 > y1 and (x2-x1) > 30 and (y2-y1) > 30:
                        crop = image[y1:y2, x1:x2]
                        crops_by_class[class_name].append({
                            'image': crop,
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': (x2-x1) * (y2-y1)
                        })
        
        # Sort by area (larger objects first)
        for class_name in crops_by_class:
            crops_by_class[class_name].sort(key=lambda x: x['area'], reverse=True)
        
        print("✅ Extracted crops:")
        for class_name, crops in crops_by_class.items():
            print(f"   {class_name}: {len(crops)}")
        
        return crops_by_class
    
    def create_background(self, size: Tuple[int, int] = (640, 640)):
        """Create simple background variations."""
        w, h = size
        backgrounds = []
        
        # Simple colored backgrounds
        colors = [
            [240, 240, 235],  # Off-white
            [220, 220, 215],  # Light gray
            [200, 180, 160],  # Beige
            [180, 160, 140],  # Wood tone
        ]
        
        for color in colors:
            bg = np.full((h, w, 3), color, dtype=np.uint8)
            backgrounds.append(bg)
        
        # Add simple gradients
        for color in colors[:2]:
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                ratio = i / h
                blend_color = [int(c * (0.8 + 0.4 * ratio)) for c in color]
                bg[i, :] = blend_color
            backgrounds.append(bg)
        
        return backgrounds
    
    def place_objects_simple(self, objects: List[Dict], scene_size: Tuple[int, int]):
        """Simple object placement avoiding major overlaps."""
        w, h = scene_size
        placements = []
        occupied_areas = []
        
        # Sort by size (largest first)
        objects = sorted(objects, key=lambda x: x['area'], reverse=True)
        
        for obj in objects:
            # Random scale
            scale = random.uniform(0.4, 0.9)
            obj_h, obj_w = obj['image'].shape[:2]
            new_w = int(obj_w * scale)
            new_h = int(obj_h * scale)
            
            # Find position
            attempts = 0
            placed = False
            
            while attempts < 50 and not placed:
                margin = 30
                x = random.randint(margin, max(margin, w - new_w - margin))
                y = random.randint(margin, max(margin, h - new_h - margin))
                
                # Check overlap with existing objects
                new_area = {'x1': x, 'y1': y, 'x2': x + new_w, 'y2': y + new_h}
                
                overlap = False
                for existing in occupied_areas:
                    if self._areas_overlap(new_area, existing):
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
        
        return placements
    
    def _areas_overlap(self, area1: Dict, area2: Dict) -> bool:
        """Check if two areas overlap significantly."""
        x_overlap = max(0, min(area1['x2'], area2['x2']) - max(area1['x1'], area2['x1']))
        y_overlap = max(0, min(area1['y2'], area2['y2']) - max(area1['y1'], area2['y1']))
        
        if x_overlap == 0 or y_overlap == 0:
            return False
        
        overlap_area = x_overlap * y_overlap
        area1_size = (area1['x2'] - area1['x1']) * (area1['y2'] - area1['y1'])
        area2_size = (area2['x2'] - area2['x1']) * (area2['y2'] - area2['y1'])
        
        # Overlap is significant if it's more than 20% of either object
        min_area = min(area1_size, area2_size)
        return overlap_area > 0.2 * min_area
    
    def compose_scene(self, objects: List[Dict], background: np.ndarray):
        """Compose multi-object scene."""
        scene = background.copy()
        h, w = scene.shape[:2]
        
        placements = self.place_objects_simple(objects, (w, h))
        annotations = []
        
        for placement in placements:
            obj = placement['object']
            x, y = placement['x'], placement['y']
            obj_w, obj_h = placement['width'], placement['height']
            
            # Resize object
            resized_obj = cv2.resize(obj['image'], (obj_w, obj_h))
            
            # Simple blending (you can improve this)
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
    
    def generate_dataset(self, num_scenes: int = 500, objects_per_scene: Tuple[int, int] = (2, 4)):
        """Generate multi-object dataset."""
        print(f"\n🎨 Generating {num_scenes} multi-object scenes...")
        
        # Extract crops
        crops_by_class = self.extract_object_crops('train', max_per_class=30)
        
        if not crops_by_class:
            print("❌ No crops extracted")
            return
        
        # Create backgrounds
        backgrounds = self.create_background()
        
        # Split ratios
        splits = {
            'train': int(num_scenes * 0.7),
            'valid': int(num_scenes * 0.2),
            'test': int(num_scenes * 0.1)
        }
        
        scene_count = 0
        
        for split, count in splits.items():
            print(f"\n📦 Creating {count} scenes for {split} split...")
            
            for i in range(count):
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
                    scene, annotations = self.compose_scene(selected_objects, background)
                    
                    if len(annotations) < 2:
                        continue
                    
                    # Save scene
                    filename = f"multi_{scene_count:06d}.jpg"
                    self._save_scene(scene, annotations, filename, split)
                    scene_count += 1
                    
                    if (i + 1) % 50 == 0:
                        print(f"   ✅ {i + 1}/{count} scenes created")
                
                except Exception as e:
                    print(f"⚠️  Error creating scene {i}: {e}")
                    continue
        
        # Create config file
        self._create_config()
        
        print(f"\n🎉 Dataset creation complete!")
        print(f"   📊 Total scenes: {scene_count}")
        print(f"   📁 Output: {self.output_path}")
    
    def _save_scene(self, scene: np.ndarray, annotations: List, filename: str, split: str):
        """Save scene and annotations."""
        # Save image
        img_path = self.output_path / split / 'images' / filename
        cv2.imwrite(str(img_path), scene)
        
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
        
        print(f"📋 Config saved: {self.output_path / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(description='Create multi-object YOLO dataset')
    parser.add_argument('--source', default='image_classification/utensils-wp5hm-yolo8',
                       help='Source YOLO dataset path')
    parser.add_argument('--output', default='image_classification/multi_object_dataset',
                       help='Output path for multi-object dataset')
    parser.add_argument('--scenes', type=int, default=500,
                       help='Number of scenes to generate')
    parser.add_argument('--objects', type=int, nargs=2, default=[2, 4],
                       help='Min and max objects per scene')
    
    args = parser.parse_args()
    
    creator = MultiObjectCreator(args.source, args.output)
    creator.generate_dataset(
        num_scenes=args.scenes,
        objects_per_scene=tuple(args.objects)
    )

if __name__ == "__main__":
    main()