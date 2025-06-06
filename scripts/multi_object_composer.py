"""
Multi-Object Scene Composer for Kitchen Utensils YOLO Training

This script creates realistic multi-object training scenes from single-object YOLO data
by intelligently composing objects with proper spatial relationships and realistic
augmentations using albumentations.
"""

import os
import cv2
import numpy as np
import random
import json
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from dataclasses import dataclass
import shutil
from collections import defaultdict

@dataclass
class ObjectCrop:
    """Represents a cropped object with metadata."""
    image: np.ndarray
    class_id: int
    class_name: str
    original_bbox: List[float]  # [x_center, y_center, width, height] normalized
    source_image: str
    confidence_area: float  # Area as confidence proxy

@dataclass
class PlacementConstraints:
    """Constraints for realistic object placement."""
    min_scale: float = 0.3
    max_scale: float = 1.2
    avoid_edge_margin: float = 0.05  # 5% margin from edges
    min_object_distance: float = 0.02  # Minimum distance between objects
    max_overlap_ratio: float = 0.15  # Maximum allowed overlap

class MultiObjectComposer:
    """Creates realistic multi-object scenes from single-object YOLO data."""
    
    def __init__(self, yolo_dataset_path: str, output_path: str, class_names: List[str]):
        """
        Initialize the multi-object composer.
        
        Args:
            yolo_dataset_path: Path to existing YOLO dataset
            output_path: Path to save generated multi-object scenes
            class_names: List of class names in order
        """
        self.yolo_path = Path(yolo_dataset_path)
        self.output_path = Path(output_path)
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        
        # Object relationship rules for realistic placement
        self.placement_rules = self._create_placement_rules()
        
        # Kitchen-specific augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
        # Ensure output directories exist
        self._setup_output_directories()
        
        print(f"🎨 Multi-Object Composer initialized")
        print(f"   📁 Source: {self.yolo_path}")
        print(f"   📁 Output: {self.output_path}")
        print(f"   🎯 Classes: {len(self.class_names)}")
    
    def _create_placement_rules(self) -> Dict[str, Dict]:
        """Create realistic placement rules for kitchen utensils."""
        return {
            'size_groups': {
                'large': ['pan', 'saucepan', 'tray', 'choppingboard', 'bowl'],
                'medium': ['whisk', 'ladle', 'tongs', 'colander', 'blender'],
                'small': ['spoon', 'fork', 'teaspoon', 'knife', 'peeler', 'canopener']
            },
            'grouping_preferences': {
                # Objects that often appear together
                'utensil_sets': ['dinnerfork', 'dinnerknife', 'spoon'],
                'cooking_tools': ['whisk', 'ladle', 'tongs', 'woodenspoon'],
                'cutting_tools': ['kitchenknife', 'choppingboard', 'peeler'],
                'containers': ['bowl', 'cup', 'saucepan', 'pan']
            },
            'placement_hierarchy': {
                # Large items typically placed first (background)
                'background': ['tray', 'choppingboard', 'pan', 'saucepan'],
                'foreground': ['spoon', 'fork', 'knife', 'whisk', 'ladle']
            }
        }
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create kitchen-specific augmentation pipeline using albumentations."""
        return A.Compose([
            # Geometric transformations
            A.OneOf([
                A.RandomSizedBBoxSafeCrop(height=640, width=640, p=0.7),
                A.BBoxSafeRandomCrop(erosion_rate=0.1, p=0.3),
            ], p=0.5),
            
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.15, 
                rotate_limit=10, 
                border_mode=cv2.BORDER_REFLECT,
                p=0.6
            ),
            
            # Kitchen lighting simulation
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.25, 
                    contrast_limit=0.25, 
                    p=1.0
                ),
                A.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.15, 
                    hue=0.05, 
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(85, 120), p=1.0),
            ], p=0.7),
            
            # Environmental effects
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Kitchen environment simulation
            A.RandomShadow(p=0.15),
            A.OpticalDistortion(distort_limit=0.05, p=0.1),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.2,  # Keep objects with at least 20% visibility
            min_area=50,         # Remove very small objects
        ))
    
    def _setup_output_directories(self):
        """Create output directory structure."""
        for split in ['train', 'valid', 'test']:
            (self.output_path / 'multi_object' / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / 'multi_object' / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def extract_object_crops(self, split: str = 'train', max_crops_per_class: int = 100) -> Dict[str, List[ObjectCrop]]:
        """
        Extract clean object crops from YOLO dataset.
        
        Args:
            split: Dataset split to extract from ('train', 'valid', 'test')
            max_crops_per_class: Maximum crops to extract per class
            
        Returns:
            Dictionary mapping class names to lists of ObjectCrop instances
        """
        print(f"\n🔍 Extracting object crops from {split} split...")
        
        crops_by_class = defaultdict(list)
        images_dir = self.yolo_path / split / 'images'
        labels_dir = self.yolo_path / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"❌ Split {split} not found in {self.yolo_path}")
            return {}
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Read annotations
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                if class_id >= len(self.class_names):
                    continue
                
                class_name = self.class_names[class_id].lower()
                
                # Skip if we have enough crops for this class
                if len(crops_by_class[class_name]) >= max_crops_per_class:
                    continue
                
                # Parse YOLO bbox (normalized)
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert to pixel coordinates
                x_center_px = x_center * w
                y_center_px = y_center * h
                width_px = width * w
                height_px = height * h
                
                # Calculate crop bounds with padding
                padding = 0.1  # 10% padding around object
                x1 = max(0, int(x_center_px - width_px * (0.5 + padding)))
                y1 = max(0, int(y_center_px - height_px * (0.5 + padding)))
                x2 = min(w, int(x_center_px + width_px * (0.5 + padding)))
                y2 = min(h, int(y_center_px + height_px * (0.5 + padding)))
                
                # Skip if crop is too small
                if x2 - x1 < 32 or y2 - y1 < 32:
                    continue
                
                # Extract crop
                crop = image[y1:y2, x1:x2]
                
                # Calculate confidence proxy based on area
                crop_area = (x2 - x1) * (y2 - y1)
                confidence_area = crop_area / (w * h)  # Normalized area
                
                # Create ObjectCrop
                object_crop = ObjectCrop(
                    image=crop,
                    class_id=class_id,
                    class_name=class_name,
                    original_bbox=[x_center, y_center, width, height],
                    source_image=img_path.name,
                    confidence_area=confidence_area
                )
                
                crops_by_class[class_name].append(object_crop)
        
        # Sort crops by confidence (area) and keep the best ones
        for class_name in crops_by_class:
            crops_by_class[class_name].sort(key=lambda x: x.confidence_area, reverse=True)
            crops_by_class[class_name] = crops_by_class[class_name][:max_crops_per_class]
        
        print(f"✅ Extracted crops by class:")
        for class_name, crops in crops_by_class.items():
            print(f"   {class_name}: {len(crops)} crops")
        
        return crops_by_class
    
    def create_background_variants(self, base_size: Tuple[int, int] = (640, 640), 
                                 kitchen_photos_path: Optional[str] = None) -> List[np.ndarray]:
        """
        Create various background images for composition.
        
        Args:
            base_size: Size of background images
            kitchen_photos_path: Optional path to real kitchen photos
            
        Returns:
            List of background images
        """
        backgrounds = []
        w, h = base_size
        
        # Generate synthetic backgrounds
        synthetic_backgrounds = [
            # Clean countertop colors
            np.full((h, w, 3), [245, 245, 240], dtype=np.uint8),  # Off-white
            np.full((h, w, 3), [220, 220, 215], dtype=np.uint8),  # Light gray
            np.full((h, w, 3), [200, 180, 160], dtype=np.uint8),  # Beige
            np.full((h, w, 3), [180, 160, 140], dtype=np.uint8),  # Wood tone
            
            # Gradient backgrounds
            self._create_gradient_background(base_size, [240, 240, 235], [200, 200, 195]),
            self._create_gradient_background(base_size, [220, 200, 180], [180, 160, 140]),
        ]
        
        backgrounds.extend(synthetic_backgrounds)
        
        # Add real kitchen photos if provided
        if kitchen_photos_path and os.path.exists(kitchen_photos_path):
            kitchen_photos = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                kitchen_photos.extend(Path(kitchen_photos_path).glob(ext))
            
            for photo_path in kitchen_photos[:10]:  # Limit to 10 kitchen photos
                try:
                    img = cv2.imread(str(photo_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, base_size)
                    backgrounds.append(img)
                except Exception as e:
                    print(f"⚠️  Could not load kitchen photo {photo_path}: {e}")
        
        print(f"📸 Created {len(backgrounds)} background variants")
        return backgrounds
    
    def _create_gradient_background(self, size: Tuple[int, int], 
                                  color1: List[int], color2: List[int]) -> np.ndarray:
        """Create a gradient background."""
        w, h = size
        background = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(h):
            ratio = i / h
            blended_color = [
                int(color1[j] * (1 - ratio) + color2[j] * ratio)
                for j in range(3)
            ]
            background[i, :] = blended_color
        
        return background
    
    def calculate_placement_positions(self, objects: List[ObjectCrop], 
                                    scene_size: Tuple[int, int]) -> List[Dict]:
        """
        Calculate realistic placement positions for objects.
        
        Args:
            objects: List of objects to place
            scene_size: Size of the scene (width, height)
            
        Returns:
            List of placement dictionaries with position and scale info
        """
        w, h = scene_size
        placements = []
        constraints = PlacementConstraints()
        
        # Sort objects by placement hierarchy (large items first)
        sorted_objects = self._sort_by_placement_priority(objects)
        
        occupied_areas = []  # Track occupied areas to avoid overlap
        
        for obj in sorted_objects:
            # Determine scale based on object size group
            scale = self._calculate_object_scale(obj, constraints)
            
            # Calculate object dimensions after scaling
            obj_h, obj_w = obj.image.shape[:2]
            scaled_w = int(obj_w * scale)
            scaled_h = int(obj_h * scale)
            
            # Find valid placement position
            position = self._find_valid_position(
                scaled_w, scaled_h, scene_size, occupied_areas, constraints
            )
            
            if position is None:
                print(f"⚠️  Could not place {obj.class_name}, skipping")
                continue
            
            x, y = position
            
            # Add some random variation to position
            x += random.randint(-10, 10)
            y += random.randint(-10, 10)
            
            # Ensure bounds
            x = max(0, min(x, w - scaled_w))
            y = max(0, min(y, h - scaled_h))
            
            placement = {
                'object': obj,
                'x': x,
                'y': y,
                'scale': scale,
                'width': scaled_w,
                'height': scaled_h
            }
            
            placements.append(placement)
            
            # Mark area as occupied
            occupied_areas.append({
                'x1': x, 'y1': y,
                'x2': x + scaled_w, 'y2': y + scaled_h
            })
        
        return placements
    
    def _sort_by_placement_priority(self, objects: List[ObjectCrop]) -> List[ObjectCrop]:
        """Sort objects by placement priority (background items first)."""
        def get_priority(obj):
            class_name = obj.class_name.lower()
            if class_name in self.placement_rules['placement_hierarchy']['background']:
                return 0  # Highest priority (placed first)
            elif class_name in self.placement_rules['size_groups']['large']:
                return 1
            elif class_name in self.placement_rules['size_groups']['medium']:
                return 2
            else:
                return 3  # Lowest priority (placed last)
        
        return sorted(objects, key=get_priority)
    
    def _calculate_object_scale(self, obj: ObjectCrop, constraints: PlacementConstraints) -> float:
        """Calculate appropriate scale for object based on its characteristics."""
        class_name = obj.class_name.lower()
        
        # Base scale based on size group
        if class_name in self.placement_rules['size_groups']['large']:
            base_scale = random.uniform(0.6, 1.0)
        elif class_name in self.placement_rules['size_groups']['medium']:
            base_scale = random.uniform(0.4, 0.8)
        else:  # small
            base_scale = random.uniform(0.3, 0.6)
        
        # Add random variation
        scale_variation = random.uniform(0.9, 1.1)
        final_scale = base_scale * scale_variation
        
        # Constrain to limits
        return max(constraints.min_scale, min(constraints.max_scale, final_scale))
    
    def _find_valid_position(self, obj_w: int, obj_h: int, scene_size: Tuple[int, int],
                           occupied_areas: List[Dict], constraints: PlacementConstraints) -> Optional[Tuple[int, int]]:
        """Find a valid position for an object that doesn't overlap too much."""
        w, h = scene_size
        margin_w = int(w * constraints.avoid_edge_margin)
        margin_h = int(h * constraints.avoid_edge_margin)
        
        max_attempts = 50
        for _ in range(max_attempts):
            x = random.randint(margin_w, max(margin_w, w - obj_w - margin_w))
            y = random.randint(margin_h, max(margin_h, h - obj_h - margin_h))
            
            # Check overlap with existing objects
            new_area = {'x1': x, 'y1': y, 'x2': x + obj_w, 'y2': y + obj_h}
            
            valid = True
            for occupied in occupied_areas:
                overlap_ratio = self._calculate_overlap_ratio(new_area, occupied)
                if overlap_ratio > constraints.max_overlap_ratio:
                    valid = False
                    break
            
            if valid:
                return (x, y)
        
        return None
    
    def _calculate_overlap_ratio(self, area1: Dict, area2: Dict) -> float:
        """Calculate overlap ratio between two rectangular areas."""
        # Calculate intersection
        x1 = max(area1['x1'], area2['x1'])
        y1 = max(area1['y1'], area2['y1'])
        x2 = min(area1['x2'], area2['x2'])
        y2 = min(area1['y2'], area2['y2'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0  # No overlap
        
        intersection_area = (x2 - x1) * (y2 - y1)
        area1_size = (area1['x2'] - area1['x1']) * (area1['y2'] - area1['y1'])
        area2_size = (area2['x2'] - area2['x1']) * (area2['y2'] - area2['y1'])
        
        union_area = area1_size + area2_size - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def compose_multi_object_scene(self, objects: List[ObjectCrop], 
                                 background: np.ndarray) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Compose a multi-object scene from individual object crops.
        
        Args:
            objects: List of object crops to compose
            background: Background image
            
        Returns:
            Tuple of (composed_image, yolo_annotations)
        """
        scene = background.copy()
        h, w = scene.shape[:2]
        
        # Calculate placement positions
        placements = self.calculate_placement_positions(objects, (w, h))
        
        yolo_annotations = []
        
        for placement in placements:
            obj = placement['object']
            x, y = placement['x'], placement['y']
            scale = placement['scale']
            
            # Resize object
            obj_resized = cv2.resize(obj.image, None, fx=scale, fy=scale)
            obj_h, obj_w = obj_resized.shape[:2]
            
            # Ensure object fits in scene
            if x + obj_w > w or y + obj_h > h:
                continue
            
            # Create alpha blending for realistic placement
            mask = self._create_object_mask(obj_resized)
            
            # Blend object onto scene
            for c in range(3):
                scene[y:y+obj_h, x:x+obj_w, c] = (
                    mask * obj_resized[:, :, c] + 
                    (1 - mask) * scene[y:y+obj_h, x:x+obj_w, c]
                )
            
            # Create YOLO annotation
            x_center = (x + obj_w / 2) / w
            y_center = (y + obj_h / 2) / h
            width_norm = obj_w / w
            height_norm = obj_h / h
            
            yolo_annotation = [obj.class_id, x_center, y_center, width_norm, height_norm]
            yolo_annotations.append(yolo_annotation)
        
        return scene, yolo_annotations
    
    def _create_object_mask(self, obj_image: np.ndarray) -> np.ndarray:
        """Create a soft mask for realistic object blending."""
        # Simple approach: create soft edges
        h, w = obj_image.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        # Apply soft edges
        edge_width = min(5, min(h, w) // 10)
        if edge_width > 0:
            mask[:edge_width, :] *= np.linspace(0.3, 1.0, edge_width).reshape(-1, 1)
            mask[-edge_width:, :] *= np.linspace(1.0, 0.3, edge_width).reshape(-1, 1)
            mask[:, :edge_width] *= np.linspace(0.3, 1.0, edge_width)
            mask[:, -edge_width:] *= np.linspace(1.0, 0.3, edge_width)
        
        return mask[:, :, np.newaxis]
    
    def generate_multi_object_dataset(self, num_scenes: int = 1000, 
                                    objects_per_scene: Tuple[int, int] = (2, 5),
                                    kitchen_photos_path: Optional[str] = None,
                                    split_ratios: Dict[str, float] = None):
        """
        Generate complete multi-object dataset.
        
        Args:
            num_scenes: Total number of scenes to generate
            objects_per_scene: Range of objects per scene (min, max)
            kitchen_photos_path: Optional path to kitchen photos for backgrounds
            split_ratios: Dataset split ratios
        """
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'valid': 0.2, 'test': 0.1}
        
        print(f"\n🎨 Generating {num_scenes} multi-object scenes...")
        print(f"   📦 Objects per scene: {objects_per_scene[0]}-{objects_per_scene[1]}")
        
        # Extract object crops from training data
        crops_by_class = self.extract_object_crops('train', max_crops_per_class=150)
        
        if not crops_by_class:
            print("❌ No object crops extracted. Check your dataset path.")
            return
        
        # Create backgrounds
        backgrounds = self.create_background_variants(kitchen_photos_path=kitchen_photos_path)
        
        # Generate scenes
        scene_count = 0
        split_counts = {split: 0 for split in split_ratios.keys()}
        
        for i in range(num_scenes):
            try:
                # Determine split for this scene
                rand_val = random.random()
                cumulative = 0
                current_split = 'train'
                for split, ratio in split_ratios.items():
                    cumulative += ratio
                    if rand_val <= cumulative:
                        current_split = split
                        break
                
                # Select random objects for this scene
                num_objects = random.randint(*objects_per_scene)
                selected_objects = self._select_objects_for_scene(crops_by_class, num_objects)
                
                if len(selected_objects) < 2:  # Ensure multi-object
                    continue
                
                # Select random background
                background = random.choice(backgrounds).copy()
                
                # Compose scene
                scene, annotations = self.compose_multi_object_scene(selected_objects, background)
                
                if len(annotations) < 2:  # Ensure multi-object annotations
                    continue
                
                # Apply augmentation
                augmented = self._apply_augmentation(scene, annotations)
                if augmented is None:
                    continue
                
                scene_aug, annotations_aug = augmented
                
                # Save scene and annotations
                scene_filename = f"multi_scene_{scene_count:06d}.jpg"
                self._save_scene(scene_aug, annotations_aug, scene_filename, current_split)
                
                split_counts[current_split] += 1
                scene_count += 1
                
                if scene_count % 100 == 0:
                    print(f"   ✅ Generated {scene_count}/{num_scenes} scenes")
                
            except Exception as e:
                print(f"⚠️  Error generating scene {i}: {e}")
                continue
        
        # Create data.yaml for the multi-object dataset
        self._create_dataset_config(split_counts)
        
        print(f"\n🎉 Multi-object dataset generation complete!")
        print(f"   📊 Total scenes: {scene_count}")
        for split, count in split_counts.items():
            print(f"   {split}: {count} scenes")
        print(f"   📁 Output: {self.output_path / 'multi_object'}")
    
    def _select_objects_for_scene(self, crops_by_class: Dict[str, List[ObjectCrop]], 
                                num_objects: int) -> List[ObjectCrop]:
        """Select objects for a scene with realistic combinations."""
        available_classes = [cls for cls, crops in crops_by_class.items() if crops]
        
        if len(available_classes) < 2:
            return []
        
        selected_objects = []
        
        # Try to create realistic groupings
        used_classes = set()
        
        for _ in range(num_objects):
            # Select class (avoid too many of the same class)
            available_for_selection = [
                cls for cls in available_classes 
                if cls not in used_classes or len(used_classes) < len(available_classes) // 2
            ]
            
            if not available_for_selection:
                break
            
            class_name = random.choice(available_for_selection)
            crop = random.choice(crops_by_class[class_name])
            selected_objects.append(crop)
            used_classes.add(class_name)
        
        return selected_objects
    
    def _apply_augmentation(self, scene: np.ndarray, 
                          annotations: List[List[float]]) -> Optional[Tuple[np.ndarray, List[List[float]]]]:
        """Apply albumentations augmentation to scene."""
        try:
            # Convert annotations to required format
            bboxes = []
            class_labels = []
            
            for ann in annotations:
                class_id, x_center, y_center, width, height = ann
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(int(class_id))
            
            # Apply augmentation
            augmented = self.augmentation_pipeline(
                image=scene,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            # Convert back to YOLO format
            aug_annotations = []
            for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                x_center, y_center, width, height = bbox
                aug_annotations.append([class_id, x_center, y_center, width, height])
            
            return augmented['image'], aug_annotations
            
        except Exception as e:
            print(f"⚠️  Augmentation failed: {e}")
            return None
    
    def _save_scene(self, scene: np.ndarray, annotations: List[List[float]], 
                   filename: str, split: str):
        """Save scene image and YOLO annotations."""
        # Save image
        image_path = self.output_path / 'multi_object' / split / 'images' / filename
        scene_bgr = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), scene_bgr)
        
        # Save annotations
        label_path = self.output_path / 'multi_object' / split / 'labels' / f"{Path(filename).stem}.txt"
        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id, x_center, y_center, width, height = ann
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def _create_dataset_config(self, split_counts: Dict[str, int]):
        """Create data.yaml configuration for the multi-object dataset."""
        config = {
            'train': str(self.output_path / 'multi_object' / 'train' / 'images'),
            'val': str(self.output_path / 'multi_object' / 'valid' / 'images'),
            'test': str(self.output_path / 'multi_object' / 'test' / 'images'),
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        config_path = self.output_path / 'multi_object' / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📋 Dataset configuration saved: {config_path}")


def main():
    """Example usage of the MultiObjectComposer."""
    # Configuration
    YOLO_DATASET_PATH = "image_classification/utensils-wp5hm-yolo8"
    OUTPUT_PATH = "image_classification/multi_object_data"
    KITCHEN_PHOTOS_PATH = None  # Set to your kitchen photos directory if available
    
    # Load class names from existing data.yaml
    with open(f"{YOLO_DATASET_PATH}/data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config['names']
    
    # Initialize composer
    composer = MultiObjectComposer(
        yolo_dataset_path=YOLO_DATASET_PATH,
        output_path=OUTPUT_PATH,
        class_names=class_names
    )
    
    # Generate multi-object dataset
    composer.generate_multi_object_dataset(
        num_scenes=1000,  # Adjust based on your needs
        objects_per_scene=(2, 5),
        kitchen_photos_path=KITCHEN_PHOTOS_PATH,
        split_ratios={'train': 0.7, 'valid': 0.2, 'test': 0.1}
    )


if __name__ == "__main__":
    main() 