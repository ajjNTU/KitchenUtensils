#!/usr/bin/env python3
"""
Combine Single-Object and Multi-Object Datasets

Combines your original single-object YOLO dataset with the new multi-object dataset
for comprehensive training that improves multi-object detection capabilities.
"""

import os
import shutil
import yaml
from pathlib import Path
import argparse

class DatasetCombiner:
    def __init__(self, single_object_path: str, multi_object_path: str, output_path: str):
        """Initialize the dataset combiner."""
        self.single_path = Path(single_object_path)
        self.multi_path = Path(multi_object_path)
        self.output_path = Path(output_path)
        
        print(f"🔗 Dataset Combiner initialized")
        print(f"   📁 Single-object source: {self.single_path}")
        print(f"   📁 Multi-object source: {self.multi_path}")
        print(f"   📁 Combined output: {self.output_path}")
        
        # Load class names from original dataset
        with open(self.single_path / 'data.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        self.class_names = self.config['names']
        
        self._setup_output_dirs()
    
    def _setup_output_dirs(self):
        """Create output directory structure."""
        for split in ['train', 'valid', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def combine_datasets(self, multi_object_ratio: float = 0.3):
        """
        Combine datasets with specified ratio of multi-object scenes.
        
        Args:
            multi_object_ratio: Ratio of multi-object scenes in final dataset (0.0-1.0)
        """
        print(f"\n🎯 Combining datasets with {multi_object_ratio:.1%} multi-object scenes...")
        
        total_files = {'train': 0, 'valid': 0, 'test': 0}
        
        for split in ['train', 'valid', 'test']:
            print(f"\n📦 Processing {split} split...")
            
            # Get file counts
            single_images = list((self.single_path / split / 'images').glob('*.jpg'))
            multi_images = list((self.multi_path / split / 'images').glob('*.jpg'))
            
            print(f"   📊 Available: {len(single_images)} single-object, {len(multi_images)} multi-object")
            
            # Calculate how many files to use from each dataset
            total_multi = len(multi_images)
            if multi_object_ratio > 0 and total_multi > 0:
                # Use all available multi-object images
                target_single = int(total_multi * (1 - multi_object_ratio) / multi_object_ratio)
                target_single = min(target_single, len(single_images))
                target_multi = total_multi
            else:
                # No multi-object images requested
                target_single = len(single_images)
                target_multi = 0
            
            print(f"   🎯 Using: {target_single} single-object, {target_multi} multi-object")
            
            # Copy single-object files
            single_count = 0
            for i, img_path in enumerate(single_images[:target_single]):
                if self._copy_file_pair(img_path, split, f"single_{i:06d}", 'single'):
                    single_count += 1
            
            # Copy multi-object files
            multi_count = 0
            for i, img_path in enumerate(multi_images[:target_multi]):
                if self._copy_file_pair(img_path, split, f"multi_{i:06d}", 'multi'):
                    multi_count += 1
            
            total_files[split] = single_count + multi_count
            print(f"   ✅ Copied: {single_count} single + {multi_count} multi = {total_files[split]} total")
        
        # Create combined config
        self._create_combined_config()
        
        print(f"\n🎉 Dataset combination complete!")
        print(f"   📊 Total files:")
        for split, count in total_files.items():
            print(f"     {split}: {count} images")
        print(f"   📁 Combined dataset: {self.output_path}")
        print(f"   🎯 Multi-object ratio: {multi_object_ratio:.1%}")
    
    def _copy_file_pair(self, img_path: Path, split: str, new_name: str, source_type: str) -> bool:
        """Copy image and corresponding label file."""
        try:
            # Determine source paths
            if source_type == 'single':
                base_path = self.single_path
            else:
                base_path = self.multi_path
            
            # Source paths
            src_img = img_path
            src_label = base_path / split / 'labels' / f"{img_path.stem}.txt"
            
            # Destination paths
            dst_img = self.output_path / split / 'images' / f"{new_name}.jpg"
            dst_label = self.output_path / split / 'labels' / f"{new_name}.txt"
            
            # Copy files
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            else:
                return False
            
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                # Create empty label file for images without annotations
                dst_label.touch()
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error copying {img_path}: {e}")
            return False
    
    def _create_combined_config(self):
        """Create data.yaml for the combined dataset."""
        config = {
            'train': str(self.output_path / 'train' / 'images'),
            'val': str(self.output_path / 'valid' / 'images'), 
            'test': str(self.output_path / 'test' / 'images'),
            'nc': len(self.class_names),
            'names': self.class_names,
            'description': 'Combined single-object and multi-object kitchen utensils dataset'
        }
        
        config_path = self.output_path / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📋 Combined config saved: {config_path}")
    
    def analyze_dataset(self):
        """Analyze the combined dataset composition."""
        print(f"\n📊 Dataset Analysis:")
        print("=" * 50)
        
        for split in ['train', 'valid', 'test']:
            images_dir = self.output_path / split / 'images'
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob('*.jpg'))
            single_count = len([f for f in image_files if f.stem.startswith('single_')])
            multi_count = len([f for f in image_files if f.stem.startswith('multi_')])
            total = len(image_files)
            
            if total > 0:
                multi_ratio = multi_count / total
                print(f"\n{split.upper()} Split:")
                print(f"  📸 Total images: {total}")
                print(f"  🔷 Single-object: {single_count} ({single_count/total:.1%})")
                print(f"  🔶 Multi-object: {multi_count} ({multi_ratio:.1%})")
            else:
                print(f"\n{split.upper()} Split: No images found")


def main():
    parser = argparse.ArgumentParser(description='Combine single-object and multi-object YOLO datasets')
    parser.add_argument('--single', default='image_classification/utensils-wp5hm-yolo8',
                       help='Path to single-object dataset')
    parser.add_argument('--multi', default='image_classification/multi_object_enhanced',
                       help='Path to multi-object dataset')
    parser.add_argument('--output', default='image_classification/combined_dataset',
                       help='Output path for combined dataset')
    parser.add_argument('--ratio', type=float, default=0.3,
                       help='Ratio of multi-object scenes (0.0-1.0)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze the dataset after combination')
    
    args = parser.parse_args()
    
    # Validate ratio
    if not 0.0 <= args.ratio <= 1.0:
        print("❌ Error: ratio must be between 0.0 and 1.0")
        return
    
    combiner = DatasetCombiner(args.single, args.multi, args.output)
    combiner.combine_datasets(multi_object_ratio=args.ratio)
    
    if args.analyze:
        combiner.analyze_dataset()

if __name__ == "__main__":
    main() 