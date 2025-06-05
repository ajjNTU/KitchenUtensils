"""
Analyze the kitchen utensils dataset for parameter optimization.
"""

import os

def analyze_dataset():
    """Analyze dataset distribution and characteristics."""
    
    print("🔍 Kitchen Utensils Dataset Analysis")
    print("=" * 50)
    
    # Paths
    train_dir = os.path.join("image_classification", "cls_data", "train")
    test_dir = os.path.join("image_classification", "cls_data", "test")
    valid_dir = os.path.join("image_classification", "cls_data", "valid")
    
    datasets = [
        ("Training", train_dir),
        ("Test", test_dir),
        ("Validation", valid_dir)
    ]
    
    total_images = 0
    class_distribution = {}
    
    for dataset_name, dataset_dir in datasets:
        if not os.path.exists(dataset_dir):
            print(f"❌ {dataset_name} directory not found: {dataset_dir}")
            continue
            
        print(f"\n📁 {dataset_name} Dataset: {dataset_dir}")
        print("-" * 30)
        
        classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        dataset_total = 0
        
        for cls in sorted(classes):
            cls_path = os.path.join(dataset_dir, cls)
            if os.path.isdir(cls_path):
                images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                count = len(images)
                dataset_total += count
                
                if cls not in class_distribution:
                    class_distribution[cls] = {"train": 0, "test": 0, "valid": 0}
                
                if dataset_name.lower() in class_distribution[cls]:
                    class_distribution[cls][dataset_name.lower()] = count
                
                print(f"  {cls:<15}: {count:4d} images")
        
        print(f"\n  Total {dataset_name} Images: {dataset_total}")
        total_images += dataset_total
    
    print(f"\n📊 DATASET SUMMARY")
    print("=" * 50)
    print(f"Total Classes: {len(class_distribution)}")
    print(f"Total Images: {total_images}")
    
    # Class balance analysis
    print(f"\n📈 CLASS DISTRIBUTION")
    print("-" * 50)
    print(f"{'Class':<15} {'Train':<8} {'Test':<8} {'Valid':<8} {'Total':<8}")
    print("-" * 50)
    
    min_images = float('inf')
    max_images = 0
    
    for cls in sorted(class_distribution.keys()):
        dist = class_distribution[cls]
        total_cls = sum(dist.values())
        min_images = min(min_images, total_cls)
        max_images = max(max_images, total_cls)
        
        print(f"{cls:<15} {dist.get('train', 0):<8} {dist.get('test', 0):<8} {dist.get('valid', 0):<8} {total_cls:<8}")
    
    print("-" * 50)
    
    # Balance analysis
    balance_ratio = min_images / max_images if max_images > 0 else 0
    
    print(f"\n🎯 BALANCE ANALYSIS")
    print("-" * 30)
    print(f"Min images per class: {min_images}")
    print(f"Max images per class: {max_images}")
    print(f"Balance ratio: {balance_ratio:.3f}")
    
    if balance_ratio > 0.8:
        print("✅ Dataset is well balanced")
    elif balance_ratio > 0.5:
        print("⚠️  Dataset has moderate imbalance")
    else:
        print("❌ Dataset has significant imbalance")
    
    # Memory estimation
    print(f"\n💾 MEMORY ESTIMATION")
    print("-" * 30)
    
    batch_sizes = [16, 32, 64, 128]
    input_sizes = [(224, 224), (299, 299), (331, 331)]
    
    for input_size in input_sizes:
        print(f"\nInput Size: {input_size[0]}x{input_size[1]}")
        for batch_size in batch_sizes:
            # Rough memory estimation (MB)
            # Formula: batch_size * height * width * channels * 4 bytes * 2 (forward + backward)
            memory_mb = batch_size * input_size[0] * input_size[1] * 3 * 4 * 2 / (1024 * 1024)
            memory_mb += 100  # Add base model memory (~100MB for ResNet50V2)
            
            status = "✅" if memory_mb < 4000 else "⚠️" if memory_mb < 8000 else "❌"
            print(f"  Batch {batch_size:3d}: {memory_mb:6.1f} MB {status}")
    
    print(f"\n🎯 PARAMETER RECOMMENDATIONS")
    print("-" * 40)
    
    # Based on dataset size
    train_images = sum(dist.get('train', 0) for dist in class_distribution.values())
    
    if train_images < 1000:
        print("📉 Small dataset detected:")
        print("  - Use higher dropout (0.3-0.5)")
        print("  - Use smaller batch size (16-32)")
        print("  - Use more aggressive data augmentation")
        print("  - Consider early stopping")
    elif train_images < 5000:
        print("📊 Medium dataset detected:")
        print("  - Standard parameters should work well")
        print("  - Batch size 32-64 recommended")
        print("  - Moderate dropout (0.2-0.3)")
    else:
        print("📈 Large dataset detected:")
        print("  - Can use larger batch sizes (64-128)")
        print("  - Lower dropout may be sufficient")
        print("  - More epochs may be beneficial")
    
    return {
        "total_images": total_images,
        "num_classes": len(class_distribution),
        "train_images": train_images,
        "balance_ratio": balance_ratio,
        "class_distribution": class_distribution
    }

if __name__ == "__main__":
    analyze_dataset() 