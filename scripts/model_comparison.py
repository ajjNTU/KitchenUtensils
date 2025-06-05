"""
Model Comparison: MobileNetV3 vs ResNet50V2 Performance Analysis
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_model_results():
    """Analyze and compare model performance results."""
    
    print("=" * 80)
    print("MODEL COMPARISON: MobileNetV3 vs ResNet50V2")
    print("=" * 80)
    
    # Previous MobileNetV3 results (from memory bank)
    mobilenet_results = {
        "architecture": "MobileNetV3Large",
        "baseline_accuracy": 0.12,  # ~12% from memory bank
        "training_epochs": 15,
        "parameters": "~4.2M (estimated)",
        "description": "Lightweight mobile-optimized architecture"
    }
    
    # ResNet50V2 results (just obtained)
    resnet_results = {
        "architecture": "ResNet50V2", 
        "test_accuracy": 0.9448,  # 94.48%
        "validation_accuracy": 0.9299,  # 92.99%
        "training_accuracy": 0.8915,  # 89.15%
        "training_epochs": 2,
        "total_parameters": 23829781,
        "trainable_parameters": 264981,
        "non_trainable_parameters": 23564800,
        "training_time": 211.14,  # seconds
        "description": "Deep residual network with skip connections"
    }
    
    print("\n📊 ARCHITECTURE COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<25} {'MobileNetV3':<15} {'ResNet50V2':<15}")
    print("-" * 50)
    print(f"{'Total Parameters':<25} {mobilenet_results['parameters']:<15} {resnet_results['total_parameters']:,}")
    print(f"{'Trainable Params':<25} {'~200K (est.)':<15} {resnet_results['trainable_parameters']:,}")
    print(f"{'Model Size':<25} {'Small':<15} {'Large':<15}")
    print(f"{'Mobile Optimized':<25} {'Yes':<15} {'No':<15}")
    
    print("\n🎯 ACCURACY COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<25} {'MobileNetV3':<15} {'ResNet50V2':<15}")
    print("-" * 50)
    print(f"{'Test Accuracy':<25} {mobilenet_results['baseline_accuracy']*100:.2f}%{'':<10} {resnet_results['test_accuracy']*100:.2f}%")
    print(f"{'Validation Accuracy':<25} {'N/A':<15} {resnet_results['validation_accuracy']*100:.2f}%")
    print(f"{'Training Epochs':<25} {mobilenet_results['training_epochs']:<15} {resnet_results['training_epochs']}")
    
    # Calculate improvements
    accuracy_improvement = (resnet_results['test_accuracy'] - mobilenet_results['baseline_accuracy']) * 100
    
    print("\n🚀 PERFORMANCE IMPROVEMENTS")
    print("-" * 50)
    print(f"Accuracy Improvement: +{accuracy_improvement:.2f} percentage points")
    print(f"Relative Improvement: {(resnet_results['test_accuracy'] / mobilenet_results['baseline_accuracy'] - 1) * 100:.1f}x better")
    print(f"Training Efficiency: {resnet_results['test_accuracy']*100:.1f}% accuracy in just {resnet_results['training_epochs']} epochs")
    
    print("\n⏱️ TRAINING PERFORMANCE")
    print("-" * 50)
    print(f"ResNet50V2 Training Time: {resnet_results['training_time']:.1f} seconds ({resnet_results['training_time']/60:.1f} minutes)")
    print(f"Epochs Required: {resnet_results['training_epochs']} (vs {mobilenet_results['training_epochs']} for MobileNetV3)")
    print(f"Time per Epoch: {resnet_results['training_time']/resnet_results['training_epochs']:.1f} seconds")
    
    print("\n📈 DETAILED RESNET50V2 RESULTS")
    print("-" * 50)
    print(f"Final Training Accuracy: {resnet_results['training_accuracy']*100:.2f}%")
    print(f"Final Validation Accuracy: {resnet_results['validation_accuracy']*100:.2f}%")
    print(f"Final Test Accuracy: {resnet_results['test_accuracy']*100:.2f}%")
    print(f"Model Architecture: Transfer Learning (ImageNet pretrained)")
    print(f"Frozen Base Layers: {resnet_results['non_trainable_parameters']:,} parameters")
    print(f"Custom Classification Head: {resnet_results['trainable_parameters']:,} parameters")
    
    print("\n🎯 CONCLUSIONS")
    print("-" * 50)
    print("✅ ResNet50V2 significantly outperforms MobileNetV3")
    print(f"✅ Achieved {resnet_results['test_accuracy']*100:.1f}% test accuracy vs {mobilenet_results['baseline_accuracy']*100:.1f}% baseline")
    print(f"✅ Fast convergence: {resnet_results['test_accuracy']*100:.1f}% accuracy in just 2 epochs")
    print("✅ Strong generalization: validation and test accuracy are very close")
    print("✅ Transfer learning effectiveness: ResNet50V2 base features work well for utensils")
    
    print("\n💡 RECOMMENDATIONS")
    print("-" * 50)
    print("🔹 Continue with ResNet50V2 for production model")
    print("🔹 Consider fine-tuning for even better accuracy")
    print("🔹 ResNet50V2 is suitable for the kitchen utensils classification task")
    print("🔹 Current accuracy (94.48%) is excellent for the 21-class problem")
    
    print("\n" + "=" * 80)
    
    return {
        "mobilenet_results": mobilenet_results,
        "resnet_results": resnet_results,
        "improvement": accuracy_improvement
    }

if __name__ == "__main__":
    analyze_model_results() 