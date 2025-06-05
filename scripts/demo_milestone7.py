"""
Demo script for Milestone 7: CNN Image Classifier Integration
Kitchen Utensils Chatbot - Vision Capabilities
"""

import os
from main import vision_reply, BotReply

def demo_milestone7():
    """Demonstrate Milestone 7 - CNN Image Classifier integration."""
    print("=" * 60)
    print("🎯 MILESTONE 7 DEMO: CNN Image Classifier Integration")
    print("=" * 60)
    print()
    
    print("✅ COMPLETED FEATURES:")
    print("• CNN classifier using MobileNetV3 transfer learning")
    print("• Image preprocessing and prediction pipeline")
    print("• Integration with chatbot vision_reply function")
    print("• Image input handling in main chatbot loop")
    print("• Error handling for missing images and model failures")
    print("• Confidence-based response formatting")
    print()
    
    # Test with available images from different classes
    test_images = [
        ("Kitchen Knife", "image_classification/cls_data/test/Kitchenknife/GOPR0447_JPG.rf.2cda5c519c424fa93f4d7f0775916f0d_0.jpg"),
        ("Spoon", "image_classification/cls_data/test/Spoon"),
        ("Bowl", "image_classification/cls_data/test/Bowl"),
    ]
    
    for class_name, path in test_images:
        # Find first image in directory if path is a directory
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                image_path = os.path.join(path, files[0])
            else:
                continue
        else:
            image_path = path
        
        if os.path.exists(image_path):
            print(f"🖼️ Testing {class_name} image:")
            print(f"   File: {os.path.basename(image_path)}")
            
            result = vision_reply(image_path)
            if result:
                print(f"   Result: {result.text}")
            else:
                print("   Result: Failed to classify")
            print()
        else:
            print(f"⚠️ {class_name} test image not found: {image_path}")
            print()
    
    print("🔧 USAGE INSTRUCTIONS:")
    print("1. Run: python main.py")
    print("2. Type: image: path/to/your/image.jpg")
    print("3. The chatbot will classify the kitchen utensil in the image")
    print()
    
    print("📊 MODEL PERFORMANCE:")
    print("• Architecture: MobileNetV3Large + Custom Classification Head")
    print("• Training: 15 epochs with data augmentation")
    print("• Classes: 21 kitchen utensil categories")
    print("• Current accuracy: ~12% (baseline model)")
    print("• Note: Accuracy can be improved with fine-tuning (scripts/train_cnn.py)")
    print()
    
    print("🚀 NEXT STEPS (Milestone 8):")
    print("• Implement YOLOv8 object detection")
    print("• Multi-object detection capabilities")
    print("• Compare CNN vs YOLO performance")
    print("• Enhanced image input interface")
    print()
    
    print("✅ MILESTONE 7 COMPLETE!")
    print("CNN Image Classifier successfully integrated into Kitchen Utensils Chatbot")

if __name__ == "__main__":
    demo_milestone7() 