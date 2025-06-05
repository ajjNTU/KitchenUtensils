"""
Quick 3-epoch test to verify CNN training approach is working.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def quick_test_cnn():
    """Quick test to verify CNN training approach."""
    print("🧪 Quick CNN Test (3 epochs)")
    print("=" * 35)
    
    # Paths
    train_dir = os.path.join("image_classification", "cls_data", "train")
    valid_dir = os.path.join("image_classification", "cls_data", "valid")
    
    # Get class names
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    print(f"📊 Testing on {num_classes} classes")
    
    # Simple data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    IMG_SIZE = 224
    BATCH_SIZE = 16
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=False
    )
    
    print(f"📈 Data: Train={train_gen.samples}, Val={val_gen.samples}")
    
    # Create simple model
    print("🏗️ Creating test model...")
    
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📋 Model ready!")
    
    # Quick training test
    print("\n🧪 Testing training (3 epochs)...")
    print("-" * 40)
    
    history = model.fit(
        train_gen,
        epochs=3,
        validation_data=val_gen,
        verbose=1
    )
    
    # Analyze results
    print("\n📊 Test Results:")
    print("-" * 25)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"Final Training Accuracy: {final_train_acc:.1%}")
    print(f"Final Validation Accuracy: {final_val_acc:.1%}")
    
    # Check if learning is happening
    acc_improvement = history.history['accuracy'][-1] - history.history['accuracy'][0]
    val_acc_improvement = history.history['val_accuracy'][-1] - history.history['val_accuracy'][0]
    
    print(f"Training Accuracy Improvement: {acc_improvement:.1%}")
    print(f"Validation Accuracy Improvement: {val_acc_improvement:.1%}")
    
    # Diagnosis
    print("\n🔍 Diagnosis:")
    print("-" * 15)
    
    if final_val_acc > 0.5:
        print("🎉 EXCELLENT! >50% accuracy in just 3 epochs!")
        print("   ✅ This approach will definitely work well")
        print("   ✅ Full training should achieve 80-90%+")
    elif final_val_acc > 0.3:
        print("👍 GOOD! >30% accuracy in 3 epochs")
        print("   ✅ Model is learning properly")
        print("   ✅ Full training should achieve 60-80%")
    elif final_val_acc > 0.15:
        print("📈 LEARNING! Better than random (4.8%)")
        print("   ✅ Model is working")
        print("   ⚠️ May need more epochs or tuning")
    elif acc_improvement > 0.1:
        print("📊 TRAINING! Training accuracy improving")
        print("   ✅ Model can learn")
        print("   ⚠️ May have overfitting or validation issues")
    else:
        print("❌ ISSUE! Model not learning")
        print("   ❌ Need to investigate data or approach")
    
    # Random baseline
    random_acc = 1.0 / num_classes
    print(f"\n📌 Random baseline: {random_acc:.1%}")
    
    if final_val_acc > random_acc * 2:
        print("✅ Significantly better than random!")
    elif final_val_acc > random_acc * 1.5:
        print("✅ Better than random")
    else:
        print("⚠️ Close to random - investigate!")
    
    return model, final_val_acc

if __name__ == "__main__":
    quick_test_cnn() 