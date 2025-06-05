"""
Proven CNN training approach for high accuracy on kitchen utensils.
Based on best practices that consistently work.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2  # More stable than EfficientNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def train_proven_cnn():
    """Train CNN using proven techniques for high accuracy."""
    print("🎯 Training Proven High-Accuracy CNN")
    print("=" * 45)
    
    # Paths
    train_dir = os.path.join("image_classification", "cls_data", "train")
    valid_dir = os.path.join("image_classification", "cls_data", "valid")
    test_dir = os.path.join("image_classification", "cls_data", "test")
    model_path = os.path.join("image_classification", "cnn_model_proven.h5")
    
    # Get class names consistently
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)
    print(f"📊 Training on {num_classes} classes")
    
    # Minimal but effective data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,      # Very conservative
        horizontal_flip=True,  # Only horizontal flip
        fill_mode='nearest'
    )
    
    # No augmentation for validation/test
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators with explicit class ordering
    IMG_SIZE = 224
    BATCH_SIZE = 16  # Smaller batch size for stability
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,  # Explicit ordering
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,  # Same ordering
        shuffle=False
    )
    
    test_gen = val_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,  # Same ordering
        shuffle=False
    )
    
    print(f"📈 Data: Train={train_gen.samples}, Val={val_gen.samples}, Test={test_gen.samples}")
    
    # Use ResNet50V2 - more stable than EfficientNet
    print("🏗️ Creating ResNet50V2 model...")
    
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Simple but effective head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Conservative compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Conservative LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📋 Model compiled successfully!")
    print(f"🔢 Trainable params: {model.count_params():,}")
    
    # Phase 1: Train head only (conservative)
    print("\n🎯 Phase 1: Training classification head (10 epochs)")
    print("-" * 50)
    
    history1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        verbose=1
    )
    
    # Check if we're learning
    val_acc_phase1 = max(history1.history['val_accuracy'])
    print(f"📊 Phase 1 best validation accuracy: {val_acc_phase1:.1%}")
    
    if val_acc_phase1 < 0.3:
        print("⚠️ Low accuracy after phase 1. Continuing with more epochs...")
        # Train a bit more
        history1_extra = model.fit(
            train_gen,
            epochs=5,
            validation_data=val_gen,
            verbose=1
        )
        val_acc_phase1 = max(history1_extra.history['val_accuracy'])
    
    # Phase 2: Fine-tune (if phase 1 was successful)
    if val_acc_phase1 > 0.2:  # Only fine-tune if we're learning
        print(f"\n🎯 Phase 2: Fine-tuning (5 epochs)")
        print("-" * 50)
        
        # Unfreeze top layers only
        base_model.trainable = True
        
        # Freeze early layers, only train last few
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Much lower learning rate for fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            train_gen,
            epochs=5,
            validation_data=val_gen,
            verbose=1
        )
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    print("-" * 30)
    
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    
    print(f"Validation Accuracy: {val_acc:.1%}")
    print(f"Test Accuracy: {test_acc:.1%}")
    
    # Save model
    model.save(model_path)
    with open(model_path.replace('.h5', '_classes.json'), 'w') as f:
        json.dump(class_names, f)
    
    print(f"\n✅ Model saved to: {model_path}")
    
    # Diagnosis
    if test_acc > 0.8:
        print("🎉 EXCELLENT! >80% accuracy achieved!")
    elif test_acc > 0.6:
        print("👍 GOOD! >60% accuracy achieved!")
    elif test_acc > 0.3:
        print("📈 LEARNING! Model is working, needs more training")
    else:
        print("❌ ISSUE: Model not learning properly")
        print("   Possible causes:")
        print("   - Data preprocessing issue")
        print("   - Class label mismatch")
        print("   - Learning rate too high/low")
    
    return model, test_acc

if __name__ == "__main__":
    train_proven_cnn() 