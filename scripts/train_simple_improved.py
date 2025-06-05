"""
Simple but improved CNN training script for better accuracy.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def train_simple_improved():
    """Train a simple but improved CNN model."""
    print("🚀 Training Simple Improved CNN")
    print("=" * 40)
    
    # Paths
    train_dir = os.path.join("image_classification", "cls_data", "train")
    valid_dir = os.path.join("image_classification", "cls_data", "valid")
    test_dir = os.path.join("image_classification", "cls_data", "test")
    model_path = os.path.join("image_classification", "cnn_model_simple_improved.h5")
    
    # Class names
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    print(f"📊 Classes: {len(class_names)}")
    
    # Better data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=class_names
    )
    
    val_gen = val_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=class_names
    )
    
    test_gen = val_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=class_names,
        shuffle=False
    )
    
    print(f"📈 Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}")
    
    # Create model with EfficientNetB0
    print("🏗️ Creating model...")
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📋 Model created successfully!")
    
    # Phase 1: Train head
    print("\n🎯 Phase 1: Training classification head...")
    history1 = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        verbose=1
    )
    
    # Phase 2: Fine-tune
    print("\n🎯 Phase 2: Fine-tuning...")
    base_model.trainable = True
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Final Evaluation:")
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"Test Accuracy: {test_acc:.1%}")
    
    # Save model
    model.save(model_path)
    with open(model_path.replace('.h5', '_classes.json'), 'w') as f:
        json.dump(class_names, f)
    
    print(f"✅ Model saved to: {model_path}")
    
    if test_acc > 0.8:
        print("🎉 Excellent! >80% accuracy achieved!")
    elif test_acc > 0.6:
        print("👍 Good! >60% accuracy achieved!")
    else:
        print("⚠️ Accuracy could be better, but model is functional")
    
    return model

if __name__ == "__main__":
    train_simple_improved() 