"""
Improved CNN training script for high accuracy (80-90%+) on kitchen utensils dataset.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import json

def create_improved_model(num_classes=21, img_size=(224, 224)):
    """Create an improved CNN model with better architecture."""
    
    # Use EfficientNetB0 instead of MobileNetV3 for better accuracy
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Improved classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def get_class_weights(train_generator):
    """Calculate class weights to handle imbalance."""
    # Get class counts from the generator
    class_counts = {}
    labels = train_generator.labels
    
    for class_idx in range(len(train_generator.class_indices)):
        class_counts[class_idx] = np.sum(labels == class_idx)
    
    # Calculate weights
    total_samples = sum(class_counts.values())
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (len(class_counts) * count)
    
    return class_weights

def train_improved_cnn():
    """Train improved CNN with proper techniques for high accuracy."""
    print("🚀 Training Improved CNN for High Accuracy")
    print("=" * 50)
    
    # Paths
    train_dir = os.path.join("image_classification", "cls_data", "train")
    valid_dir = os.path.join("image_classification", "cls_data", "valid")
    test_dir = os.path.join("image_classification", "cls_data", "test")
    model_path = os.path.join("image_classification", "cnn_model_improved.h5")
    
    if not os.path.exists(train_dir):
        print(f"❌ Training data not found: {train_dir}")
        return None
    
    # Class names (sorted to ensure consistency)
    class_names = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    print(f"📊 Found {len(class_names)} classes: {class_names}")
    
    # Improved data generators with better augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,          # Reduced from 20
        width_shift_range=0.1,      # Reduced from 0.2
        height_shift_range=0.1,     # Reduced from 0.2
        shear_range=0.1,           # Added
        zoom_range=0.1,            # Reduced from 0.2
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    batch_size = 32
    img_size = (224, 224)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_names,
        shuffle=True
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_names,
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_names,
        shuffle=False
    )
    
    print(f"📈 Training samples: {train_generator.samples}")
    print(f"📈 Validation samples: {valid_generator.samples}")
    print(f"📈 Test samples: {test_generator.samples}")
    
    # Calculate class weights
    class_weights = get_class_weights(train_generator)
    print(f"⚖️ Using class weights to handle imbalance")
    
    # Create improved model
    print("🏗️ Creating improved model with EfficientNetB0...")
    model, base_model = create_improved_model(len(class_names), img_size)
    
    # Compile with appropriate learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    print(f"📋 Model summary:")
    model.summary()
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_path.replace('.h5', '_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Phase 1: Train classification head
    print("\n🎯 Phase 1: Training classification head (20 epochs)")
    print("-" * 50)
    
    history1 = model.fit(
        train_generator,
        epochs=20,
        validation_data=valid_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen layers
    print("\n🎯 Phase 2: Fine-tuning with unfrozen layers (15 epochs)")
    print("-" * 50)
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Use a much lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    # Reset callbacks with different patience
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=2,
            min_lr=1e-8,
            verbose=1
        ),
        ModelCheckpoint(
            model_path.replace('.h5', '_finetuned.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history2 = model.fit(
        train_generator,
        epochs=15,
        validation_data=valid_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    print("-" * 30)
    
    # Evaluate on validation set
    val_results = model.evaluate(valid_generator, verbose=0)
    print(f"Validation - Loss: {val_results[0]:.4f}, Accuracy: {val_results[1]:.4f}, Top-3: {val_results[2]:.4f}")
    
    # Evaluate on test set
    test_results = model.evaluate(test_generator, verbose=0)
    print(f"Test - Loss: {test_results[0]:.4f}, Accuracy: {test_results[1]:.4f}, Top-3: {test_results[2]:.4f}")
    
    # Save final model
    model.save(model_path)
    
    # Save class names
    with open(model_path.replace('.h5', '_classes.json'), 'w') as f:
        json.dump(class_names, f)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {model_path}")
    print(f"🎯 Final test accuracy: {test_results[1]:.1%}")
    
    if test_results[1] > 0.8:
        print("🎉 Excellent! Achieved >80% accuracy!")
    elif test_results[1] > 0.6:
        print("👍 Good! Achieved >60% accuracy!")
    else:
        print("⚠️ Accuracy could be improved. Consider more data or different architecture.")
    
    return model

if __name__ == "__main__":
    train_improved_cnn() 