"""
Optimized CNN Image Classifier for Kitchen Utensils using ResNet50V2.
Parameters specifically tuned based on dataset analysis and ResNet50V2 characteristics.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json

class OptimizedCNNClassifier:
    def __init__(self, model_path=None, class_names=None, input_size=(299, 299)):
        """
        Initialize optimized CNN classifier for ResNet50V2.
        
        Args:
            model_path: Path to saved model file
            class_names: List of class names in order
            input_size: Input image size (optimized for ResNet50V2)
        """
        self.model = None
        self.class_names = class_names or [
            "Blender", "Bowl", "Canopener", "Choppingboard", "Colander", "Cup", 
            "Dinnerfork", "Dinnerknife", "Fishslice", "Garlicpress", "Kitchenknife", 
            "Ladle", "Pan", "Peeler", "Saucepan", "Spoon", "Teaspoon", "Tongs", 
            "Tray", "Whisk", "Woodenspoon"
        ]
        # Optimized input size for ResNet50V2 (better than 224x224)
        self.img_size = input_size
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def create_model(self, num_classes=21, dropout_rate=0.4, dense_units=256, learning_rate=0.0003):
        """
        Create optimized CNN model using ResNet50V2 transfer learning.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization (increased for small dataset)
            dense_units: Number of units in dense layer (increased for ResNet50V2)
            learning_rate: Learning rate (reduced for transfer learning)
        """
        print(f"🏗️  Creating optimized ResNet50V2 model:")
        print(f"   📐 Input size: {self.img_size}")
        print(f"   🎯 Classes: {num_classes}")
        print(f"   🧠 Dense units: {dense_units}")
        print(f"   💧 Dropout rate: {dropout_rate}")
        print(f"   📈 Learning rate: {learning_rate}")
        
        # Load pre-trained ResNet50V2 with optimized input size
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Optimized classification head for kitchen utensils
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),  # Added for better convergence
            layers.Dropout(dropout_rate),  # Increased dropout for small dataset
            layers.Dense(dense_units, activation='relu'),  # Larger dense layer
            layers.BatchNormalization(),  # Additional batch norm
            layers.Dropout(dropout_rate * 0.8),  # Slightly lower dropout in second layer
            layers.Dense(num_classes, activation='softmax')
        ], name='optimized_resnet50v2_classifier')
        
        # Compile with optimized parameters
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']  # Fixed - removed problematic top_3_accuracy
        )
        
        self.model = model
        
        # Print model summary
        print(f"\n📊 Model Summary:")
        print(f"   Total parameters: {model.count_params():,}")
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        print(f"   Trainable: {trainable_params:,}")
        print(f"   Non-trainable: {non_trainable_params:,}")
        
        return model
    
    def train(self, data_dir, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the optimized CNN model.
        
        Args:
            data_dir: Path to training data directory
            epochs: Maximum number of training epochs
            batch_size: Training batch size (optimized for dataset size)
            validation_split: Fraction of data to use for validation
        """
        if self.model is None:
            self.create_model(len(self.class_names))
        
        print(f"\n🚀 Starting optimized training:")
        print(f"   📊 Dataset: {data_dir}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Max epochs: {epochs}")
        print(f"   ✂️  Validation split: {validation_split}")
        
        # Enhanced data augmentation for small dataset
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            # Geometric augmentations
            rotation_range=25,           # Increased from 20
            width_shift_range=0.25,      # Increased from 0.2
            height_shift_range=0.25,     # Increased from 0.2
            horizontal_flip=True,
            zoom_range=0.3,              # Increased from 0.2
            shear_range=0.15,            # Added shear transformation
            # Photometric augmentations
            brightness_range=[0.8, 1.2], # Added brightness variation
            channel_shift_range=20.0,    # Added channel shifting
            fill_mode='reflect',         # Better fill mode
            validation_split=validation_split
        )
        
        # Validation data (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            classes=self.class_names,
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            classes=self.class_names,
            shuffle=False
        )
        
        print(f"   📈 Training samples: {train_generator.samples}")
        print(f"   📊 Validation samples: {validation_generator.samples}")
        
        # Setup optimized callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1,
                cooldown=3
            )
        ]
        
        # Train model
        print(f"\n⏱️  Training started...")
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✅ Training completed!")
        return history
    
    def evaluate(self, test_dir, batch_size=32):
        """Evaluate model on test data with detailed metrics."""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        print(f"\n📊 Evaluating model on test set...")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        print(f"   📁 Test samples: {test_generator.samples}")
        
        results = self.model.evaluate(test_generator, verbose=1)
        result_dict = dict(zip(self.model.metrics_names, results))
        
        print(f"\n📈 Test Results:")
        for metric, value in result_dict.items():
            if 'accuracy' in metric:
                print(f"   {metric}: {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"   {metric}: {value:.4f}")
        
        return result_dict
    
    def predict(self, image_path, top_k=5):
        """Enhanced prediction with confidence analysis."""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            class_name = self.class_names[idx]
            confidence = float(predictions[idx])
            results.append((class_name, confidence))
        
        return results
    
    def save_model(self, model_path):
        """Save trained model with metadata."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'class_names': self.class_names,
            'input_size': self.img_size,
            'model_type': 'optimized_resnet50v2',
            'num_classes': len(self.class_names)
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Metadata saved to {metadata_path}")
    
    def load_model(self, model_path):
        """Load trained model with metadata."""
        self.model = keras.models.load_model(model_path)
        
        # Load metadata if available
        metadata_path = model_path.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.class_names = metadata.get('class_names', self.class_names)
                self.img_size = tuple(metadata.get('input_size', self.img_size))
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   📐 Input size: {self.img_size}")
        print(f"   🎯 Classes: {len(self.class_names)}")


def train_optimized_cnn():
    """Train optimized CNN model on kitchen utensils dataset."""
    print("🚀 Training Optimized ResNet50V2 CNN Classifier")
    print("=" * 60)
    
    # Paths
    train_dir = os.path.join("image_classification", "cls_data", "train")
    test_dir = os.path.join("image_classification", "cls_data", "test")
    model_path = os.path.join("image_classification", "optimized_resnet50v2_model.h5")
    
    # Check if data exists
    if not os.path.exists(train_dir):
        print(f"❌ Error: Training data not found at {train_dir}")
        return None
    
    # Create optimized classifier with larger input size
    classifier = OptimizedCNNClassifier(input_size=(299, 299))
    
    print("\n🏗️  Creating optimized model...")
    classifier.create_model(
        num_classes=21,
        dropout_rate=0.4,     # Higher dropout for small dataset
        dense_units=256,      # Larger dense layer for ResNet50V2
        learning_rate=0.0003  # Lower learning rate for transfer learning
    )
    
    print("\n🚀 Starting optimized training...")
    history = classifier.train(
        data_dir=train_dir,
        epochs=50,            # More epochs with early stopping
        batch_size=32,        # Optimal batch size for dataset
        validation_split=0.2
    )
    
    # Save model
    classifier.save_model(model_path)
    
    # Evaluate on test set
    if os.path.exists(test_dir):
        print("\n📊 Evaluating on test set...")
        test_results = classifier.evaluate(test_dir)
    
    print("\n✅ Optimized training complete!")
    return classifier


if __name__ == "__main__":
    train_optimized_cnn() 