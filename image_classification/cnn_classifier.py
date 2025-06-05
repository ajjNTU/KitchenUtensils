"""
CNN Image Classifier for Kitchen Utensils using Transfer Learning.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

class CNNClassifier:
    def __init__(self, model_path=None, class_names=None):
        """
        Initialize CNN classifier.
        
        Args:
            model_path: Path to saved model file
            class_names: List of class names in order
        """
        self.model = None
        self.class_names = class_names or [
            "Blender", "Bowl", "Canopener", "Choppingboard", "Colander", "Cup", 
            "Dinnerfork", "Dinnerknife", "Fishslice", "Garlicpress", "Kitchenknife", 
            "Ladle", "Pan", "Peeler", "Saucepan", "Spoon", "Teaspoon", "Tongs", 
            "Tray", "Whisk", "Woodenspoon"
        ]
        self.img_size = (224, 224)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def create_model(self, num_classes=21):
        """Create CNN model using ResNet50V2 transfer learning."""
        # Load pre-trained ResNet50V2
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head with optimized parameters
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),  # Increased for small dataset
            layers.Dense(256, activation='relu'),  # Larger for ResNet50V2
            layers.Dropout(0.3),  # Increased for small dataset
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model with optimized parameters
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),  # Reduced for transfer learning
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, data_dir, epochs=10, batch_size=32, validation_split=0.2):
        """
        Train the CNN model.
        
        Args:
            data_dir: Path to training data directory
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
        """
        if self.model is None:
            self.create_model(len(self.class_names))
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=validation_split
        )
        
        # Validation data (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            classes=self.class_names
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            classes=self.class_names
        )
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_dir, batch_size=32):
        """Evaluate model on test data."""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        results = self.model.evaluate(test_generator, verbose=1)
        return dict(zip(self.model.metrics_names, results))
    
    def predict(self, image_path, top_k=3):
        """
        Predict utensil class from image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, confidence) tuples
        """
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
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(model_path)
        
        # Save class names
        class_names_path = model_path.replace('.h5', '_classes.json')
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f)
        
        print(f"Model saved to {model_path}")
        print(f"Class names saved to {class_names_path}")
    
    def load_model(self, model_path):
        """Load trained model."""
        self.model = keras.models.load_model(model_path)
        
        # Load class names if available
        class_names_path = model_path.replace('.h5', '_classes.json')
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        
        print(f"Model loaded from {model_path}")


def train_cnn_model():
    """Train CNN model on kitchen utensils dataset."""
    print("Training CNN classifier for kitchen utensils...")
    
    # Paths
    train_dir = os.path.join("image_classification", "cls_data", "train")
    test_dir = os.path.join("image_classification", "cls_data", "test")
    model_path = os.path.join("image_classification", "cnn_model.h5")
    
    # Check if data exists
    if not os.path.exists(train_dir):
        print(f"Error: Training data not found at {train_dir}")
        return None
    
    # Create and train classifier
    classifier = CNNClassifier()
    
    print("Creating model...")
    classifier.create_model()
    
    print("Starting training...")
    history = classifier.train(
        data_dir=train_dir,
        epochs=15,  # Reasonable for transfer learning
        batch_size=32,
        validation_split=0.2
    )
    
    # Evaluate on test set if available
    if os.path.exists(test_dir):
        print("Evaluating on test set...")
        test_results = classifier.evaluate(test_dir)
        print(f"Test Results: {test_results}")
    
    # Save model
    classifier.save_model(model_path)
    
    print("Training complete!")
    return classifier


if __name__ == "__main__":
    train_cnn_model() 