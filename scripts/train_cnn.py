"""
Improved CNN training script with fine-tuning for better accuracy.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_classification.cnn_classifier import CNNClassifier
import tensorflow as tf

def train_improved_cnn():
    """Train CNN with fine-tuning for better accuracy."""
    print("Training improved CNN classifier with fine-tuning...")
    
    # Paths
    train_dir = os.path.join("image_classification", "cls_data", "train")
    test_dir = os.path.join("image_classification", "cls_data", "test")
    model_path = os.path.join("image_classification", "cnn_model_finetuned.h5")
    
    # Check if data exists
    if not os.path.exists(train_dir):
        print(f"Error: Training data not found at {train_dir}")
        return None
    
    # Create classifier
    classifier = CNNClassifier()
    
    print("Creating model...")
    model = classifier.create_model()
    
    print("Phase 1: Training classification head (15 epochs)...")
    history1 = classifier.train(
        data_dir=train_dir,
        epochs=15,
        batch_size=32,
        validation_split=0.2
    )
    
    print("Phase 2: Fine-tuning with unfrozen layers (10 epochs)...")
    # Unfreeze the base model for fine-tuning
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training with fine-tuning
    history2 = classifier.train(
        data_dir=train_dir,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )
    
    # Evaluate on test set if available
    if os.path.exists(test_dir):
        print("Evaluating on test set...")
        test_results = classifier.evaluate(test_dir)
        print(f"Final Test Results: {test_results}")
    
    # Save model
    classifier.save_model(model_path)
    
    print("Improved training complete!")
    return classifier

if __name__ == "__main__":
    train_improved_cnn() 