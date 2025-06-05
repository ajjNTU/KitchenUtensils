"""
Test script to train ResNet50V2 CNN classifier for 2 epochs and evaluate accuracy.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_classification.cnn_classifier import CNNClassifier
import tensorflow as tf
import time

def test_resnet50v2_accuracy():
    """Test ResNet50V2 model accuracy with 2 epochs training."""
    print("=" * 60)
    print("Testing ResNet50V2 CNN Classifier - 2 Epochs")
    print("=" * 60)
    
    # Paths
    train_dir = os.path.join("image_classification", "cls_data", "train")
    test_dir = os.path.join("image_classification", "cls_data", "test")
    model_path = os.path.join("image_classification", "resnet50v2_test_model.h5")
    
    # Check if data exists
    if not os.path.exists(train_dir):
        print(f"Error: Training data not found at {train_dir}")
        print("Please run dataset preparation scripts first.")
        return None
    
    # Create classifier
    print("Initializing ResNet50V2 classifier...")
    classifier = CNNClassifier()
    
    print("Creating ResNet50V2 model...")
    model = classifier.create_model()
    
    # Print model summary
    print("\nModel Architecture Summary:")
    print(f"Base Model: ResNet50V2")
    print(f"Input Shape: {classifier.img_size + (3,)}")
    print(f"Number of Classes: {len(classifier.class_names)}")
    print(f"Total Parameters: {model.count_params():,}")
    
    # Count trainable vs non-trainable parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    
    print("\n" + "=" * 40)
    print("Starting 2-Epoch Training")
    print("=" * 40)
    
    start_time = time.time()
    
    # Train for exactly 2 epochs
    history = classifier.train(
        data_dir=train_dir,
        epochs=2,  # Test with 2 epochs as requested
        batch_size=32,
        validation_split=0.2
    )
    
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Extract training metrics
    if history and hasattr(history, 'history'):
        final_train_acc = history.history['accuracy'][-1] if 'accuracy' in history.history else 0
        final_val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
        final_train_loss = history.history['loss'][-1] if 'loss' in history.history else 0
        final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 0
        
        print("\n" + "=" * 40)
        print("Training Results (2 Epochs)")
        print("=" * 40)
        print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    # Evaluate on test set if available
    if os.path.exists(test_dir):
        print("\n" + "=" * 40)
        print("Test Set Evaluation")
        print("=" * 40)
        test_results = classifier.evaluate(test_dir)
        test_accuracy = test_results.get('accuracy', 0)
        test_loss = test_results.get('loss', 0)
        
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
    else:
        print(f"\nWarning: Test data not found at {test_dir}")
        test_accuracy = None
    
    # Save model
    print(f"\nSaving model to {model_path}...")
    classifier.save_model(model_path)
    
    print("\n" + "=" * 60)
    print("ResNet50V2 Test Summary")
    print("=" * 60)
    print(f"Architecture: ResNet50V2 (Transfer Learning)")
    print(f"Training Epochs: 2")
    print(f"Training Time: {training_time:.2f} seconds")
    if history and hasattr(history, 'history'):
        print(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")
    if test_accuracy is not None:
        print(f"Test Set Accuracy: {test_accuracy*100:.2f}%")
    print(f"Model saved: {model_path}")
    print("=" * 60)
    
    return classifier

if __name__ == "__main__":
    test_resnet50v2_accuracy() 