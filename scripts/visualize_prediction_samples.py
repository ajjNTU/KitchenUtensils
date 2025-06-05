"""
Prediction Samples Visualization Script
Visualizes sample predictions with actual images, showing both correct and incorrect predictions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_classification.cnn_classifier import CNNClassifier

class PredictionVisualizer:
    def __init__(self, model_path, data_dir):
        """Initialize prediction visualizer."""
        self.classifier = CNNClassifier()
        self.classifier.load_model(model_path)
        self.data_dir = data_dir
        self.test_dir = os.path.join(data_dir, 'test')
        self.class_names = self.classifier.class_names
        
    def get_sample_predictions(self, num_samples=20):
        """Get sample predictions from test set."""
        # Create test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.classifier.img_size,
            batch_size=1,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        # Get all predictions
        predictions = self.classifier.model.predict(test_generator, verbose=1)
        filenames = test_generator.filenames
        true_labels = test_generator.classes
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Separate correct and incorrect predictions
        correct_indices = []
        incorrect_indices = []
        
        for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
            if pred == true:
                correct_indices.append(i)
            else:
                incorrect_indices.append(i)
        
        # Sample from each category
        num_correct = min(num_samples // 2, len(correct_indices))
        num_incorrect = min(num_samples // 2, len(incorrect_indices))
        
        selected_correct = random.sample(correct_indices, num_correct) if correct_indices else []
        selected_incorrect = random.sample(incorrect_indices, num_incorrect) if incorrect_indices else []
        
        return {
            'correct': [(i, filenames[i], true_labels[i], predicted_labels[i], predictions[i]) 
                       for i in selected_correct],
            'incorrect': [(i, filenames[i], true_labels[i], predicted_labels[i], predictions[i]) 
                         for i in selected_incorrect],
            'all_filenames': filenames,
            'all_predictions': predictions,
            'all_true_labels': true_labels,
            'all_predicted_labels': predicted_labels
        }
    
    def visualize_prediction_samples(self, samples, save_path='prediction_samples.png'):
        """Visualize sample predictions with images."""
        correct_samples = samples['correct']
        incorrect_samples = samples['incorrect']
        
        # Calculate grid size
        total_samples = len(correct_samples) + len(incorrect_samples)
        cols = 5
        rows = (total_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        sample_idx = 0
        
        # Plot correct predictions
        for idx, filename, true_label, pred_label, pred_probs in correct_samples:
            if sample_idx >= total_samples:
                break
                
            row = sample_idx // cols
            col = sample_idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            
            # Load and display image
            img_path = os.path.join(self.test_dir, filename)
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            
            # Add title with prediction info
            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]
            confidence = pred_probs[pred_label]
            
            ax.set_title(f'✓ CORRECT\nTrue: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}', 
                        fontsize=10, color='green', fontweight='bold')
            ax.axis('off')
            sample_idx += 1
        
        # Plot incorrect predictions
        for idx, filename, true_label, pred_label, pred_probs in incorrect_samples:
            if sample_idx >= total_samples:
                break
                
            row = sample_idx // cols
            col = sample_idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            
            # Load and display image
            img_path = os.path.join(self.test_dir, filename)
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            
            # Add title with prediction info
            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]
            confidence = pred_probs[pred_label]
            true_confidence = pred_probs[true_label]
            
            ax.set_title(f'✗ INCORRECT\nTrue: {true_class} ({true_confidence:.3f})\nPred: {pred_class} ({confidence:.3f})', 
                        fontsize=10, color='red', fontweight='bold')
            ax.axis('off')
            sample_idx += 1
        
        # Hide unused subplots
        for i in range(sample_idx, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row][col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.suptitle('Sample Predictions: Kitchen Utensils CNN Model', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_worst_predictions(self, samples, num_worst=10, save_path='worst_predictions.png'):
        """Visualize the worst (most confident incorrect) predictions."""
        incorrect_samples = samples['incorrect']
        
        if not incorrect_samples:
            print("No incorrect predictions found!")
            return
        
        # Sort by confidence (highest confidence mistakes are worst)
        sorted_incorrect = sorted(incorrect_samples, 
                                key=lambda x: x[4][x[3]], reverse=True)
        
        # Take worst predictions
        worst_samples = sorted_incorrect[:num_worst]
        
        # Create visualization
        cols = 3
        rows = (len(worst_samples) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (idx, filename, true_label, pred_label, pred_probs) in enumerate(worst_samples):
            row = i // cols
            col = i % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            
            # Load and display image
            img_path = os.path.join(self.test_dir, filename)
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            
            # Add detailed title
            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]
            confidence = pred_probs[pred_label]
            true_confidence = pred_probs[true_label]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(pred_probs)[-3:][::-1]
            top_3_text = []
            for j, idx_pred in enumerate(top_3_indices):
                marker = "→" if idx_pred == pred_label else " "
                top_3_text.append(f"{marker}{self.class_names[idx_pred]}: {pred_probs[idx_pred]:.3f}")
            
            title = f'WORST MISTAKE #{i+1}\n'
            title += f'True: {true_class} ({true_confidence:.3f})\n'
            title += f'File: {os.path.basename(filename)}\n'
            title += 'Top 3:\n' + '\n'.join(top_3_text)
            
            ax.set_title(title, fontsize=9, color='darkred', fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(worst_samples), rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row][col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.suptitle('Worst Predictions (Most Confident Mistakes)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return worst_samples
    
    def create_class_confusion_samples(self, samples, save_path='class_confusion_samples.png'):
        """Show sample images for most confused class pairs."""
        all_true = samples['all_true_labels']
        all_pred = samples['all_predicted_labels']
        all_files = samples['all_filenames']
        all_probs = samples['all_predictions']
        
        # Find most confused class pairs
        confusion_pairs = {}
        for true, pred, filename, probs in zip(all_true, all_pred, all_files, all_probs):
            if true != pred:
                pair = (self.class_names[true], self.class_names[pred])
                if pair not in confusion_pairs:
                    confusion_pairs[pair] = []
                confusion_pairs[pair].append((filename, probs[pred], probs[true]))
        
        # Sort by frequency and get top pairs
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: len(x[1]), reverse=True)
        top_pairs = sorted_pairs[:6]  # Top 6 confused pairs
        
        if not top_pairs:
            print("No confusion pairs found!")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, ((true_class, pred_class), examples) in enumerate(top_pairs):
            if i >= 6:
                break
                
            ax = axes[i]
            
            # Take first example (you could also take the most confident mistake)
            filename, pred_conf, true_conf = examples[0]
            
            # Load and display image
            img_path = os.path.join(self.test_dir, filename)
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            
            title = f'Confusion: {true_class} → {pred_class}\n'
            title += f'Frequency: {len(examples)} times\n'
            title += f'True conf: {true_conf:.3f} | Pred conf: {pred_conf:.3f}\n'
            title += f'File: {os.path.basename(filename)}'
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(top_pairs), 6):
            axes[i].axis('off')
        
        plt.suptitle('Most Common Class Confusions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return confusion_pairs

def main():
    """Main visualization function."""
    # Paths
    model_path = 'image_classification/cnn_model.h5'
    data_dir = 'image_classification/cls_data'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first or update the model path.")
        return
    
    # Create visualizer
    visualizer = PredictionVisualizer(model_path, data_dir)
    
    # Get sample predictions
    print("Analyzing predictions...")
    samples = visualizer.get_sample_predictions(num_samples=20)
    
    print(f"Found {len(samples['correct'])} correct and {len(samples['incorrect'])} incorrect samples")
    
    # Create visualizations
    print("\nCreating prediction samples visualization...")
    visualizer.visualize_prediction_samples(samples, 'results/prediction_samples.png')
    
    print("\nCreating worst predictions visualization...")
    worst_samples = visualizer.visualize_worst_predictions(samples, num_worst=9, 
                                                          save_path='results/worst_predictions.png')
    
    print("\nCreating class confusion samples...")
    confusion_pairs = visualizer.create_class_confusion_samples(samples, 
                                                               save_path='results/class_confusion_samples.png')
    
    # Print summary of failed filenames
    if samples['incorrect']:
        print("\nFAILED PREDICTION FILENAMES:")
        print("=" * 40)
        for i, (idx, filename, true_label, pred_label, pred_probs) in enumerate(samples['incorrect'], 1):
            true_class = visualizer.class_names[true_label]
            pred_class = visualizer.class_names[pred_label]
            confidence = pred_probs[pred_label]
            print(f"{i:2d}. {filename}")
            print(f"    True: {true_class} | Predicted: {pred_class} | Confidence: {confidence:.3f}")
    
    print(f"\nVisualization complete! Check the 'results' directory for image outputs.")

if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Set random seed for reproducible sampling
    random.seed(42)
    np.random.seed(42)
    
    main() 