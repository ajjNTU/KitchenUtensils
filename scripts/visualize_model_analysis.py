"""
Comprehensive Model Analysis and Visualization Script
Analyzes CNN model performance and creates visualizations of predictions and errors.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_classification.cnn_classifier import CNNClassifier

class ModelAnalyzer:
    def __init__(self, model_path, data_dir):
        """Initialize model analyzer."""
        self.classifier = CNNClassifier()
        self.classifier.load_model(model_path)
        self.data_dir = data_dir
        self.test_dir = os.path.join(data_dir, 'test')
        self.class_names = self.classifier.class_names
        
        # Results storage
        self.predictions = []
        self.true_labels = []
        self.filenames = []
        self.prediction_probs = []
        self.failed_predictions = []
        
    def analyze_test_set(self):
        """Analyze entire test set and collect detailed predictions."""
        print("Analyzing test set predictions...")
        
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
        
        # Get predictions for all test images
        predictions = self.classifier.model.predict(test_generator, verbose=1)
        
        # Extract filenames and true labels
        filenames = test_generator.filenames
        true_labels = test_generator.classes
        
        # Process predictions
        predicted_labels = np.argmax(predictions, axis=1)
        max_probs = np.max(predictions, axis=1)
        
        # Store results
        self.predictions = predicted_labels
        self.true_labels = true_labels
        self.filenames = filenames
        self.prediction_probs = predictions
        
        # Identify failed predictions
        for i, (pred, true, filename, prob) in enumerate(zip(predicted_labels, true_labels, filenames, max_probs)):
            if pred != true:
                self.failed_predictions.append({
                    'filename': filename,
                    'true_class': self.class_names[true],
                    'predicted_class': self.class_names[pred],
                    'confidence': prob,
                    'true_class_prob': predictions[i][true],
                    'full_predictions': predictions[i]
                })
        
        print(f"Analysis complete. Found {len(self.failed_predictions)} failed predictions out of {len(self.predictions)} total.")
        
    def create_confusion_matrix_plot(self, save_path='confusion_matrix.png'):
        """Create and save confusion matrix visualization."""
        plt.figure(figsize=(16, 14))
        
        # Create confusion matrix
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        # Create heatmap
        sns.heatmap(cm, 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   square=True)
        
        plt.title('Confusion Matrix - Kitchen Utensils CNN Model', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
        
    def create_error_analysis_plot(self, save_path='error_analysis.png'):
        """Create detailed error analysis visualization."""
        if not self.failed_predictions:
            print("No failed predictions to analyze!")
            return
            
        # Create subplots for error analysis
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Error distribution by true class
        true_classes = [fp['true_class'] for fp in self.failed_predictions]
        predicted_classes = [fp['predicted_class'] for fp in self.failed_predictions]
        
        ax1 = axes[0, 0]
        true_class_counts = pd.Series(true_classes).value_counts()
        true_class_counts.plot(kind='bar', ax=ax1, color='lightcoral')
        ax1.set_title('Failed Predictions by True Class', fontweight='bold')
        ax1.set_xlabel('True Class')
        ax1.set_ylabel('Number of Errors')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Error distribution by predicted class
        ax2 = axes[0, 1]
        pred_class_counts = pd.Series(predicted_classes).value_counts()
        pred_class_counts.plot(kind='bar', ax=ax2, color='lightblue')
        ax2.set_title('Failed Predictions by Predicted Class', fontweight='bold')
        ax2.set_xlabel('Predicted Class')
        ax2.set_ylabel('Number of Errors')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Confidence distribution of failed predictions
        ax3 = axes[1, 0]
        confidences = [fp['confidence'] for fp in self.failed_predictions]
        ax3.hist(confidences, bins=20, color='orange', alpha=0.7)
        ax3.set_title('Confidence Distribution of Failed Predictions', fontweight='bold')
        ax3.set_xlabel('Prediction Confidence')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax3.legend()
        
        # 4. True class probability for failed predictions
        ax4 = axes[1, 1]
        true_probs = [fp['true_class_prob'] for fp in self.failed_predictions]
        ax4.hist(true_probs, bins=20, color='lightgreen', alpha=0.7)
        ax4.set_title('True Class Probability in Failed Predictions', fontweight='bold')
        ax4.set_xlabel('True Class Probability')
        ax4.set_ylabel('Frequency')
        ax4.axvline(np.mean(true_probs), color='red', linestyle='--',
                   label=f'Mean: {np.mean(true_probs):.3f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_failed_predictions_report(self, save_path='failed_predictions_report.txt'):
        """Save detailed report of failed predictions with filenames."""
        with open(save_path, 'w') as f:
            f.write("FAILED PREDICTIONS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total test images: {len(self.predictions)}\n")
            f.write(f"Failed predictions: {len(self.failed_predictions)}\n")
            f.write(f"Accuracy: {(len(self.predictions) - len(self.failed_predictions)) / len(self.predictions) * 100:.2f}%\n\n")
            
            f.write("DETAILED FAILED PREDICTIONS:\n")
            f.write("-" * 30 + "\n\n")
            
            for i, fp in enumerate(self.failed_predictions, 1):
                f.write(f"{i}. FAILED PREDICTION\n")
                f.write(f"   Filename: {fp['filename']}\n")
                f.write(f"   True Class: {fp['true_class']}\n")
                f.write(f"   Predicted Class: {fp['predicted_class']}\n")
                f.write(f"   Prediction Confidence: {fp['confidence']:.4f}\n")
                f.write(f"   True Class Probability: {fp['true_class_prob']:.4f}\n")
                
                # Top 3 predictions for this image
                top_indices = np.argsort(fp['full_predictions'])[-3:][::-1]
                f.write(f"   Top 3 Predictions:\n")
                for j, idx in enumerate(top_indices, 1):
                    f.write(f"      {j}. {self.class_names[idx]}: {fp['full_predictions'][idx]:.4f}\n")
                f.write("\n")
        
        print(f"Failed predictions report saved to: {save_path}")
        
    def create_class_performance_plot(self, save_path='class_performance.png'):
        """Create per-class performance visualization."""
        # Calculate per-class metrics
        class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            # True positives, false positives, false negatives
            tp = np.sum((self.true_labels == i) & (self.predictions == i))
            fp = np.sum((self.true_labels != i) & (self.predictions == i))
            fn = np.sum((self.true_labels == i) & (self.predictions != i))
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'total_samples': np.sum(self.true_labels == i)
            }
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        classes = list(class_metrics.keys())
        precisions = [class_metrics[c]['precision'] for c in classes]
        recalls = [class_metrics[c]['recall'] for c in classes]
        f1_scores = [class_metrics[c]['f1_score'] for c in classes]
        sample_counts = [class_metrics[c]['total_samples'] for c in classes]
        
        # Precision plot
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(classes)), precisions, color='skyblue')
        ax1.set_title('Precision by Class', fontweight='bold')
        ax1.set_ylabel('Precision')
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.set_ylim(0, 1.1)
        
        # Recall plot
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(classes)), recalls, color='lightgreen')
        ax2.set_title('Recall by Class', fontweight='bold')
        ax2.set_ylabel('Recall')
        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.set_ylim(0, 1.1)
        
        # F1-score plot
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(classes)), f1_scores, color='orange')
        ax3.set_title('F1-Score by Class', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_xticks(range(len(classes)))
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.set_ylim(0, 1.1)
        
        # Sample count plot
        ax4 = axes[1, 1]
        bars4 = ax4.bar(range(len(classes)), sample_counts, color='coral')
        ax4.set_title('Test Samples by Class', fontweight='bold')
        ax4.set_ylabel('Number of Samples')
        ax4.set_xticks(range(len(classes)))
        ax4.set_xticklabels(classes, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return class_metrics
        
    def print_failed_filenames(self):
        """Print all failed prediction filenames to console."""
        print("\nFAILED PREDICTION FILENAMES:")
        print("=" * 40)
        
        for i, fp in enumerate(self.failed_predictions, 1):
            print(f"{i:2d}. {fp['filename']}")
            print(f"    True: {fp['true_class']} | Predicted: {fp['predicted_class']} | Confidence: {fp['confidence']:.3f}")
        
        print(f"\nTotal failed files: {len(self.failed_predictions)}")

def main():
    """Main analysis function."""
    # Paths
    model_path = 'image_classification/cnn_model.h5'
    data_dir = 'image_classification/cls_data'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first or update the model path.")
        return
    
    # Create analyzer
    analyzer = ModelAnalyzer(model_path, data_dir)
    
    # Run analysis
    analyzer.analyze_test_set()
    
    # Create visualizations
    print("\nCreating confusion matrix...")
    cm = analyzer.create_confusion_matrix_plot('results/confusion_matrix.png')
    
    print("\nCreating error analysis...")
    analyzer.create_error_analysis_plot('results/error_analysis.png')
    
    print("\nCreating class performance analysis...")
    class_metrics = analyzer.create_class_performance_plot('results/class_performance.png')
    
    # Save reports
    print("\nSaving failed predictions report...")
    analyzer.save_failed_predictions_report('results/failed_predictions_report.txt')
    
    # Print failed filenames to console
    analyzer.print_failed_filenames()
    
    print(f"\nAnalysis complete! Check the 'results' directory for visualizations and reports.")

if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)
    main() 