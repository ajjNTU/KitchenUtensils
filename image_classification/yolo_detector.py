"""
YOLOv8 Object Detection for Kitchen Utensils

This module provides YOLOv8-based object detection for kitchen utensils,
including training, inference, and visualization capabilities.
"""

import os
import cv2
import torch
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class YOLODetector:
    """YOLOv8 object detector for kitchen utensils."""
    
    def __init__(self, model_path: Optional[str] = None, data_yaml_path: Optional[str] = None):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to trained YOLO model. If None, uses YOLOv8n pretrained.
            data_yaml_path: Path to data.yaml file for training/class names.
        """
        self.data_yaml_path = data_yaml_path
        self.class_names = self._load_class_names()
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded trained model: {model_path}")
        else:
            # Use pretrained YOLOv8 model for initialization
            self.model = YOLO('yolov8n.pt')  # nano version for faster inference
            print("Loaded YOLOv8n pretrained model")
    
    def _load_class_names(self) -> List[str]:
        """Load class names from data.yaml file."""
        if self.data_yaml_path and os.path.exists(self.data_yaml_path):
            import yaml
            with open(self.data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('names', [])
        
        # Default kitchen utensil classes
        return [
            'Blender', 'Bowl', 'Canopener', 'Choppingboard', 'Colander', 'Cup', 
            'Dinnerfork', 'Dinnerknife', 'Fishslice', 'Garlicpress', 'Kitchenknife', 
            'Ladle', 'Pan', 'Peeler', 'Saucepan', 'Spoon', 'Teaspoon', 'Tongs', 
            'Tray', 'Whisk', 'Woodenspoon'
        ]
    
    def train(self, data_yaml_path: str, epochs: int = 100, img_size: int = 640, 
              batch_size: int = 16, patience: int = 50, save_dir: str = "runs/detect/train"):
        """
        Train YOLOv8 model on kitchen utensils dataset.
        
        Args:
            data_yaml_path: Path to data.yaml configuration file
            epochs: Number of training epochs
            img_size: Input image size
            batch_size: Training batch size
            patience: Early stopping patience
            save_dir: Directory to save training results
        """
        print(f"Starting YOLOv8 training...")
        print(f"Data config: {data_yaml_path}")
        print(f"Epochs: {epochs}, Image size: {img_size}, Batch size: {batch_size}")
        
        # Train the model
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=patience,
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            plots=True,      # Generate training plots
            val=True,        # Validate during training
            project=save_dir.split('/')[0],  # Project name
            name=save_dir.split('/')[-1],    # Run name
            exist_ok=True,   # Overwrite existing runs
            verbose=True,    # Verbose output
            device='0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
        )
        
        print(f"Training completed. Results saved to: {save_dir}")
        return results
    
    def detect(self, image_path: str, conf_threshold: float = 0.25, 
               iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """
        Detect objects in an image with robust error handling.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detection dictionaries with keys: 'class', 'confidence', 'bbox'
            Returns empty list if detection fails
        """
        try:
            # Validate image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Validate image can be opened
            try:
                from PIL import Image
                test_img = Image.open(image_path)
                test_img.close()
            except Exception as e:
                raise ValueError(f"Cannot open image file (corrupted or invalid format): {e}")
            
            # Run inference with error handling
            try:
                results = self.model(image_path, conf=conf_threshold, iou=iou_threshold)
            except Exception as e:
                raise RuntimeError(f"YOLO inference failed: {e}")
            
            detections = []
            try:
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            try:
                                # Extract detection information with validation
                                bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                                confidence = float(boxes.conf[i].cpu())
                                class_id = int(boxes.cls[i].cpu())
                                
                                # Validate detection data
                                if not (0 <= confidence <= 1):
                                    continue  # Skip invalid confidence
                                if class_id < 0:
                                    continue  # Skip invalid class ID
                                if len(bbox) != 4:
                                    continue  # Skip invalid bbox
                                
                                # Get class name with fallback
                                if class_id < len(self.class_names):
                                    class_name = self.class_names[class_id]
                                else:
                                    class_name = f"class_{class_id}"
                                
                                detections.append({
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                                    'class_id': class_id
                                })
                            except Exception as e:
                                # Skip this detection but continue with others
                                print(f"Warning: Skipping invalid detection: {e}")
                                continue
            except Exception as e:
                raise RuntimeError(f"Error processing detection results: {e}")
            
            return detections
            
        except Exception as e:
            # Log error and return empty list for graceful degradation
            print(f"YOLO detection failed for {image_path}: {str(e)}")
            return []
    
    def display_annotated_image(self, image_path: str, detections: Optional[List[Dict]] = None,
                              conf_threshold: float = 0.25, save_path: Optional[str] = None,
                              show_plot: bool = True) -> str:
        """
        Display image with bounding box annotations.
        
        Args:
            image_path: Path to input image
            detections: Pre-computed detections. If None, will run detection.
            conf_threshold: Confidence threshold for detections
            save_path: Path to save annotated image. If None, generates automatic name.
            show_plot: Whether to display the plot using matplotlib
            
        Returns:
            Path to saved annotated image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get detections if not provided
        if detections is None:
            detections = self.detect(image_path, conf_threshold=conf_threshold)
        
        # Load image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_array)
        
        # Color palette for different classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        color_map = {class_name: colors[i] for i, class_name in enumerate(self.class_names)}
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Get color for this class
            color = color_map.get(class_name, 'red')
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            ax.text(x1, y1 - 5, label, fontsize=10, color=color, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Set title and remove axes
        ax.set_title(f"YOLOv8 Detection Results - {len(detections)} objects found", fontsize=14)
        ax.axis('off')
        
        # Generate save path if not provided
        if save_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_dir = os.path.join(os.path.dirname(image_path), 'annotated')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{base_name}_yolo_annotated.jpg")
        
        # Save the annotated image
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        print(f"Annotated image saved to: {save_path}")
        return save_path
    
    def predict_and_display(self, image_path: str, conf_threshold: float = 0.25,
                          save_annotated: bool = True, show_plot: bool = True) -> Tuple[List[Dict], str]:
        """
        Run detection and display annotated results in one step with robust error handling.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            save_annotated: Whether to save the annotated image
            show_plot: Whether to display the plot
            
        Returns:
            Tuple of (detections, annotated_image_path)
            Returns ([], "") if processing fails
        """
        try:
            print(f"Processing image: {image_path}")
            
            # Run detection with error handling
            detections = self.detect(image_path, conf_threshold=conf_threshold)
            
            print(f"Found {len(detections)} objects:")
            for det in detections:
                print(f"  - {det['class']}: {det['confidence']:.3f}")
            
            # Display annotated image with error handling
            annotated_path = ""
            if save_annotated or show_plot:
                try:
                    annotated_path = self.display_annotated_image(
                        image_path, detections, save_path=None, show_plot=show_plot
                    )
                except Exception as e:
                    print(f"Warning: Could not create annotated image: {e}")
                    annotated_path = ""
            
            return detections, annotated_path
            
        except Exception as e:
            print(f"YOLO predict_and_display failed for {image_path}: {str(e)}")
            return [], ""


def main():
    """Example usage of YOLODetector."""
    # Initialize detector
    data_yaml = os.path.join("image_classification", "utensils-wp5hm-yolo8", "data.yaml")
    detector = YOLODetector(data_yaml_path=data_yaml)
    
    # Example detection on a test image
    test_images = [
        "chopboard.jpg",
        "fork.jpg", 
        "garlicpress.jpg",
        "peeler.png"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n{'='*50}")
            print(f"Testing detection on: {img_path}")
            try:
                detections, annotated_path = detector.predict_and_display(
                    img_path, conf_threshold=0.25, show_plot=False
                )
                print(f"Annotated image saved: {annotated_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main() 