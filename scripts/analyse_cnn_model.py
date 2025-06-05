# Add this to your existing cnn_classifier.py or run separately
from image_classification.cnn_classifier import CNNClassifier
import os
classifier = CNNClassifier('image_classification/cnn_model.h5')

# Test a few samples and see what goes wrong
test_dir = 'image_classification/cls_data/test'
for class_name in ['Canopener', 'Peeler', 'Garlicpress']:
    class_path = os.path.join(test_dir, class_name)
    if os.path.exists(class_path):
        images = os.listdir(class_path)
        if images:
            img_path = os.path.join(class_path, images[0])
            predictions = classifier.predict(img_path, top_k=3)
            print(f'\nActual: {class_name}')
            print(f'Predicted: {predictions[0][0]} ({predictions[0][1]:.1%} confidence)')
            if predictions[0][0].lower() != class_name.lower():
                print('❌ WRONG! Top 3 predictions:')
                for i, (pred, conf) in enumerate(predictions, 1):
                    print(f'  {i}. {pred}: {conf:.1%}')
