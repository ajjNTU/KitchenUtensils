# Tech Context: Kitchen Utensils Chatbot

## Technologies Used
- Python 3.10.x
- python-aiml (AIML pattern matching)
- scikit-learn (TF-IDF similarity)
- spaCy (en_core_web_md, embeddings)
- simpful (fuzzy logic)
- ultralytics, torch (YOLOv8, trained and integrated)
- tensorflow/keras (CNN classification, integrated)
- pillow (image processing)
- pyspellchecker (input normalization)
- NLTK (FOL logic engine)
- albumentations (image augmentation for YOLO training)

## Development Setup
- Virtual environment (.venv)
- Install dependencies: `pip install -r requirements.txt`
- Download spaCy model: `python -m spacy download en_core_web_md`

## Technical Constraints
- CLI prototype (no web or GUI yet)
- No persistent user state
- All QnA and logic data in CSV files
- Embedding cache auto-updates on QnA change
- **PyTorch 2.6 Compatibility**: Resolved loading issues for YOLO models using legacy loading methods

## Dependencies
- See requirements.txt for full list

## Tool Usage Patterns
- All NLP modules importable from nlp/
- Logic and fuzzy modules importable from logic/
- Vision modules in image_classification/ with CNN and YOLO models trained and integrated
- Dataset preparation scripts in scripts/ for reproducibility
- Robust FOL negation: all rules use tilde (~) for negation, not custom NotX predicates
- Canonical property parsing: multi-word properties (e.g., "microwave safe") are parsed to CamelCase (e.g., MicrowaveSafe)
- Default sharpness is now 5.0 (medium) if no explicit fact is present, for correct fuzzy demo
- Fuzzy safety routing: logic/fuzzy always routed before NLP
- Dual-path fuzzy membership: safety_score uses fuzzy memberships if available, falls back to crisp value thresholds if not
- Demo utensils for all fuzzy safety levels: kitchenknife (low), woodenspoon (high), ladle (moderate)
- **Production/Debug Modes**: `python main.py` (clean interface) vs `python main.py --debug` (technical details)
- **Message Suppression**: Comprehensive suppression of verbose library output in production mode
- **Conditional Imports**: StringIO redirection during module imports to suppress initialization messages
- **Environment Configuration**: TensorFlow logging levels and library warning suppression
- **Argument Parsing**: argparse integration for command-line flag control

## Dataset Information
- **Source:** Kitchen Utensils Dataset from Roboflow
- **URL:** https://universe.roboflow.com/utensils/utensils-wp5hm
- **Format:** YOLO (detection) and CNN (classification) splits prepared
- **Classes:** 21 kitchen utensil categories
- **Access Date:** 04/06/2025

## Enhanced YOLO Training Infrastructure
### **Multi-Object Dataset Generation**
- **Object Extraction**: Intelligent cropping from single-object YOLO data
- **Scene Composition**: Size-based placement with kitchen-specific rules
- **Augmentation**: Albumentations pipeline for realistic kitchen effects
- **Background Generation**: Synthetic kitchen countertop environments
- **Dataset Integration**: Combined original + synthetic (30% multi-object ratio)

### **Training Configuration**
- **Model**: YOLOv8s (small variant for balance of speed and accuracy)
- **Training Duration**: 0.228 hours (13.7 minutes) for 20 epochs
- **Dataset Size**: 1,228 train + 388 valid images
- **Performance**: 76.6% mAP50, 58.7% mAP50-95 (training metrics)
- **Compatibility**: PyTorch 2.6 loading issues resolved with legacy methods

### **Real-World Validation Challenges**
- **Performance Gap**: Strong training metrics but poor real-world performance
- **Synthetic Quality Issues**: Generated scenes lack realism for deployment
- **Domain Mismatch**: Training environment doesn't match real kitchen photos
- **Quality Priority**: High-quality data more important than quantity

---

Enhanced YOLO training complete with technical success but real-world performance challenges identified. PyTorch 2.6 compatibility resolved. CNN classification remains excellent with 96.73% accuracy. Multi-object detection requires synthetic dataset quality improvement for production deployment. 