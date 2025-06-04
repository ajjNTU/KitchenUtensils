# Active Context: Kitchen Utensils Chatbot

**Milestone 6 (Vision Stub & Dataset Preparation) complete. Dataset citation properly added to README.md. Ready for CNN and YOLO implementation (Milestone 7-8).**

## Current Work Focus
- Implementing CNN image classifier for utensil recognition (Milestone 7)
- Preparing for YOLOv8 object detection integration (Milestone 8)
- Both YOLO and CNN datasets prepared and verified

## Recent Changes
- Completed Milestone 6: Vision stub structure and dataset preparation
- Added proper dataset citation to README.md with corrected URL (https://universe.roboflow.com/utensils/utensils-wp5hm)
- Organized both YOLO (utensils-wp5hm-yolo8/) and CNN (cls_data/) datasets with train/valid/test splits
- Created scripts for dataset conversion (crop_yolo_to_classification.py) and verification (count_split_images.py)
- Added acknowledgments section to README.md for proper attribution
- Updated TODO.md with expanded milestones and practical implementation steps
- All 21 utensil classes present in both dataset formats

## Next Steps
- Train CNN classifier using transfer learning (MobileNetV3/EfficientNet-B0)
- Integrate CNN into chatbot with image input handling
- Train YOLOv8 model for multi-object detection
- Compare CNN vs YOLO performance and document results

## Active Decisions & Considerations
- Centralized threshold management for routing
- Stateless, modular design
- All QnA and logic data in CSV for easy updates
- CLI interface for rapid prototyping
- Proper academic citation in README.md for dataset attribution
- Both CNN (single-label classification) and YOLO (multi-object detection) approaches
- Dataset preparation scripts for reproducibility
- Fuzzy safety only uses sharpness and grip; KB must provide these for demo utensils
- Fuzzy safety label is robust to Simpful API changes (dual-path logic)
- All logic/fuzzy queries are routed before NLP for clarity and demo reliability
- Demo utensils: kitchenknife (low safety), woodenspoon (high safety), ladle (moderate safety)

## Important Patterns & Preferences
- Fallback chain: AIML → TF-IDF → Embedding → Logic → Vision
- Input normalization before all processing
- Debug output for transparency and testing
- Proper dataset citation and acknowledgments in documentation
- Modular vision architecture supporting both CNN classification and YOLO detection 