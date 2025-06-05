# System Patterns: Kitchen Utensils Chatbot

## Architecture Overview
- Modular pipeline: AIML → TF-IDF → Embedding → Logic → Vision
- Central router manages input normalization and fallback
- Data-driven QnA (qna.csv) for similarity and semantic matching
- CNN image classifier integrated with vision_reply function
- Image input handling with "image: path" syntax
- Stateless CLI interface for prototyping

## Key Technical Decisions
- Use AIML for exact pattern matching
- Use scikit-learn TF-IDF for token-based similarity
- Use spaCy en_core_web_md for embedding-based fallback
- Centralized threshold management for routing
- Input normalization (case, punctuation, contractions, spelling correction)
- Robust FOL negation: all rules use tilde (~) for negation, not custom NotX predicates
- Canonical property parsing: multi-word properties (e.g., "microwave safe") are parsed to CamelCase (e.g., MicrowaveSafe)
- Default sharpness is now 5.0 (medium) if no explicit fact is present, for correct fuzzy demo
- Fuzzy safety routing: logic/fuzzy always routed before NLP
- Dual-path fuzzy membership: safety_score uses fuzzy memberships if available, falls back to crisp value thresholds if not
- Demo utensils for all fuzzy safety levels: kitchenknife (low), woodenspoon (high), ladle (moderate)
- Dual vision approach: CNN for single-label classification, YOLO for multi-object detection
- CNN uses ResNet50V2 transfer learning with custom classification head (upgraded from MobileNetV3)
- Achieved 94.48% test accuracy with ResNet50V2 in just 2 epochs
- 23.8M total parameters: 23.5M frozen (ResNet50V2 base), 265K trainable (classification head)
- Confidence-based response formatting for vision predictions
- Proper dataset citation and acknowledgments in project documentation

## Design Patterns
- Fallback chain: Each module only triggers if previous fails/confidence is low
- Data-driven QnA: All similarity/embedding answers come from qna.csv
- Stateless: No user session or context tracking
- Modular vision architecture: Separate CNN and YOLO modules with unified interface

## Component Relationships
- main.py: Central router, CLI, debug output, vision integration
- nlp/: Similarity, embedding, normalization
- logic/: Logic and fuzzy reasoning (integrated)
- image_classification/: CNN classifier (trained), YOLO stub (ready)
- scripts/: Dataset preparation, verification, and training tools

---

Milestone 7 complete: CNN image classifier successfully trained and integrated. MobileNetV3-based model handles 21 utensil classes with image input via "image: path" syntax. Ready for YOLOv8 implementation (Milestone 8). 