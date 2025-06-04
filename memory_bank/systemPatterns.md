# System Patterns: Kitchen Utensils Chatbot

## Architecture Overview
- Modular pipeline: AIML → TF-IDF → Embedding → Logic → Vision
- Central router manages input normalization and fallback
- Data-driven QnA (qna.csv) for similarity and semantic matching
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
- Proper dataset citation and acknowledgments in project documentation

## Design Patterns
- Fallback chain: Each module only triggers if previous fails/confidence is low
- Data-driven QnA: All similarity/embedding answers come from qna.csv
- Stateless: No user session or context tracking
- Modular vision architecture: Separate CNN and YOLO modules with unified interface

## Component Relationships
- main.py: Central router, CLI, debug output
- nlp/: Similarity, embedding, normalization
- logic/: Logic and fuzzy reasoning (integrated)
- image_classification/: Vision modules (CNN and YOLO with stubs ready)
- scripts/: Dataset preparation and verification tools

---

Dataset preparation complete: Both YOLO (object detection) and CNN (classification) datasets organized with proper train/valid/test splits. Vision stub architecture ready for model training and integration (Milestone 7-8). 