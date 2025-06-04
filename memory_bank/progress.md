# Progress: Kitchen Utensils Chatbot

## What Works
- AIML pattern matching for all utensil classes
- TF-IDF similarity with comprehensive QnA coverage
- Embedding fallback (spaCy) for semantic matching
- Input normalization (case, punctuation, contractions, spelling correction)
- Debug output for routing and confidence scores
- Logic engine (FOL, NLTK) for fact checking (robust tilde negation)
- Fuzzy safety (Simpful) for utensil safety queries (sharpness/grip, dual-path membership fallback)
- Canonical property parsing for multi-word properties (CamelCase)
- Default sharpness is now 5.0 (medium) for fuzzy demo utensils
- Router always routes logic/fuzzy before NLP
- Demo utensils for all fuzzy safety levels: kitchenknife (low), woodenspoon (high), ladle (moderate)
- Dev tool for KB integrity (build_kb.py)
- Unit tests for logic/fuzzy
- Vision stub structure with placeholder functions
- Complete dataset preparation for both CNN and YOLO approaches
- Dataset conversion and verification scripts
- Proper dataset citation and acknowledgments in README.md

## What's Left to Build
- CNN image classifier training and integration
- YOLOv8 object detection training and integration
- Image input handling in chatbot interface
- Performance comparison and documentation
- Unit tests for vision modules, polish, and final documentation

## Current Status
- Milestone 6 complete (vision stubs, dataset preparation, documentation)
- Both YOLO and CNN datasets organized with train/valid/test splits
- System robust for utensil QnA, semantic, logic, and fuzzy queries
- Ready for vision training and integration milestones

## Known Issues
- Vision models not yet trained (CNN and YOLO)
- Image input interface not yet implemented
- No persistent user state or web interface

## Evolution of Project Decisions
- Raised TF-IDF fallback threshold to 0.65 for better embedding use
- Lowered embedding threshold to 0.6 for improved semantic fallback
- Centralized threshold management for easier tuning
- Expanded QnA and improved normalization for accuracy
- Added FOL logic, fuzzy safety, and dev tools for KB integrity
- Robust FOL negation, canonical property parsing, default sharpness=5.0, robust fuzzy routing, dual-path fuzzy membership fallback, and demo utensils for all fuzzy safety levels
- Added proper dataset citation in README.md with corrected URL
- Prepared dual approach: CNN for single-label classification, YOLO for multi-object detection
- Created reproducible dataset preparation workflow with conversion scripts 