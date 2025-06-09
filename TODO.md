# Kitchen Utensils Chatbot - TODO

**PROJECT SCOPE: University Module Assessment**
This project is being developed for a university module assessment. The scope is intentionally limited to demonstrate core AI/ML concepts and implementation skills. Full production deployment is not the objective.

## Milestone 1: Router Skeleton ✅ COMPLETE
- ✅ BotReply dataclass with text and end_conversation
- ✅ Stub functions for each module (aiml_reply, tfidf_reply, embed_reply, logic_reply, vision_reply)
- ✅ Routing order: AIML → TF-IDF → Embedding → fallback
- ✅ CLI loop with fallback message
- ✅ Central router manages all input and fallback logic
- ✅ Stateless design: no user session or persistent state
- ✅ Debug output for routing and confidence scores

## Milestone 2: AIML Baseline ✅ COMPLETE  
- ✅ Create `aiml/utensils.aiml` with patterns for all 21 classes
- ✅ Include spaced and unspaced variants (e.g., "GARLIC PRESS" and "GARLICPRESS")
- ✅ Load AIML kernel in main.py and implement aiml_reply
- ✅ Enhanced intro message with examples and class list
- ✅ Data-driven QnA: all patterns and answers in maintainable files
- ✅ Input normalization before AIML (case, punctuation, contractions, spelling correction)

## Milestone 3: TF-IDF Similarity ✅ COMPLETE
- ✅ Create `nlp/similarity.py` with TfidfSimilarity class
- ✅ Use scikit-learn TF-IDF with 0.5 similarity threshold (later raised to 0.65 for better fallback)
- ✅ Create comprehensive `qna.csv` with 200+ question variations
- ✅ Integrate into main.py as fallback after AIML
- ✅ Centralized threshold management for easy tuning
- ✅ Debug output: always show top TF-IDF candidates and scores
- ✅ QnA and logic data in CSV for easy updates
- ✅ Input normalization applied before TF-IDF

## Milestone 4: Embedding Fallback ✅ COMPLETE & TESTED
- ✅ Add spaCy en_core_web_md to requirements.txt
- ✅ Create `nlp/embedding.py` with EmbeddingSimilarity class
- ✅ Generate embeddings for each question; save to .npy files
- ✅ Use embedding similarity (≥0.6 threshold) only when TF-IDF confidence is <0.65
- ✅ Centralize TF-IDF threshold as a constant in codebase (easy to change)
- ✅ Input normalization: lowercase, remove punctuation, expand contractions, spellcheck (pyspellchecker)
- ✅ Apply normalization to all user input before AIML, TF-IDF, and embedding
- ✅ Expand QnA dataset with generic entries for all utensils (e.g., "What is a knife?", "What can I use a fork for?")
- ✅ Improved debug output: always show top TF-IDF and embedding candidates and scores
- ✅ Fix embedding threshold bug (now always 0.6)
- ✅ Remove duplicate/noisy fallback and warning messages in debug output
- ✅ Remove old embedding cache on QnA update
- ✅ Comprehensive manual and automated testing of fallback logic and accuracy

## Milestone 5: Logic Engine & Fuzzy Safety ✅ COMPLETE
- ✅ Implement FOL logic engine with NLTK (logic_engine.py)
- ✅ Add and parse logical-kb.csv (≥10 entries, tight syntax)
- ✅ Integrate synonym map (aliases.py) for robust query handling
- ✅ Integrate logic/fuzzy into main router (main.py)
- ✅ Fuzzy safety model (simpful) with sharpness/grip/safety
- ✅ Add dev tool: logic/build_kb.py for KB integrity
- ✅ Add unit tests: tests/test_logic.py
- ✅ Update docs: ARCHITECTURE.md (diagram), FLOW_EXAMPLES.md (scenarios)
- ✅ Robust FOL negation: all rules use tilde (~) for negation, not custom NotX predicates
- ✅ Canonical property parsing: multi-word properties (e.g., "microwave safe") are parsed to CamelCase (e.g., MicrowaveSafe)
- ✅ Default sharpness is now 5.0 (medium) if no explicit fact is present, for correct fuzzy demo
- ✅ Fuzzy safety routing: logic_reply now always called first, handles all logic/fuzzy queries before NLP
- ✅ Dual-path fuzzy membership: safety_score uses fuzzy memberships if available, falls back to crisp value thresholds if not
- ✅ Demo utensils for all fuzzy safety levels: kitchenknife (low), woodenspoon (high), ladle (moderate)
- ✅ All logic/fuzzy milestones and demos validated in CLI

## Milestone 6: Vision Stub & Dataset Preparation ✅ COMPLETE
- ✅ Structure image_classification/ with __init__.py, cnn.py, yolo.py, utils.py
- ✅ Add placeholder predict(image) in cnn.py and yolo.py for chatbot integration
- ✅ Add dependency guard for Ultralytics (YOLO) import
- ✅ Prepare and organize datasets for both CNN and YOLO:
    - YOLO: utensils-wp5hm-yolo8/ (YOLOv8 format, images + labels)
    - CNN: cls_data/ (cropped images per class, train/val/test split)
- ✅ Use scripts/crop_yolo_to_classification.py to generate CNN crops from YOLO labels
- ✅ Confirm dataset splits and counts with scripts/count_split_images.py

## Milestone 7: Custom CNN Image Classifier ✅ COMPLETE
- ✅ Use transfer learning (ResNet50V2 - upgraded from MobileNetV3) for utensil classification
- ✅ Apply augmentations: flips, rotations, width/height shifts, zoom
- ✅ Train on cls_data/ (single-label, one crop per object) - 15 epochs
- ✅ Save model as cnn_model.h5 and document model architecture
- ✅ Evaluate: 96.73% test accuracy, confusion matrix, per-class analysis
- ✅ Document model architecture, training, and optimization journey (CNN_OPTIMIZATION_JOURNEY.md)
- ✅ Integrate into chatbot: "What is in this image?" opens file dialog, runs classifier, returns results
- ✅ Add main loop handler for both "image: path" syntax and natural language trigger
- ✅ CNN optimization complete - conservative parameter tuning achieved optimal performance

## Milestone 8: YOLOv8 Object Detection ✅ COMPLETE
- ✅ Train YOLOv8 on utensils-wp5hm-yolo8/ (detection format) - YOLOv8-small, 100 epochs, 2.15 hours on GTX 1080 Ti
- ✅ Save weights as runs/detect/train/weights/best.pt
- ✅ Ensure class mapping matches chatbot labels (21 classes)
- ✅ Integrate YOLO into chatbot: "Detect everything in this image" opens file dialog, runs detection, returns list of detected utensils and confidences
- ✅ Save/display annotated image with bounding boxes and confidence scores
- ✅ Compare YOLO mAP@0.5 to CNN accuracy: YOLO 97.2% mAP50 vs CNN 96.73% accuracy
- ✅ Document training results: Test set performance 97.2% mAP50, 73.8% mAP50-95, 95.7% precision, 94.7% recall
- ✅ Separate trigger phrases: "What is in this image?" (CNN) vs "Detect everything in this image" (YOLO)
- ✅ PyTorch 2.6 compatibility fixes for model loading

## Milestone 9: Critical Bug Fixes ✅ COMPLETE

### **✅ CRITICAL BUG FIXES (Priority 1) - ALL COMPLETE**
- [x] **Fix startup examples**: Replace "spatula" with actual class names (e.g., "What is a fishslice?", "What is a ladle?")
- [x] **Fix incomplete QnA responses**: Fixed CSV parsing issue where unquoted commas in answer fields caused truncation (e.g., "describe a ladle" now returns full text)
- [x] **Fix logic pipeline fallback**: Simple fix - logic_reply() now returns BotReply for ALL results including "Unknown." (keeps logic pipeline separate from NLP)
- [x] **Implement material inference**: Enable logic system to infer properties through material rules (e.g., woodenspoon → wood → MicrowaveSafe)
- [x] **Add debug/production modes**: Implement `--debug` flag for main.py to toggle between clean chatbot output and detailed debug information

### **✅ POST-PRODUCTION DEBUG ENHANCEMENTS - ALL COMPLETE**
- [x] **Logic debug message suppression**: Fixed logic engine debug messages showing in production mode by adding `set_debug_mode()` function
- [x] **Improved error messages**: Enhanced `assert_fact()` error handling with specific, actionable feedback instead of generic "Sorry, I couldn't process that fact"
- [x] **Multi-word utensil names**: Fixed "chopping board", "kitchen knife", "wooden spoon" parsing by adding missing aliases to canonical name mapping

## Task 4: Code Quality & Robustness - Simplified Plan

**UNIVERSITY MODULE SCOPE**: These tasks demonstrate software engineering best practices for academic assessment. Full production deployment is not the objective.

### **Phase 2: Input Validation & Sanitization (Quick Fixes)**
- [ ] Strengthen existing input validation
- [ ] Add basic safety checks for file paths and user input
- [ ] Improve error messages for invalid inputs
- [ ] Keep implementation simple and practical

### **Phase 3.1: Lazy Loading Only**
- [ ] Print welcome message immediately on startup
- [ ] Load models in background while user reads welcome message
- [ ] Show loading progress/status for model initialization
- [ ] Improve user experience (10s startup → instant welcome)

### **Phase 5.1: Code Organization & Cleanup**
- [ ] Extract common utilities into shared modules
- [ ] Reduce code duplication across modules
- [ ] Improve inline documentation and comments
- [ ] Add type hints where helpful for clarity
- [ ] Clean up the codebase structure

### **Phase 6.2: Documentation Update**
- [ ] Update README and docs with recent improvements
- [ ] Document the lazy loading feature and user experience
- [ ] Clean up any outdated information in documentation
- [ ] Ensure all docs reflect university module scope

---

## Future Improvements (Due: Probably Never)

**Note**: These are advanced features that would be relevant for production systems but are beyond the scope of a university module assessment.

### **Logging Infrastructure**
- Centralised logging system with log rotation
- Structured logging with context
- Performance timing logs
- Memory usage tracking

### **Advanced Performance**
- Prediction caching for repeated queries
- Memory management and cleanup
- Configuration management system
- Runtime configuration updates

### **Error Handling Framework**
- Custom exception classes
- Standardised error patterns
- Retry logic for transient failures
- Error reporting/metrics

### **Advanced Monitoring**
- Performance monitoring decorators
- Response time tracking
- System health checks
- Memory leak detection

### **Advanced Testing**
- Comprehensive unit test suite
- Integration tests for error handling
- Performance benchmarks
- Automated testing pipeline

### **Production Features**
- Web interface development
- User authentication and sessions
- Database integration
- API development
- Containerisation and deployment
- Load balancing and scaling
- Security hardening
- Compliance and auditing

---

**Current Status:** ✅ **MILESTONE 9 COMPLETE** - All critical functionality implemented for university module assessment. System demonstrates:
- Multi-modal AI pipeline (NLP + Computer Vision + Logic/Fuzzy reasoning)
- Robust fallback mechanisms and error handling
- Clean production interface with comprehensive debug capabilities
- Software engineering best practices (modular design, testing, documentation)

**Next Phase:** Task 4 implementation to demonstrate code quality and robustness principles for academic assessment.

**Project Scope Reminder**: This is a university module assessment project designed to demonstrate AI/ML implementation skills and software engineering practices. The focus is on technical competency rather than production deployment. 