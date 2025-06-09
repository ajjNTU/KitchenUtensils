# System Patterns: Kitchen Utensils Chatbot

**PROJECT SCOPE: University Module Assessment**
This project demonstrates AI/ML implementation skills and software engineering practices for academic assessment. The architecture showcases multi-modal AI integration and robust system design principles.

## Architecture Overview
- **Dual Pipeline System**: Logic/Fuzzy pipeline (separate) + NLP pipeline (fallback chain)
- **Logic/Fuzzy Pipeline**: Runs first for fact assertions, fact checks, and safety queries - no NLP fallback
- **NLP Pipeline**: AIML → TF-IDF → Embedding (only runs if Logic/Fuzzy doesn't match)
- **Vision Pipeline**: CNN and YOLO integration with multiple input methods
- Central router manages input normalization and pipeline routing
- Data-driven QnA (qna.csv) for similarity and semantic matching
- CNN image classifier integrated with vision_reply function
- Image input handling with "image: path" syntax and natural language triggers
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
- **Material Inference**: Enhanced assert_fact() with universal quantification for material-based rules
- **FOL Reasoning**: Material properties now connect to specific utensils via inference (wood → woodenspoon)
- **Logic Pipeline Separation**: Clean separation from NLP, no fallback mixing for "Unknown." results
- **Production Bug Fixes**: 4/5 critical fixes complete for deployment readiness
- Dual vision approach: CNN for single-label classification, YOLO for multi-object detection
- CNN uses ResNet50V2 transfer learning with custom classification head (upgraded from MobileNetV3)
- Achieved 94.48% test accuracy with ResNet50V2 in just 2 epochs
- 23.8M total parameters: 23.5M frozen (ResNet50V2 base), 265K trainable (classification head)
- Confidence-based response formatting for vision predictions
- Proper dataset citation and acknowledgments in project documentation
- **Enhanced YOLO Training**: Combined single+multi-object dataset approach
- **Multi-Object Dataset Generation**: Synthetic scene composition with object extraction and placement
- **Real-World Validation**: Essential for validating computer vision performance beyond training metrics

## Design Patterns
- **Dual Pipeline Architecture**: Logic/Fuzzy pipeline completely separate from NLP pipeline
- **Logic Pipeline Isolation**: Logic results (including "Unknown.") stop processing - no NLP fallback mixing
- **NLP Fallback Chain**: Each NLP module only triggers if previous fails/confidence is low
- Data-driven QnA: All similarity/embedding answers come from qna.csv
- Stateless: No user session or context tracking
- Modular vision architecture: Separate CNN and YOLO modules with unified interface
- **Material-based reasoning**: Universal quantification enables inference from materials to specific utensils
- **Simple over complex**: One-line fixes often more robust than elaborate systems (Fix #3 lesson)
- **Clean separation**: Logic and NLP pipelines remain completely distinct
- **Incremental bug fixing**: Address one issue at a time with thorough testing
- **Real-world testing priority**: Training metrics must be validated with actual deployment scenarios
- **Quality over quantity**: High-quality training data more important than large synthetic datasets
- **Domain alignment**: Training and deployment environments must be carefully matched
- **Production-first design**: Clean user experience prioritised for deployment readiness
- **Conditional functionality**: Single flag controls development vs production features
- **Message suppression**: Comprehensive approach to hiding verbose library output in production
- **Debug preservation**: Complete technical information maintained for development needs

## Component Relationships
- main.py: Central router, CLI, debug output, vision integration
- nlp/: Similarity, embedding, normalization
- logic/: Logic and fuzzy reasoning (integrated)
- image_classification/: CNN classifier (trained), YOLO models (original + enhanced)
- scripts/: Dataset preparation, verification, training tools, and testing infrastructure

## Enhanced YOLO Architecture Patterns
### **Multi-Object Dataset Generation System**
- **Object Extraction**: Intelligent cropping from single-object YOLO data with padding
- **Scene Composition**: Size-based placement hierarchy with realistic spatial rules
- **Augmentation Pipeline**: Albumentations integration for kitchen-specific effects
- **Background Generation**: Synthetic kitchen countertop backgrounds
- **Dataset Integration**: Combined original + synthetic data (30% multi-object ratio)

### **Training Infrastructure**
- **PyTorch Compatibility**: Resolved PyTorch 2.6 loading issues during training
- **Performance Monitoring**: Comprehensive training metrics and validation tracking
- **Model Management**: Organized weight storage and training result archival
- **Testing Framework**: Automated inference and real-world validation systems

### **Quality Validation Patterns**
- **Real-World Testing**: Essential validation step beyond training metrics
- **Performance Gap Analysis**: Systematic comparison of training vs deployment performance
- **Synthetic Quality Assessment**: Evaluation of generated data realism and effectiveness
- **Domain Gap Detection**: Identification of training/deployment environment mismatches

## Critical Lessons Learned
### **Synthetic Dataset Quality Issues**
- **Domain Gap**: Synthetic multi-object scenes don't match real kitchen environments
- **Unrealistic Placement**: Objects appear artificially positioned without natural context
- **Lighting Inconsistencies**: Extracted objects don't blend naturally with backgrounds
- **Scale Problems**: Object sizes may not reflect realistic proportions
- **Missing Context**: Objects lack realistic support surfaces and spatial relationships

### **Training vs Real-World Performance**
- **Metrics Can Mislead**: Strong training performance (76.6% mAP50) doesn't guarantee real-world success
- **Validation Strategy**: Real-world testing reveals issues not apparent in training metrics
- **Quality Priority**: High-quality training data more important than quantity
- **Early Testing**: Test with real data early and often to guide development

### **Technical Success Patterns**
- **PyTorch Compatibility**: Successfully resolved version compatibility issues
- **Training Pipeline**: Robust infrastructure with comprehensive monitoring
- **Dataset Integration**: Effective merging of single and multi-object datasets
- **Performance Tracking**: Systematic analysis and comparison capabilities

---

Enhanced YOLO training complete with important lessons learned about synthetic dataset quality. While technical training succeeded (76.6% mAP50), real-world performance revealed critical issues with synthetic data quality that require addressing. The project demonstrates the importance of real-world validation beyond training metrics. 