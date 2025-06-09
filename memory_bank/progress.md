# Progress: Kitchen Utensils Chatbot

**PROJECT SCOPE: University Module Assessment**
This project demonstrates AI/ML implementation skills and software engineering practices for academic assessment. The focus is on technical competency rather than production deployment.

## What Works
- **Logic/Fuzzy Pipeline (Step 0)**: Runs first for fact assertions, fact checks, and safety queries ✅ **ENHANCED**: Clean separation from NLP, no fallback mixing
- **NLP Pipeline (Steps 1-3)**: Only runs if Logic/Fuzzy doesn't match ✅
- AIML pattern matching for all utensil classes
- TF-IDF similarity with comprehensive QnA coverage ✅ **ENHANCED**: CSV formatting fixed for complete responses
- Embedding fallback (spaCy) for semantic matching ✅ **ENHANCED**: Embeddings regenerated after CSV fix
- Input normalization (case, punctuation, contractions, spelling correction)
- Debug output for routing and confidence scores
- Logic engine (FOL, NLTK) for fact checking (robust tilde negation) ✅ **ENHANCED**: Material inference with universal quantification
- Fuzzy safety (Simpful) for utensil safety queries (sharpness/grip, dual-path membership fallback)
- Canonical property parsing for multi-word properties (CamelCase)
- Default sharpness is now 5.0 (medium) for fuzzy demo utensils
- Router routes logic/fuzzy first, then NLP pipeline if no logic match
- Demo utensils for all fuzzy safety levels: kitchenknife (low), woodenspoon (high), ladle (moderate)
- Dev tool for KB integrity (build_kb.py)
- Unit tests for logic/fuzzy
- **CNN image classifier OPTIMIZED and COMPLETE**: ResNet50V2 with 96.73% test accuracy
- **CNN Optimization Journey**: Thorough parameter analysis → Conservative optimization → Validation of diminishing returns
- **Optimal CNN Parameters**: Learning rate 0.0003, Dense 256 neurons, Dropout 0.3, 224x224 input
- **Enhanced image input handling**: Dual approach with direct syntax and natural language trigger
  - Direct syntax: "image: path/to/image.jpg"
  - Natural language: "What is in this image?" (opens file dialog box)
  - Specific phrase trigger with tkinter file dialog and image type filters
- Vision_reply function with confidence-based responses
- Complete dataset preparation for both CNN and YOLO approaches
- Dataset conversion and verification scripts
- Comprehensive model comparison and testing scripts
- Parameter optimization methodology and documentation
- Proper dataset citation and acknowledgments in README.md
- **YOLOv8 TRAINED AND OPERATIONAL**: Single-object detection with 97.2% mAP50
- **Multi-Object Dataset Generation System**: Comprehensive augmentation pipeline implemented
  - Object extraction from single-object YOLO data with intelligent cropping
  - Realistic multi-object scene composition with kitchen-specific placement rules
  - Albumentations integration for kitchen-specific augmentations (lighting, blur, shadows)
  - Generated 447 multi-object training scenes (307 train, 97 valid, 43 test)
  - Combined dataset: 1,788 total images with 25% multi-object scenes
  - Kitchen background generation with realistic countertop colors and textures
  - Size-based placement hierarchy (large items first, realistic spatial relationships)
  - Object grouping preferences (utensil sets, cooking tools, prep tools, containers)
- **Enhanced YOLO Training COMPLETE**: Combined single+multi-object dataset training successful
  - Training Duration: 0.228 hours (13.7 minutes) for 20 epochs
  - Final Performance: 76.6% mAP50, 58.7% mAP50-95
  - Dataset: 1,228 train + 388 valid images (30% multi-object ratio)
  - PyTorch 2.6 compatibility issues resolved during training
- **Real-World Testing Infrastructure**: Comprehensive testing and analysis system
  - Automated inference script for PXL_ photos
  - JSON result summaries with detailed detection statistics
  - Annotated image generation with bounding boxes and confidence scores
  - Performance tracking and comparison capabilities

### ✅ **MILESTONE 9: CRITICAL BUG FIXES (5/5 COMPLETE)**

#### **Fix #1: Startup Examples ✅ COMPLETE**
- **Problem Fixed**: Welcome message showed invalid "spatula" example (not in 21 supported classes)
- **Solution**: Replaced with valid class names in main.py:
  - "What is a fishslice?"
  - "What is a ladle?"
- **Result**: Users now see working examples that demonstrate actual system capabilities

#### **Fix #2: Incomplete QnA Responses ✅ COMPLETE**  
- **Problem Fixed**: "describe a ladle" returned truncated "A ladle is a large" (missing 28 characters)
- **Root Cause**: CSV unquoted commas in answer fields caused DictReader parsing errors
- **Investigation**: Debug scripts revealed answer splitting: "A ladle is a large, deep-bowled spoon for serving" → Column 1: "A ladle is a large", Column 2: " deep-bowled spoon for serving."
- **Solution**: Created fix_csv.py script that properly quoted 50+ problematic answer fields
- **Verification**: All QnA responses now return complete text (full 50-character responses)
- **Files Updated**: qna.csv (proper CSV formatting), qna_embeddings.npy (regenerated)

#### **Fix #3: Logic Pipeline Fallback ✅ COMPLETE**
- **Problem Fixed**: Logic queries returning "Unknown." were falling through to NLP instead of stopping
- **User Requirement**: Keep logic pipeline completely separate from NLP (no mixed responses)
- **Solution**: Simplified logic_reply() to return BotReply for ALL logic results including "Unknown."
- **Implementation**: One-line change removing result filtering
- **Testing**: ✅ "check that woodenspoon is microwave safe" → "Unknown." (stops cleanly)
- **Key Insight**: Simple solutions often more robust than complex fallback systems

#### **Fix #4: Material Inference ✅ COMPLETE**
- **Problem Fixed**: Logic system couldn't infer properties through material rules
- **Goal**: Enable material-based universal quantification and inference
- **Use Cases**: 
  1. "I know that wood is microwave safe" → Add rule: `all x.(Wood(x) -> MicrowaveSafe(x))`
  2. "check that woodenspoon is microwave safe" → Return "Correct." via inference
- **Implementation**: Enhanced `assert_fact()` function in logic/logic_engine.py
  - **Material Detection**: Recognises known materials (wood, metal, plastic, ceramic) vs specific utensils
  - **Universal Rule Generation**: For materials, creates `all x.(Material(x) -> Property(x))` format
  - **Individual Facts**: Maintains existing logic for specific utensils
  - **Contradiction Checking**: Validates individual facts, universal rules handled separately
- **Testing Results** (All ✅ PASS):
  - ✅ Material rule addition: "wood is microwave safe" → `all x.(Wood(x) -> MicrowaveSafe(x))`
  - ✅ Material inference: "check that woodenspoon is microwave safe" → "Correct."
  - ✅ Individual facts: "tray is microwave safe" → `MicrowaveSafe(tray)` (still works)
  - ✅ Negative material rule: "plastic is not oven safe" → `all x.(Plastic(x) -> ~OvenSafe(x))`
  - ✅ Negative inference: "check that colander is not oven safe" → "Correct."
- **Key Achievement**: FOL reasoning now connects material properties to specific utensils via inference

#### **Fix #5: Debug/Production Modes ✅ COMPLETE**
- **Problem Fixed**: No clean production interface - debug output always displayed
- **Goal**: Professional user experience with optional technical details for developers
- **Implementation**: Comprehensive argument parsing and message suppression system
- **Command Line Interface**:
  - **Production Mode**: `python main.py` - Clean, professional interface
  - **Debug Mode**: `python main.py --debug` - Complete technical information
- **Production Mode Features**:
  - ✅ Clean welcome message: "Welcome to the Kitchen Utensils Chatbot!"
  - ✅ No routing debug output (🔍, ─, 0️⃣, 1️⃣, 2️⃣, 3️⃣, 5️⃣, ✅)
  - ✅ Suppressed TensorFlow verbose messages (oneDNN, CPU optimization, deprecation warnings)
  - ✅ Suppressed AIML loading messages
  - ✅ Suppressed embedding loading/caching messages
  - ✅ Suppressed Simpful banner (ASCII art logo)
  - ✅ Essential model loading confirmations only
  - ✅ Just final answers and user-friendly responses
- **Debug Mode Features**:
  - ✅ Detailed welcome: "Welcome to the Kitchen Utensils Chatbot (Prototype) - DEBUG MODE"
  - ✅ Complete routing pipeline visibility
  - ✅ All TensorFlow diagnostic information
  - ✅ AIML loading details
  - ✅ Embedding generation/loading messages
  - ✅ Simpful banner for development reference
  - ✅ Technical details and confidence scores
- **Technical Implementation**:
  - **Argument Parsing**: Added argparse with `--debug` flag
  - **Global Debug Flag**: `DEBUG_MODE` controls all verbose output
  - **Environment Variables**: TensorFlow logging levels and oneDNN suppression
  - **Warning Filters**: Suppressed FutureWarning and DeprecationWarning categories
  - **Logging Configuration**: Set library loggers to ERROR level only
  - **Import-Time Suppression**: StringIO redirection during module imports (AIML, embeddings, Simpful)
  - **Conditional Output**: All debug prints wrapped with `if DEBUG_MODE:`
- **Testing Results**:
  - ✅ Production mode: Clean, professional interface suitable for end users
  - ✅ Debug mode: Complete technical visibility for development
  - ✅ All functionality identical in both modes
  - ✅ Single flag controls all verbose output consistently

### ✅ **POST-PRODUCTION DEBUG ENHANCEMENTS (3/3 COMPLETE)**

#### **Logic Debug Message Suppression ✅ COMPLETE**
- **Problem Fixed**: Logic engine debug messages (`[DEBUG] Loaded X FOL facts...`) showing in production mode
- **Root Cause**: Logic module didn't have access to DEBUG_MODE flag from main.py
- **Solution**: Added `set_debug_mode()` function to logic engine and integrated with main.py
- **Implementation**:
  - Added `_DEBUG_MODE` global flag and `set_debug_mode()` function to logic/logic_engine.py
  - Updated logic/__init__.py to export `set_debug_mode`
  - Modified main.py to call `set_debug_mode(DEBUG_MODE)` after importing logic module
  - Wrapped all debug print statements in logic engine with `if _DEBUG_MODE:` checks
- **Testing Results**:
  - ✅ Production mode: No `[DEBUG]` messages from logic engine
  - ✅ Debug mode: All logic debug information properly displayed
  - ✅ Functionality unchanged in both modes

#### **Improved Logic Error Messages ✅ COMPLETE**
- **Problem Fixed**: Generic "Sorry, I couldn't process that fact" error message wasn't helpful
- **Solution**: Enhanced error handling in `assert_fact()` with specific error messages based on exception type
- **New Error Messages**:
  - Parse/syntax errors: "Sorry, I couldn't understand the property 'X'. Please use simple properties like 'microwave safe', 'dishwasher safe', etc."
  - Contradiction errors: "Sorry, that contradicts what I already know."
  - Unknown utensil/material: "Sorry, I don't recognise 'X' as a known utensil or material."
  - General fallback: "Sorry, I couldn't process that fact. Please try rephrasing it as 'I know that [utensil/material] is [property]'."
- **Testing Results**:
  - ✅ Users now get specific, actionable error messages
  - ✅ Debug mode shows technical exception details for developers
  - ✅ Production mode shows user-friendly error explanations

#### **Fixed Multi-Word Utensil Names ✅ COMPLETE**
- **Problem Fixed**: "chopping board", "kitchen knife", "wooden spoon" weren't mapping to canonical forms
- **Root Cause**: Missing aliases in `logic/aliases.py` caused FOL parsing errors like "Unexpected token: 'board'"
- **Solution**: Added missing aliases to logic/aliases.py:
  - "chopping board" → "choppingboard"
  - "kitchen knife" → "kitchenknife" 
  - "wooden spoon" → "woodenspoon"
- **Testing Results**:
  - ✅ "I know that chopping board is dishwasher safe" now processes correctly
  - ✅ FOL expressions like `DishwasherSafe(choppingboard)` parse successfully
  - ✅ All multi-word utensil names now work in logic assertions and checks
  - ✅ Proper contradiction detection: "Sorry, that contradicts what I already know" for conflicting rules

## What's Left to Build

### **✅ MILESTONE 9: COMPLETE - ALL CRITICAL FIXES IMPLEMENTED**

### **TASK 4: CODE QUALITY & ROBUSTNESS (Academic Scope)**
**University Module Objectives**: Demonstrate software engineering best practices for academic assessment.

#### **Phase 2: Input Validation & Sanitization (Quick Fixes)**
- [ ] Strengthen existing input validation for user queries and file paths
- [ ] Add basic safety checks to prevent common input errors
- [ ] Improve error messages to be more specific and actionable
- [ ] Keep implementation simple and practical for academic demonstration

#### **Phase 3.1: Lazy Loading Only**
- [ ] Print welcome message immediately on startup (instant user feedback)
- [ ] Load models in background while user reads welcome message
- [ ] Show loading progress/status for model initialization
- [ ] Improve user experience from 10s startup delay to instant welcome

#### **Phase 5.1: Code Organization & Cleanup**
- [ ] Extract common utilities into shared modules (reduce duplication)
- [ ] Improve inline documentation and comments for clarity
- [ ] Add type hints where helpful for code readability
- [ ] Clean up codebase structure and organization
- [ ] Demonstrate software engineering best practices

#### **Phase 6.2: Documentation Update**
- [ ] Update README and docs with Task 4 improvements
- [ ] Document the lazy loading feature and user experience improvements
- [ ] Clean up any outdated information in documentation
- [ ] Ensure all documentation reflects university module assessment scope

### **FUTURE ENHANCEMENTS (Due: Probably Never)**
**Note**: These are advanced features beyond the scope of university module assessment.
- **Enhanced YOLO Quality**: Address synthetic dataset issues for better real-world performance
- **Advanced Image Interface**: Drag-and-drop, batch processing capabilities
- **Web Interface**: Convert CLI to web-based interface for broader accessibility
- **Production Infrastructure**: Logging, monitoring, caching, configuration management
- **Advanced Testing**: Comprehensive unit tests, integration tests, performance benchmarks
- **Production Features**: Authentication, database integration, API development, deployment

## Current Status
- **✅ MILESTONE 9 CRITICAL FIXES: 5/5 COMPLETE** 
- **✅ POST-PRODUCTION DEBUG ENHANCEMENTS: 3/3 COMPLETE**
- **✅ ACADEMIC OBJECTIVES: 100% ACHIEVED** - All core functionality demonstrated with professional polish
- **✅ CORE SYSTEMS: ALL OPERATIONAL** - AIML, TF-IDF, Embedding, Logic, CNN Vision, Original YOLO
- **✅ CSV DATA QUALITY: FIXED** - Proper formatting and complete QnA responses
- **✅ LOGIC PIPELINE: SIMPLIFIED** - Clean separation from NLP, proper "Unknown." handling
- **✅ MATERIAL INFERENCE: IMPLEMENTED** - Universal quantification enables powerful FOL reasoning
- **✅ USER EXAMPLES: VALIDATED** - All startup examples use actual supported classes
- **✅ ACADEMIC INTERFACE: IMPLEMENTED** - Clean, professional user experience suitable for assessment
- **✅ DEBUG CAPABILITIES: MAINTAINED** - Complete technical information for demonstration
- **✅ LOGIC DEBUG MESSAGES: SUPPRESSED** - No debug output in production mode, full debug info in debug mode
- **✅ ERROR MESSAGES: IMPROVED** - Specific, actionable feedback for users
- **✅ MULTI-WORD UTENSILS: FIXED** - All utensil names work correctly in logic operations
- **Current Focus**: Task 4 - Code Quality & Robustness (academic scope)
- **Status**: University module assessment objectives achieved with comprehensive multi-modal AI demonstration

## Known Issues
- **No Critical Issues Remaining**: All core functionality working for academic demonstration
- **Enhanced YOLO performance**: Training metrics vs real-world performance gap (valuable academic lesson)
- **CLI-only interface**: Appropriate for university module scope (web interface beyond assessment requirements)
- **Academic Scope Limitations**: Advanced production features intentionally excluded per module requirements

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
- **CNN Architecture Evolution**: MobileNetV3 (12%) → ResNet50V2 initial (94.48%) → ResNet50V2 optimized (96.73%)
- **Parameter Optimization Methodology**: Analysis-driven conservative optimization proved most effective
- **Optimization Lesson**: Aggressive optimization counterproductive at high baseline performance
- **YOLO Enhancement Strategy**: Multi-object augmentation through intelligent scene composition rather than simple data multiplication
- **Augmentation Philosophy**: Kitchen-specific, realistic placement over generic augmentation
- **Dataset Balance**: 25% multi-object ratio provides substantial multi-object training without overwhelming single-object baseline
- **Enhanced Training Success**: Combined dataset training technically successful with strong metrics
- **Real-World Validation Critical**: Training metrics can be misleading without real-world testing
- **Synthetic Data Quality Priority**: Poor synthetic data can hurt rather than help performance
- **Domain Gap Awareness**: Training and deployment environments must be carefully aligned
- **🔄 MAJOR PRIORITY SHIFT**: From YOLO enhancement to critical bug fixes for production readiness
- **Bug Fix Methodology**: Address one issue at a time with thorough testing and simple solutions
- **Simple vs Complex Solutions**: One-line fixes often more robust than elaborate systems (Fix #3 lesson)
- **User Experience Priority**: All examples and features must work from user perspective (Fix #1, #2)
- **Data Quality Critical**: CSV formatting and parsing issues cause subtle but important problems (Fix #2)
- **Clean Separation**: Logic and NLP pipelines should remain completely distinct (Fix #3)
- **Material Inference Success**: Enhanced assert_fact() with universal quantification enables powerful FOL reasoning (Fix #4)
- **FOL Reasoning Enhancement**: Material-based rules now connect to specific utensils via inference 
- **✅ PRODUCTION ACHIEVEMENT**: All critical bug fixes completed successfully
- **Production Interface Priority**: Clean user experience essential for deployment readiness
- **Debug Capabilities Maintained**: Complete technical information preserved for development
- **Message Suppression Strategy**: Comprehensive approach to hiding verbose library output
- **Conditional Functionality**: Single flag controls development vs production features
- **Testing Methodology**: Thorough validation of both production and debug modes
- **Simple Implementation**: Focused approach with minimal code changes for maximum impact
- **✅ POST-PRODUCTION POLISH**: Logic debug message suppression and improved error messages
- **Debug Mode Integration**: Logic engine now properly respects global DEBUG_MODE flag
- **User-Friendly Error Messages**: Specific, actionable feedback replaces generic error messages
- **Multi-Word Utensil Support**: All utensil names (including spaces) work correctly in logic operations
- **Alias Completeness**: Comprehensive mapping of natural language utensil names to canonical forms 