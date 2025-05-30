# Kitchen Utensils Chatbot - TODO

## Milestone 1: Router Skeleton ✅ COMPLETE
- ✅ BotReply dataclass with text and end_conversation
- ✅ Stub functions for each module (aiml_reply, tfidf_reply, embed_reply, logic_reply, vision_reply)
- ✅ Routing order: AIML → TF-IDF → Embedding → fallback
- ✅ CLI loop with fallback message

## Milestone 2: AIML Baseline ✅ COMPLETE  
- ✅ Create `aiml/utensils.aiml` with patterns for all 21 classes
- ✅ Include spaced and unspaced variants (e.g., "GARLIC PRESS" and "GARLICPRESS")
- ✅ Load AIML kernel in main.py and implement aiml_reply
- ✅ Enhanced intro message with examples and class list

## Milestone 3: TF-IDF Similarity ✅ COMPLETE
- ✅ Create `nlp/similarity.py` with TfidfSimilarity class
- ✅ Use scikit-learn TF-IDF with 0.5 similarity threshold
- ✅ Create comprehensive `qna.csv` with 200+ question variations
- ✅ Integrate into main.py as fallback after AIML

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

## Milestone 5: Logic Engine
- [ ] Create `logic/rules.py` for fuzzy logic reasoning
- [ ] Use simpful library for fuzzy sets and rules
- [ ] Handle inputs like "small cutting tool" → kitchen knife
- [ ] Integrate as fallback when similarity methods fail

## Milestone 6: Fuzzy Safety
- [ ] Add input validation and sanitization
- [ ] Handle edge cases (empty input, very long input)
- [ ] Add confidence scoring for all methods
- [ ] Graceful degradation when models fail

## Milestone 7: Vision Stub
- [ ] Create `image_classification/` module structure
- [ ] Add placeholder functions for image processing
- [ ] Prepare for YOLO integration in next milestone

## Milestone 8: YOLO Integration
- [ ] Integrate ultralytics YOLO for kitchen utensil detection
- [ ] Map YOLO classes to our 21 utensil categories
- [ ] Handle image input through CLI or API endpoint
- [ ] Return utensil identification with confidence scores

## Milestone 9: Polish & Tests
- [ ] Add comprehensive unit tests
- [ ] Performance optimization
- [ ] Documentation improvements
- [ ] Error handling improvements
- [ ] Add logging and metrics

---

**Current Status:** Milestone 4 complete. The system now has:
- AIML pattern matching for exact queries
- TF-IDF similarity for token-based matching  
- spaCy embedding fallback for semantic similarity
- Intelligent routing with confidence thresholds 