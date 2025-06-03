# Tech Context: Kitchen Utensils Chatbot

## Technologies Used
- Python 3.10.x
- python-aiml (AIML pattern matching)
- scikit-learn (TF-IDF similarity)
- spaCy (en_core_web_md, embeddings)
- simpful (fuzzy logic)
- ultralytics, torch (YOLOv8, planned)
- pyspellchecker (input normalization)
- NLTK (FOL logic engine)

## Development Setup
- Virtual environment (.venv)
- Install dependencies: `pip install -r requirements.txt`
- Download spaCy model: `python -m spacy download en_core_web_md`

## Technical Constraints
- CLI prototype (no web or GUI yet)
- No persistent user state
- All QnA and logic data in CSV files
- Embedding cache auto-updates on QnA change

## Dependencies
- See requirements.txt for full list

## Tool Usage Patterns
- All NLP modules importable from nlp/
- Logic and fuzzy modules importable from logic/
- Image classification stub in image_classification/
- Robust FOL negation: all rules use tilde (~) for negation, not custom NotX predicates
- Canonical property parsing: multi-word properties (e.g., "microwave safe") are parsed to CamelCase (e.g., MicrowaveSafe)
- Default sharpness is now 5.0 (medium) if no explicit fact is present, for correct fuzzy demo
- Fuzzy safety routing: logic/fuzzy always routed before NLP
- Dual-path fuzzy membership: safety_score uses fuzzy memberships if available, falls back to crisp value thresholds if not
- Demo utensils for all fuzzy safety levels: kitchenknife (low), woodenspoon (high), ladle (moderate)

---

Logic engine (FOL, NLTK) and fuzzy safety (Simpful) are now fully integrated, robust, and validated for all demo utensils. Next: vision/YOLO stub. 