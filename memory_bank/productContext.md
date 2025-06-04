# Product Context: Kitchen Utensils Chatbot

## Why This Project Exists
Many adult social care users struggle to identify, select, or safely use kitchen utensils. Existing resources are not conversational, lack semantic understanding, and do not support image-based queries.

## Problems Solved
- Provides clear, accessible answers about utensils
- Handles varied phrasing and misspellings
- Offers robust logic and fuzzy safety reasoning (FOL negation, canonical property parsing, default sharpness, robust fuzzy routing, dual-path fuzzy membership fallback)
- Identifies utensils from images (in development)
- Demo utensils validated for all fuzzy safety levels (kitchenknife, woodenspoon, ladle)

## User Experience Goals
- Fast, accurate, and friendly responses
- Stateless, concise, and easy to understand
- Robust to spelling, phrasing, and input errors
- Seamless fallback between modules
- Support for both text and image queries 
- Robust logic/fuzzy reasoning for all demo utensils

---

Logic and fuzzy safety (Milestone 5/6) are now complete, robust, and fully integrated, including tests, dev tools, and documentation. Next: vision/YOLO stub.

Vision capabilities: Dataset preparation complete for both CNN classification and YOLO detection. Ready for model training and integration (Milestone 7-8). Proper dataset citation added to project documentation. 