# Active Context: Kitchen Utensils Chatbot

**Milestone 5/6 (Logic Engine, Fuzzy Safety) complete and fully integrated. All checklist items (code, tests, dev tools, docs) are done. Next: vision/YOLO stub.**

## Current Work Focus
- Preparing for vision (YOLO) integration (Milestone 7)
- Logic and fuzzy safety (Milestone 5/6) complete and robust

## Recent Changes
- Completed Milestone 5: Logic engine (FOL, NLTK) with robust tilde (~) negation
- Completed Milestone 6: Fuzzy safety (Simpful) with sharpness/grip, dual-path membership fallback
- Canonical property parsing for multi-word properties (CamelCase)
- Default sharpness is now 5.0 (medium) for fuzzy demo utensils
- Router now integrates logic/fuzzy modules and always routes logic/fuzzy before NLP
- Demo utensils for all fuzzy safety levels: kitchenknife (low), woodenspoon (high), ladle (moderate)
- Dev tool build_kb.py for KB integrity
- Tests for logic/fuzzy in tests/test_logic.py
- Documentation updated (ARCHITECTURE.md, FLOW_EXAMPLES.md)

## Next Steps
- Implement vision stubs and prepare for YOLO integration (Milestone 7-8)
- Polish, test, and document (Milestone 9)

## Active Decisions & Considerations
- Centralized threshold management for routing
- Stateless, modular design
- All QnA and logic data in CSV for easy updates
- CLI interface for rapid prototyping
- Fuzzy safety only uses sharpness and grip; KB must provide these for demo utensils
- Fuzzy safety label is robust to Simpful API changes (dual-path logic)
- All logic/fuzzy queries are routed before NLP for clarity and demo reliability
- Demo utensils: kitchenknife (low safety), woodenspoon (high safety), ladle (moderate safety)

## Important Patterns & Preferences
- Fallback chain: AIML → TF-IDF → Embedding → Logic → Vision
- Input normalization before all processing
- Debug output for transparency and testing 