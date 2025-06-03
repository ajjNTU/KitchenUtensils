# Project Brief: Kitchen Utensils Chatbot

## Purpose
A conversational assistant for adult social care users that answers questions about kitchen utensils, performs logical and fuzzy reasoning, and identifies utensils from photos.

## Core Requirements
- Natural conversation (AIML, semantic matching)
- Robust fallback: AIML → TF-IDF → Embedding (spaCy)
- Input normalization (case, punctuation, contractions, spelling correction)
- Extensive QnA coverage for all utensil classes
- Rule-based and fuzzy logic reasoning
- Multi-object utensil detection (YOLOv8)

## Goals
- High accuracy for utensil-related queries
- Clear, concise, and stateless responses
- Support for both text and image input
- Modular, testable, and extensible architecture

## Scope
- 21 kitchen utensil classes (see README)
- CLI interface (prototype)
- No user authentication or persistent user state
- No recipe or general cooking advice (utensils only) 

---

Milestone 5/6 (Logic Engine, Fuzzy Safety) is now complete and fully integrated, including tests, dev tools, and documentation. Next: vision/YOLO stub. 