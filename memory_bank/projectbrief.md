# Project Brief: Kitchen Utensils Chatbot

## Project Scope: University Module Assessment
**IMPORTANT**: This project is being developed for a university module assessment to demonstrate AI/ML implementation skills and software engineering best practices. The scope is intentionally limited to academic requirements - full production deployment is not the objective.

## Purpose
A conversational assistant for adult social care users that answers questions about kitchen utensils, performs logical and fuzzy reasoning, and identifies utensils from photos. This demonstrates multi-modal AI pipeline integration and robust software engineering practices.

## Core Requirements
- Natural conversation (AIML, semantic matching)
- Robust fallback: AIML → TF-IDF → Embedding (spaCy)
- Input normalization (case, punctuation, contractions, spelling correction)
- Extensive QnA coverage for all utensil classes
- Rule-based and fuzzy logic reasoning
- Multi-object utensil detection (YOLOv8)

## Academic Learning Goals
- Multi-modal AI system integration (NLP + Computer Vision + Logic)
- Software engineering best practices (modular design, testing, documentation)
- Error handling and graceful degradation
- User experience design (production vs debug modes)
- Code quality and maintainability principles

## Scope
- 21 kitchen utensil classes (see README)
- CLI interface (prototype for academic demonstration)
- No user authentication or persistent user state
- No recipe or general cooking advice (utensils only)
- **Academic Focus**: Demonstrate technical competency, not production deployment

---

✅ **Milestone 9 (Critical Bug Fixes) COMPLETE**: All 5 production-blocking issues resolved.
✅ **Post-Production Debug Enhancements COMPLETE**: All 3 debug improvements implemented.

**CURRENT STATUS**: System demonstrates complete multi-modal AI functionality with clean user interface, comprehensive debug capabilities, improved error messages, and polished user experience suitable for academic assessment.

**NEXT PHASE**: Task 4 - Code Quality & Robustness (simplified plan for university module scope):
- Phase 2: Input Validation & Sanitization (Quick Fixes)
- Phase 3.1: Lazy Loading Only (improved user experience)
- Phase 5.1: Code Organization & Cleanup (software engineering best practices)
- Phase 6.2: Documentation Update (academic presentation quality) 