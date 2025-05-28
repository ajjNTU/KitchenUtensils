# Architecture Overview

## High-level Flow

User Input → Router → [AIML / NLP / Logic / Image Classification] → Response

- If all modules return None, router prints "Sorry, I don't know that."

## Modules

- **nlp/**: Natural language processing (similarity matching, embeddings)
- **logic/**: Logic reasoning (FOL, fuzzy safety)
- **image_classification/**: Utensil image detection and classification
- **main.py**: Chatbot interface and router

## Data
- `qna.csv`: Q/A pairs for similarity and semantic matching
- `logical-kb.csv`: Knowledge base for logic reasoning

## Planned API
- `nlp.similarity.reply(text, context)`
- `logic.logic_engine.reply(text, context)`
- `image_classification.yolo_detector.detect(image_path)`

## TODO
- [ ] Implement command router in main.py
- [ ] Set up initial AIML patterns
- [ ] Prepare demo images and Q/A samples 