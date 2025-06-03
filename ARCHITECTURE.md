# Architecture Overview

## High-level Flow

User Input → Normalize → Router → [AIML → TF-IDF → Embedding → Logic → Vision] → Response

- If all modules return None, router prints "Sorry, I don't know that."

## Modules

- **nlp/**: Natural language processing (input normalization, similarity matching, embeddings)
- **logic/**: Logic reasoning (FOL, fuzzy safety)
- **image_classification/**: Utensil image detection and classification
- **main.py**: Chatbot interface and router (centralized threshold management)

## Data
- `qna.csv`: Q/A pairs for similarity and semantic matching (includes generic and specific questions for all utensils, with paraphrases and common misspellings)
- `logical-kb.csv`: Knowledge base for logic reasoning

## Fallback Logic
- **AIML:** Exact pattern match (e.g., "What is a colander?")
- **TF-IDF:** Token-based similarity (threshold 0.65, centralized)
- **Embedding (spaCy):** Semantic similarity (threshold 0.6)
- **Input normalization** is applied before all steps (lowercase, punctuation, contractions, spelling correction)
- If all fail, fallback message is shown

## Planned API
- `nlp.similarity.reply(text, context)`
- `logic.logic_engine.reply(text, context)`
- `image_classification.yolo_detector.detect(image_path)`

### Image Classification Classes
The YOLOv8 model is trained to detect the following utensil classes:
- Blender
- Bowl
- Canopener
- Choppingboard
- Colander
- Cup
- Dinnerfork
- Dinnerknife
- Fishslice
- Garlicpress
- Kitchenknife
- Ladle
- Pan
- Peeler
- Saucepan
- Spoon
- Teaspoon
- Tongs
- Tray
- Whisk
- Woodenspoon

## Fuzzy Logic Safety (Simpful)

```mermaid
flowchart TD
    Sharpness[Sharpness (0-10)] --> Fuzzy["Fuzzy System (Simpful)"]
    Grip[Grip (0-10)] --> Fuzzy
    Fuzzy -->|Rules| Safety[Safety (low/moderate/high)]
```

- Inputs: Sharpness, Grip (0–10 scale)
- Output: Safety (low, moderate, high)
- Rules: e.g., "IF sharpness IS high AND grip IS poor THEN safety IS low"
- Used for queries like: "Is a peeler safe for children?" 