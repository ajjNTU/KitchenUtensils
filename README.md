# Kitchen Utensils Chatbot

![Python](https://img.shields.io/badge/python-3.10-blue)

A conversational assistant for adult social care users that can answer questions about kitchen utensils, perform logical and fuzzy reasoning, and identify utensils from photos.

**Features:**
- Natural conversation (AIML, semantic matching)
- Robust fallback system: AIML → TF-IDF → Embedding (spaCy)
- Input normalization (case, punctuation, contractions, spelling correction)
- Extensive QnA coverage with generic and specific utensil questions
- Rule-based and fuzzy logic reasoning
- Multi-object utensil detection (YOLOv8)

**Assessment:** ISYS37101 Artificial Intelligence for Data Science Coursework

## Quick Start
```bash
python main.py
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

**Tested on Python 3.10.x**

## Example Queries
- What is a wood spoon?
- What can I use a knife for?
- What drains pasta?
- What does a colander do?
- How do you use a blender?

## Supported Utensil Classes (Image Recognition)
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

## Project Structure
- main.py: Chatbot CLI and router
- nlp/: NLP modules (similarity, embedding, normalization)
- logic/: Logic engine, fuzzy safety, aliases, KB
- image_classification/: Vision module (stub)
- aiml/: AIML patterns
- memory_bank/: Project documentation and context
- tests/: All unit and integration tests
- scripts/: Dev and debug scripts (e.g., debug_embeddings.py, demo_milestone4.py, test_char_sim.py, test_word_overlap.py)

All dev/test scripts are now organized in scripts/ or tests/.

## Dataset

This project uses the **Kitchen Utensils Dataset** from Roboflow for training and testing image classification models.

**Citation:**
```
Kitchen Utensils Dataset. Roboflow Universe. 
Available at: https://universe.roboflow.com/utensils/utensils-wp5hm
Accessed: 04/06/2025
```

The dataset includes:
- 21 kitchen utensil classes with bounding box annotations
- Train/validation/test splits for both YOLO detection and CNN classification
- Over 1000+ annotated images across all classes

## Acknowledgments

- **Roboflow** for providing the Kitchen Utensils dataset
- **spaCy** team for the en_core_web_md language model
- **Ultralytics** for YOLOv8 framework
- **scikit-learn** and **NLTK** for machine learning and NLP capabilities