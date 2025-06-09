# Kitchen Utensils Chatbot

![Python](https://img.shields.io/badge/python-3.10-blue)
![University](https://img.shields.io/badge/scope-university%20module-green)

**University Module Assessment Project**

A conversational assistant that demonstrates multi-modal AI integration through kitchen utensil queries, logical reasoning, and image recognition. This project showcases AI/ML implementation skills and software engineering best practices for academic assessment.

**Project Scope:** University module assessment - demonstrates technical competency in AI/ML integration and software engineering practices. Not intended for production deployment.

**Core Technologies Demonstrated:**
- **Natural Language Processing**: AIML pattern matching, TF-IDF similarity, semantic embeddings (spaCy)
- **Computer Vision**: CNN classification (96.73% accuracy), YOLO object detection (97.2% mAP50)
- **Logic & Fuzzy Reasoning**: First-order logic with material inference, fuzzy safety assessment
- **Software Engineering**: Modular architecture, error handling, graceful degradation, clean user interface

**Academic Learning Objectives:**
- Multi-modal AI pipeline integration (NLP + Computer Vision + Logic)
- Robust fallback mechanisms and error handling
- User experience design (production vs debug modes)
- Code quality and maintainability principles

**Assessment:** ISYS37101 Artificial Intelligence for Data Science Coursework

## Quick Start
```bash
# Production mode (clean interface)
python main.py

# Debug mode (technical details)
python main.py --debug
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
- I know that wood is microwave safe (logic assertion)
- Check that woodenspoon is microwave safe (logic inference)
- How safe is a kitchenknife? (fuzzy reasoning)
- What is in this image? (CNN classification)
- Detect everything in this image (YOLO detection)

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
- image_classification/: Vision modules (CNN, YOLO)
- aiml/: AIML patterns
- memory_bank/: Project documentation and context
- tests/: All unit and integration tests
- scripts/: Development and debugging utilities

## Academic Achievement Status

**✅ All Core Objectives Met:**
- Multi-modal AI pipeline integration demonstrated
- Robust error handling and graceful degradation implemented
- Clean user interface with comprehensive debug capabilities
- Software engineering best practices applied throughout
- Comprehensive documentation and technical insights

**Current Phase:** Task 4 - Code Quality & Robustness improvements to further demonstrate software engineering principles.

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