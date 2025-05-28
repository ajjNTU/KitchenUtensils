# TODO.md

## Milestone 1 – Router Skeleton
- [x] Create `main.py` CLI loop with `while True`
- [x] Add stub functions: `aiml_reply`, `tfidf_reply`, `embed_reply`, `logic_reply`, `vision_reply`
- [x] Implement routing order: AIML → TF-IDF → Embedding → fallback
- [x] Return `BotReply` dataclass with `text` + `end_conversation`

## Milestone 2 – AIML Baseline
- [ ] Make `aiml/utensils.aiml` with at least 10 utensil patterns
- [ ] Load AIML kernel in `main.py`
- [ ] Verify `What is a spatula?` returns AIML answer

## Milestone 3 – TF-IDF Similarity
- [ ] Build `nlp/similarity.py` with TF-IDF vectoriser (fit once)
- [ ] Add `qna.csv` (≥10 rows) for question/answer pairs
- [ ] In router, call TF-IDF when AIML returns `None`

## Milestone 4 – Embedding Fallback (Extra Feature)
- [ ] Load spaCy `en_core_web_md`
- [ ] Implement embedding similarity >0.65
- [ ] Cache sentence vectors for Q/A list

## Milestone 5 – Logic Engine (FOL)
- [ ] Load `logical-kb.csv` into NLTK
- [ ] Implement `assert_fact` (with contradiction check)
- [ ] Implement `check_fact` returning Correct / Incorrect / Unknown

## Milestone 6 – Fuzzy Safety
- [ ] Build `logic/fuzzy_safety.py` with Simpful rules
- [ ] Integrate safety check into `logic_reply`

## Milestone 7 – Vision Stub
- [ ] Create `image_classification/yolo_detector.py` with dummy list return
- [ ] Trigger when input contains "image" or starts with "img:"
- [ ] Replace dummy later with YOLO

## Milestone 8 – YOLO Integration
- [ ] Fine-tune YOLOv8 on utensil dataset
- [ ] Replace stub detect() with real inference (conf ≥0.4)

## Milestone 9 – Polish & Tests
- [ ] Add unit tests for similarity, logic, and vision modules
- [ ] Update `FLOW_EXAMPLES.md` with actual output
- [ ] Final README install + usage instructions 