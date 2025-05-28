# FLOW_EXAMPLES.md

> Bot replies ≤ 60 chars unless image list.

## User Flows & Demo Scenarios

Demonstrates how a user interacts with the Kitchen Utensils Chatbot, covering all core features and "first-class" extra enhancements (semantic matching, fuzzy logic, YOLO).

---

### 1. Direct AIML Pattern Match

| Step | User Input               | Module | Bot Reply                                                        |
|------|--------------------------|--------|------------------------------------------------------------------|
| 1    | What is a spatula?       | AIML   | A spatula is a flat utensil used for flipping or spreading food. |

---

### 2. Q/A Similarity Matching (TF-IDF, then Embedding)

| Step | User Input                        | Module             | Bot Reply                                                        |
|------|-----------------------------------|--------------------|------------------------------------------------------------------|
| 1    | Which tool flips pancakes easily? | TF-IDF / Embedding | A spatula is ideal for flipping pancakes and other flat foods.   |
| 2    | Tell me about a utensil for draining pasta. | TF-IDF / Embedding | A colander is a bowl-shaped strainer used for draining pasta and vegetables. |

---

### 3. Logic Reasoning (First Order Logic)

| Step | User Input                           | Module | Bot Reply                                                   |
|------|--------------------------------------|--------|-------------------------------------------------------------|
| 1    | Check that knife is safe for children | Logic  | Incorrect. Knives are rated unsafe for children.            |
| 2    | I know that garlic press is metal.    | Logic  | OK, I will remember that garlic press is metal.             |

---

### 4. Fuzzy Logic Reasoning

| Step | User Input                                         | Module | Bot Reply                                                                   |
|------|----------------------------------------------------|--------|-----------------------------------------------------------------------------|
| 1    | Is a butter knife safe for elderly users?          | Fuzzy  | Moderately safe. Butter knives have low sharpness and a comfortable grip.   |
| 2    | Is a ceramic peeler very safe?                     | Fuzzy  | Ceramic peelers are considered moderately safe due to sharp blades.         |

---

### 5. Image Recognition (YOLO Multi-object Detection)

| Step | User Input                                            | Module            | Bot Reply                                            |
|------|-------------------------------------------------------|-------------------|------------------------------------------------------|
| 1    | What utensils are in this image? [uploads photo]      | Image classifier  | I see: knife, peeler, and whisk.                    |
| 2    | Can you identify these? [uploads photo of utensil tray]| Image classifier  | I see: ladle, spatula, and tongs.                   |

---

### 6. Integrated (End-to-End) Example

| Step | User Input                         | Module            | Bot Reply                                                         |
|------|------------------------------------|-------------------|-------------------------------------------------------------------|
| 1    | What is a colander?                | AIML              | A colander is a bowl-shaped strainer used to drain foods.         |
| 2    | Is a colander sharp?               | Logic             | Incorrect. Colanders are not sharp utensils.                      |
| 3    | Which utensil is best for serving soup? | TF-IDF / Embedding | A ladle is the best utensil for serving soup.                     |
| 4    | What do you see in this image? [photo] | Image classifier  | I see: colander, ladle, and serving spoon.                        |

---

### 7. Fallback / Unhandled Query

| Step | User Input                | Module | Bot Reply                                 |
|------|---------------------------|--------|-------------------------------------------|
| 1    | How do I make pancakes?   | None   | Sorry, I can only answer questions about kitchen utensils. |

---

**Note:**  
- All responses are concise and stateless (no tracking of previous turns).
- Flows reflect only features implemented (semantic matching, fuzzy logic, no co-reference).

--- 