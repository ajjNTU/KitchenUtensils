# main.py

from dataclasses import dataclass
import aiml
import os
from nlp.similarity import TfidfSimilarity
from nlp.embedding import EmbeddingSimilarity
from nlp.utils import normalize
from logic import check_fact, assert_fact, get_fuzzy_safety_reply
from image_classification import CNNClassifier

AIML_PATH = os.path.join(os.path.dirname(__file__), 'aiml', 'utensils.aiml')
QNA_PATH = os.path.join(os.path.dirname(__file__), 'qna.csv')
CNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'image_classification', 'cnn_model.h5')
TFIDF_THRESHOLD = 0.65

@dataclass
class BotReply:
    text: str
    end_conversation: bool = False

# Initialize AIML kernel at startup
aiml_kernel = aiml.Kernel()
if os.path.exists(AIML_PATH):
    aiml_kernel.learn(AIML_PATH)
else:
    print(f"Warning: {AIML_PATH} not found. AIML replies will not work.")

# Initialize TF-IDF similarity at startup
tfidf_sim = None
try:
    tfidf_sim = TfidfSimilarity(QNA_PATH)
except Exception as e:
    print(f"Warning: Could not load TF-IDF similarity: {e}")

# Initialize embedding similarity at startup
embed_sim = None
try:
    embed_sim = EmbeddingSimilarity(QNA_PATH)
except Exception as e:
    print(f"Warning: Could not load embedding similarity: {e}")

# Initialize CNN classifier at startup
cnn_classifier = None
try:
    if os.path.exists(CNN_MODEL_PATH):
        cnn_classifier = CNNClassifier(model_path=CNN_MODEL_PATH)
        print("CNN classifier loaded successfully")
    else:
        print(f"Warning: CNN model not found at {CNN_MODEL_PATH}. Vision features will not work.")
except Exception as e:
    print(f"Warning: Could not load CNN classifier: {e}")

def aiml_reply(user_input: str) -> 'BotReply | None':
    if not aiml_kernel.numCategories():
        return None
    response = aiml_kernel.respond(user_input.upper())
    if response:
        return BotReply(text=response)
    return None

def tfidf_reply(user_input: str) -> 'BotReply | None':
    if tfidf_sim is None:
        return None
    response = tfidf_sim.reply(user_input)
    if response:
        return BotReply(text=response)
    return None

def embed_reply(user_input: str) -> 'BotReply | None':
    if embed_sim is None or tfidf_sim is None:
        return None
    
    # Only use embedding if TF-IDF confidence is < 0.6
    tfidf_score = tfidf_sim.get_best_similarity_score(user_input)
    if tfidf_score >= 0.6:
        return None
    
    # Use embedding similarity with threshold ≥ 0.6
    response = embed_sim.reply(user_input, threshold=0.6)
    if response:
        return BotReply(text=response)
    return None

def logic_reply(user_input: str) -> 'BotReply | None':
    # Route to logic if query is a fact assertion, fact check, or fuzzy safety query
    text = user_input.strip().lower()
    
    # Try fuzzy safety query first
    # Heuristic: contains "safe" and "is" or starts with "how safe is"
    if ("safe" in text and "is" in text) or text.startswith("how safe is") :
        fuzzy_result = get_fuzzy_safety_reply(user_input) # Pass original for better parsing
        if fuzzy_result:
            return BotReply(text=fuzzy_result)
    
    # Then try fact check or assertion
    try:
        if text.startswith('check that '):
            result = check_fact(user_input)
            if result and result != "Unknown.":
                return BotReply(text=result)
        elif text.startswith('i know that '):
            result = assert_fact(user_input)
            if result:
                return BotReply(text=result)
    except Exception:
        pass # Should ideally log this error
    return None

def vision_reply(image_path: str) -> 'BotReply | None':
    """
    Classify kitchen utensil from image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        BotReply with classification results or None if failed
    """
    if cnn_classifier is None:
        return BotReply(text="Sorry, vision classification is not available. CNN model not loaded.")
    
    if not os.path.exists(image_path):
        return BotReply(text=f"Sorry, I cannot find the image file: {image_path}")
    
    try:
        # Get top 3 predictions
        predictions = cnn_classifier.predict(image_path, top_k=3)
        
        if not predictions:
            return BotReply(text="Sorry, I couldn't classify this image.")
        
        # Format response
        top_class, top_confidence = predictions[0]
        
        if top_confidence < 0.1:  # Very low confidence
            response = "I'm not confident about this image. It might be a kitchen utensil, but I can't identify it clearly."
        elif top_confidence < 0.3:  # Low confidence
            response = f"I think this might be a {top_class.lower()}, but I'm not very confident (confidence: {top_confidence:.1%})."
        else:  # Reasonable confidence
            response = f"I can see a {top_class.lower()} in this image (confidence: {top_confidence:.1%})."
        
        # Add alternative predictions if they're reasonably close
        if len(predictions) > 1:
            second_class, second_confidence = predictions[1]
            if second_confidence > 0.1:  # Only show if reasonably confident
                response += f" It could also be a {second_class.lower()} (confidence: {second_confidence:.1%})."
        
        return BotReply(text=response)
        
    except Exception as e:
        return BotReply(text=f"Sorry, I encountered an error while analyzing the image: {str(e)}")

def main():
    classes = [
        "Blender", "Bowl", "Canopener", "Choppingboard", "Colander", "Cup", "Dinnerfork", "Dinnerknife", "Fishslice", "Garlicpress", "Kitchenknife", "Ladle", "Pan", "Peeler", "Saucepan", "Spoon", "Teaspoon", "Tongs", "Tray", "Whisk", "Woodenspoon"
    ]
    print(f"""
Welcome to the Kitchen Utensils Chatbot (Prototype) - DEBUG MODE

You can:
- Ask about kitchen utensils (e.g., 'What is a spatula?')
- Check facts about utensils (e.g., 'Check that tongs are microwave safe')
- Tell the chatbot facts (e.g., 'I know that a tray is metal')
- Ask about utensil safety (e.g., 'Is a kitchen knife safe for children?')
- Identify utensils from images (e.g., 'image: path/to/image.jpg')
- Type 'exit' or 'quit' to leave

Supported utensil classes:
{', '.join(classes[:8])}
{', '.join(classes[8:16])}
{', '.join(classes[16:])}
Supported fact properties: Metal, Plastic, Wood, Ceramic, Sharp, MicrowaveSafe, OvenSafe, DishwasherSafe, ChildSafe, RequiresCaution, etc.

[DEBUG MODE: Shows detailed routing decisions and logic engine steps]
""")
    while True:
        user_input_original = input("> ")
        if user_input_original.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        
        # Check if input is an image path
        if user_input_original.lower().startswith("image:"):
            image_path = user_input_original[6:].strip()
            print(f"\n🖼️ DEBUG: Processing image: {image_path}")
            print("─" * 50)
            
            vision_result = vision_reply(image_path)
            if vision_result:
                print(f"4️⃣ Vision: {vision_result.text}")
                print("✅ USING VISION ANSWER")
                print(vision_result.text)
            else:
                print("4️⃣ Vision: Failed to process image")
                print("Sorry, I couldn't process that image.")
            continue
        
        user_input_normalized = normalize(user_input_original)
        print("\n🔍 DEBUG: Processing normalized input: " + user_input_normalized + ", Original input: " + user_input_original)
        print("─" * 50)

        # Step 0: Try Logic/Fuzzy Module first
        # Pass original user input to logic_reply as it might need it for better parsing in get_fuzzy_safety_reply
        logic_result_obj = logic_reply(user_input_original) 
        if logic_result_obj:
            print(f"0️⃣ Logic/Fuzzy: {logic_result_obj.text}")
            print("✅ USING LOGIC/FUZZY ANSWER")
            print(logic_result_obj.text)
            continue # Skip to next input
        else:
            print(f"0️⃣ Logic/Fuzzy: No direct logic/fuzzy answer found. Proceeding to NLP pipeline...")

        # Step 1: AIML (using normalized input)
        aiml_result = aiml_reply(user_input_normalized)
        if aiml_result:
            print(f"1️⃣ AIML: Found match → {aiml_result.text}")
            print("✅ USING AIML ANSWER")
            print(aiml_result.text)
            if aiml_result.end_conversation:
                print("Goodbye!")
                break
            continue
        else:
            print(f"1️⃣ AIML: No match found for input: {user_input_normalized.upper()}")

        # Step 2: TF-IDF (using normalized input)
        tfidf_result = tfidf_reply(user_input_normalized)
        tfidf_score = None
        tfidf_top_answer = None
        answer_found = False
        if tfidf_sim:
            tfidf_score = tfidf_sim.get_best_similarity_score(user_input_normalized)
            tfidf_sims = tfidf_sim.vectorizer.transform([user_input_normalized])
            from sklearn.metrics.pairwise import cosine_similarity
            all_sims = cosine_similarity(tfidf_sims, tfidf_sim.question_vecs)[0]
            best_idx = all_sims.argmax()
            tfidf_top_answer = tfidf_sim.answers[best_idx]
            print(f"2️⃣ TF-IDF: Score={tfidf_score:.3f} (threshold={TFIDF_THRESHOLD})")
            print(f"   Top TF-IDF match → {tfidf_top_answer}")
            print(f"   Top TF-IDF score: {tfidf_score:.3f}")
            if tfidf_result:
                print(f"   Found match → {tfidf_result.text}")
                print("✅ USING TF-IDF ANSWER")
                print(tfidf_result.text)
                answer_found = True
        else:
            print("2️⃣ TF-IDF: Not available")
        
        # Step 3: Embedding (always show top candidate and score)
        embed_score = None
        embed_top_answer = None
        if embed_sim and tfidf_sim:
            embed_score = embed_sim.get_best_similarity_score(user_input_normalized)
            user_embedding = embed_sim.nlp(user_input_normalized).vector.reshape(1, -1)
            similarities = cosine_similarity(user_embedding, embed_sim.question_embeddings)[0]
            best_idx = similarities.argmax()
            embed_top_answer = embed_sim.answers[best_idx]
            print(f"3️⃣ Embedding: Score={embed_score:.3f}")
            print(f"   Top Embedding match → {embed_top_answer}")
            print(f"   Top Embedding score: {embed_score:.3f}")
            if not answer_found and tfidf_score is not None and tfidf_score < TFIDF_THRESHOLD:
                print(f"   TF-IDF not confident ({tfidf_score:.3f} < {TFIDF_THRESHOLD}) → Trying embedding...")
                embed_result = embed_reply(user_input_normalized) # Ensure this uses normalized
                if embed_result:
                    print(f"   Found match (≥0.6) → {embed_result.text}")
                    print("✅ USING EMBEDDING ANSWER")
                    print(embed_result.text)
                    answer_found = True
                else:
                    print(f"   No match (score < 0.6)")
            elif tfidf_score is not None:
                print(f"   TF-IDF confident ({tfidf_score:.3f} ≥ {TFIDF_THRESHOLD}) → Skipping embedding")
        else:
            print("3️⃣ Embedding: Not available")
        
        # Step 4: Fallback
        if not answer_found:
            print("5️⃣ Fallback: No matches found")
            print("Sorry, I don't know that.")

if __name__ == "__main__":
    main() 