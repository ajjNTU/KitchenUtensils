# main.py

from dataclasses import dataclass
import aiml
import os
from nlp.similarity import TfidfSimilarity
from nlp.embedding import EmbeddingSimilarity
from nlp.utils import normalize

AIML_PATH = os.path.join(os.path.dirname(__file__), 'aiml', 'utensils.aiml')
QNA_PATH = os.path.join(os.path.dirname(__file__), 'qna.csv')
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
    # TODO: Implement logic reply
    return None

def vision_reply(user_input: str) -> 'BotReply | None':
    # TODO: Implement vision reply
    return None

def main():
    classes = [
        "Blender", "Bowl", "Canopener", "Choppingboard", "Colander", "Cup", "Dinnerfork", "Dinnerknife", "Fishslice", "Garlicpress", "Kitchenknife", "Ladle", "Pan", "Peeler", "Saucepan", "Spoon", "Teaspoon", "Tongs", "Tray", "Whisk", "Woodenspoon"
    ]
    print(f"""
Welcome to the Kitchen Utensils Chatbot (Prototype) - DEBUG MODE

You can:
- Ask about kitchen utensils (e.g., 'What is a spatula?')
- Try these examples:
    * what is a spatula?
    * what is a colander?
    * what is a whisk?
- Type 'exit' or 'quit' to leave

Supported utensil classes: {', '.join(classes)}

[DEBUG MODE: Shows detailed routing decisions]
""")
    while True:
        user_input = input("> ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        user_input = normalize(user_input)
        print(f"\n🔍 DEBUG: Processing '{user_input}'")
        print("─" * 50)
        
        # Step 1: AIML
        aiml_result = aiml_reply(user_input)
        if aiml_result:
            print(f"1️⃣ AIML: Found match → {aiml_result.text}")
            print("✅ USING AIML ANSWER")
            print(aiml_result.text)
            if aiml_result.end_conversation:
                print("Goodbye!")
                break
            continue
        else:
            print(f"1️⃣ AIML: No match found for input: {user_input.upper()}")
        
        # Step 2: TF-IDF
        tfidf_result = tfidf_reply(user_input)
        tfidf_score = None
        tfidf_top_answer = None
        answer_found = False
        if tfidf_sim:
            tfidf_score = tfidf_sim.get_best_similarity_score(user_input)
            tfidf_sims = tfidf_sim.vectorizer.transform([user_input])
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
            embed_score = embed_sim.get_best_similarity_score(user_input)
            user_embedding = embed_sim.nlp(user_input).vector.reshape(1, -1)
            similarities = cosine_similarity(user_embedding, embed_sim.question_embeddings)[0]
            best_idx = similarities.argmax()
            embed_top_answer = embed_sim.answers[best_idx]
            print(f"3️⃣ Embedding: Score={embed_score:.3f}")
            print(f"   Top Embedding match → {embed_top_answer}")
            print(f"   Top Embedding score: {embed_score:.3f}")
            if not answer_found and tfidf_score is not None and tfidf_score < TFIDF_THRESHOLD:
                print(f"   TF-IDF not confident ({tfidf_score:.3f} < {TFIDF_THRESHOLD}) → Trying embedding...")
                embed_result = embed_reply(user_input)
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
            print("4️⃣ Fallback: No matches found")
            print("Sorry, I don't know that.")

if __name__ == "__main__":
    main() 