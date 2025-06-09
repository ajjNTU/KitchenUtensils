# main.py

from dataclasses import dataclass
import aiml
import os
import torch
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Kitchen Utensils Chatbot')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed routing information')
args = parser.parse_args()

# Global debug flag
DEBUG_MODE = args.debug

# Suppress verbose library messages unless in debug mode
if not DEBUG_MODE:
    import warnings
    import logging
    
    # Suppress TensorFlow messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
    
    # Suppress TensorFlow warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Suppress TensorFlow logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    # Suppress YOLO/Ultralytics verbose output
    os.environ['YOLO_VERBOSE'] = 'False'
    
    # Suppress simpful messages (if any)
    logging.getLogger('simpful').setLevel(logging.ERROR)

# PyTorch 2.6 compatibility fix
original_load = torch.load
def legacy_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = legacy_load

from nlp.similarity import TfidfSimilarity
from nlp.embedding import EmbeddingSimilarity
from nlp.utils import normalize

# Import logic module with conditional Simpful banner suppression
if DEBUG_MODE:
    from logic import check_fact, assert_fact, get_fuzzy_safety_reply, set_debug_mode
else:
    # Suppress Simpful banner in production mode
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        from logic import check_fact, assert_fact, get_fuzzy_safety_reply, set_debug_mode
    finally:
        sys.stdout = old_stdout

# Set debug mode for logic engine
set_debug_mode(DEBUG_MODE)

from image_classification import CNNClassifier
from image_classification.yolo_detector import YOLODetector

AIML_PATH = os.path.join(os.path.dirname(__file__), 'aiml', 'utensils.aiml')
QNA_PATH = os.path.join(os.path.dirname(__file__), 'qna.csv')
CNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'image_classification', 'cnn_model.h5')
YOLO_DATA_YAML = os.path.join(os.path.dirname(__file__), 'image_classification', 'utensils-wp5hm-yolo8', 'data.yaml')
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'train', 'weights', 'best.pt')
TFIDF_THRESHOLD = 0.65

@dataclass
class BotReply:
    text: str
    end_conversation: bool = False

# Initialize AIML kernel at startup
aiml_kernel = aiml.Kernel()
if os.path.exists(AIML_PATH):
    if DEBUG_MODE:
        aiml_kernel.learn(AIML_PATH)
    else:
        # Suppress AIML loading messages in production mode
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            aiml_kernel.learn(AIML_PATH)
        finally:
            sys.stdout = old_stdout
else:
    if DEBUG_MODE:
        print(f"Warning: {AIML_PATH} not found. AIML replies will not work.")

# Initialize TF-IDF similarity at startup
tfidf_sim = None
try:
    tfidf_sim = TfidfSimilarity(QNA_PATH)
except Exception as e:
    if DEBUG_MODE:
        print(f"Warning: Could not load TF-IDF similarity: {e}")

# Initialize embedding similarity at startup
embed_sim = None
try:
    if DEBUG_MODE:
        embed_sim = EmbeddingSimilarity(QNA_PATH)
    else:
        # Suppress embedding loading messages in production mode
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            embed_sim = EmbeddingSimilarity(QNA_PATH)
        finally:
            sys.stdout = old_stdout
except Exception as e:
    if DEBUG_MODE:
        print(f"Warning: Could not load embedding similarity: {e}")

# Initialize CNN classifier at startup with graceful degradation
cnn_classifier = None
cnn_available = False
try:
    if os.path.exists(CNN_MODEL_PATH):
        cnn_classifier = CNNClassifier(model_path=CNN_MODEL_PATH)
        cnn_available = True
        if DEBUG_MODE:
            print("✅ CNN classifier loaded successfully")
        else:
            print("CNN classifier loaded successfully")
    else:
        if DEBUG_MODE:
            print(f"⚠️  Warning: CNN model not found at {CNN_MODEL_PATH}")
            print("   CNN vision features will be disabled")
        else:
            print("Note: CNN classification not available (model not found)")
except Exception as e:
    if DEBUG_MODE:
        print(f"❌ Error loading CNN classifier: {e}")
        print("   CNN vision features will be disabled")
    else:
        print("Note: CNN classification not available (loading failed)")
    cnn_classifier = None
    cnn_available = False

# Initialize YOLO detector at startup with graceful degradation
yolo_detector = None
yolo_available = False
try:
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_detector = YOLODetector(model_path=YOLO_MODEL_PATH, data_yaml_path=YOLO_DATA_YAML)
        yolo_available = True
        if DEBUG_MODE:
            print("✅ YOLOv8 detector loaded successfully (trained model)")
        else:
            print("YOLOv8 detector loaded successfully (trained model)")
    elif os.path.exists(YOLO_DATA_YAML):
        yolo_detector = YOLODetector(data_yaml_path=YOLO_DATA_YAML)
        yolo_available = True
        if DEBUG_MODE:
            print("✅ YOLOv8 detector loaded with pretrained model (not trained on utensils yet)")
        else:
            print("YOLOv8 detector loaded with pretrained model (not trained on utensils yet)")
    else:
        if DEBUG_MODE:
            print(f"⚠️  Warning: YOLO data config not found at {YOLO_DATA_YAML}")
            print("   YOLO detection features will be disabled")
        else:
            print("Note: YOLO detection not available (config not found)")
except Exception as e:
    if DEBUG_MODE:
        print(f"❌ Error loading YOLO detector: {e}")
        print("   YOLO detection features will be disabled")
    else:
        print("Note: YOLO detection not available (loading failed)")
    yolo_detector = None
    yolo_available = False

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
    """
    Logic/Fuzzy Pipeline - Completely separate from NLP pipeline.
    
    Handles:
    - Fuzzy safety queries: "Is X safe for children?"
    - Fact checking: "Check that X is Y" → Returns "Correct.", "Incorrect.", or "Unknown."
    - Fact assertions: "I know that X is Y" → Adds facts with material inference
    
    Returns:
    - BotReply with result (including "Unknown.") - processing stops here
    - None if input doesn't match logic patterns → continues to NLP pipeline
    
    NO NLP FALLBACK: Logic results (including "Unknown.") do not fall through to NLP.
    """
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
            if result:
                return BotReply(text=result)  # Return any result including "Unknown."
        elif text.startswith('i know that '):
            result = assert_fact(user_input)
            if result:
                return BotReply(text=result)
    except Exception:
        pass # Should ideally log this error
    return None

def get_image_query_type(user_input: str) -> str:
    """
    Check what type of image analysis is requested.
    
    Args:
        user_input: Raw user input string
        
    Returns:
        "cnn" for CNN classification, "yolo" for YOLO detection, "none" for no image query
    """
    text = user_input.lower().strip()
    
    if text == "what is in this image?":
        return "cnn"
    elif text == "detect everything in this image":
        return "yolo"
    else:
        return "none"

def prompt_for_image_path() -> str:
    """
    Open a file dialog box to select an image file.
    
    Returns:
        Selected image file path, or empty string if cancelled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Create a root window and hide it
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # Bring dialog to front
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        # Clean up
        root.destroy()
        
        return file_path if file_path else ""
        
    except ImportError:
        # Fallback to text input if tkinter is not available
        print("\n📁 File dialog not available. Please enter the image file path:")
        image_path = input("📸 Image path: ").strip()
        return image_path if image_path.lower() != 'cancel' else ""
    except Exception as e:
        print(f"Error opening file dialog: {e}")
        print("📁 Please enter the image file path:")
        image_path = input("📸 Image path: ").strip()
        return image_path if image_path.lower() != 'cancel' else ""

def vision_reply(image_path: str, mode: str = "both") -> 'BotReply | None':
    """
    Analyze kitchen utensils from image using CNN and/or YOLO with graceful degradation.
    
    Args:
        image_path: Path to image file
        mode: "cnn" for CNN only, "yolo" for YOLO only, "both" for both models
        
    Returns:
        BotReply with analysis results or None if completely failed
    """
    # Check if image file exists
    if not os.path.exists(image_path):
        return BotReply(text=f"Sorry, I cannot find the image file: {image_path}")
    
    # Check if any vision models are available
    if not cnn_available and not yolo_available:
        return BotReply(text="Sorry, no vision models are currently available for image analysis. The system is running in text-only mode.")
    
    # Adjust mode based on available models
    original_mode = mode
    if mode == "cnn" and not cnn_available:
        if yolo_available:
            mode = "yolo"
            if DEBUG_MODE:
                print("⚠️  CNN not available, switching to YOLO detection")
        else:
            return BotReply(text="Sorry, CNN classification is not available and no alternative vision models are loaded.")
    elif mode == "yolo" and not yolo_available:
        if cnn_available:
            mode = "cnn"
            if DEBUG_MODE:
                print("⚠️  YOLO not available, switching to CNN classification")
        else:
            return BotReply(text="Sorry, YOLO detection is not available and no alternative vision models are loaded.")
    elif mode == "both":
        if not cnn_available and not yolo_available:
            return BotReply(text="Sorry, no vision models are currently available for image analysis.")
        elif not cnn_available:
            mode = "yolo"
            if DEBUG_MODE:
                print("⚠️  CNN not available, using YOLO only")
        elif not yolo_available:
            mode = "cnn"
            if DEBUG_MODE:
                print("⚠️  YOLO not available, using CNN only")
    
    responses = []
    
    # Try YOLO detection (multi-object detection with annotations)
    if mode in ["yolo", "both"] and yolo_available and yolo_detector is not None:
        try:
            if DEBUG_MODE:
                print("🔍 Running YOLO object detection...")
            detections, annotated_path = yolo_detector.predict_and_display(
                image_path, 
                conf_threshold=0.25,
                save_annotated=True,
                show_plot=True  # Display the annotated image
            )
            
            if detections:
                yolo_response = f"YOLO Detection found {len(detections)} objects:\n"
                for i, det in enumerate(detections, 1):
                    yolo_response += f"  {i}. {det['class']} (confidence: {det['confidence']:.1%})\n"
                yolo_response += f"\n📁 Annotated image saved to: {annotated_path}"
                responses.append(("YOLO", yolo_response))
            else:
                responses.append(("YOLO", "YOLO detection found no objects above confidence threshold."))
        except Exception as e:
            error_msg = f"YOLO detection encountered an error: {str(e)}"
            if DEBUG_MODE:
                print(f"❌ YOLO Error: {e}")
            responses.append(("YOLO", error_msg))
    
    # Try CNN classification (single-object classification)
    if mode in ["cnn", "both"] and cnn_available and cnn_classifier is not None:
        try:
            if DEBUG_MODE:
                print("🔍 Running CNN classification...")
            predictions = cnn_classifier.predict(image_path, top_k=3)
            
            if predictions:
                top_class, top_confidence = predictions[0]
                
                if top_confidence < 0.1:
                    cnn_response = "CNN: Low confidence - might be a kitchen utensil but unclear."
                elif top_confidence < 0.3:
                    cnn_response = f"CNN: Possibly a {top_class.lower()} (confidence: {top_confidence:.1%})"
                else:
                    cnn_response = f"CNN: Detected {top_class.lower()} (confidence: {top_confidence:.1%})"
                
                # Add second prediction if reasonably confident
                if len(predictions) > 1:
                    second_class, second_confidence = predictions[1]
                    if second_confidence > 0.1:
                        cnn_response += f", or {second_class.lower()} ({second_confidence:.1%})"
                
                responses.append(("CNN", cnn_response))
            else:
                responses.append(("CNN", "CNN classification found no predictions."))
        except Exception as e:
            error_msg = f"CNN classification encountered an error: {str(e)}"
            if DEBUG_MODE:
                print(f"❌ CNN Error: {e}")
            responses.append(("CNN", error_msg))
    
    # Handle case where no models could process the image
    if not responses:
        return BotReply(text="Sorry, I couldn't analyze this image. All available vision models encountered errors or are unavailable.")
    
    # Format combined response
    final_response = "🖼️ Image Analysis Results:\n\n"
    for model_name, response in responses:
        final_response += f"{model_name}: {response}\n\n"
    
    # Add helpful tips and mode adjustment notices
    if original_mode != mode:
        final_response += f"💡 Note: Requested {original_mode.upper()} mode, but used {mode.upper()} due to model availability.\n\n"
    
    # Add summary based on actual mode used
    if mode == "both" and len(responses) == 2 and all("error" not in resp[1].lower() for resp in responses):
        final_response += "💡 Tip: YOLO shows all detected objects with bounding boxes, while CNN provides single-object classification."
    elif mode == "cnn":
        if yolo_available:
            final_response += "💡 Using CNN for single-object classification. Try 'Detect everything in this image' for multi-object detection."
        else:
            final_response += "💡 Using CNN for single-object classification. (YOLO detection currently unavailable)"
    elif mode == "yolo":
        if cnn_available:
            final_response += "💡 Using YOLO for multi-object detection with bounding boxes. Try 'What is in this image?' for classification."
        else:
            final_response += "💡 Using YOLO for multi-object detection with bounding boxes. (CNN classification currently unavailable)"
    
    return BotReply(text=final_response.strip())
    
    
def vision_reply_cnn_only(image_path: str) -> 'BotReply | None':
    """
    Legacy CNN-only vision analysis (kept for compatibility).
    
    Args:
        image_path: Path to image file
        
    Returns:
        BotReply with CNN classification results or None if failed
    """
    return vision_reply(image_path, mode="cnn")

def main():
    classes = [
        "Blender", "Bowl", "Canopener", "Choppingboard", "Colander", "Cup", "Dinnerfork", "Dinnerknife", "Fishslice", "Garlicpress", "Kitchenknife", "Ladle", "Pan", "Peeler", "Saucepan", "Spoon", "Teaspoon", "Tongs", "Tray", "Whisk", "Woodenspoon"
    ]
    
    # Build vision features description based on availability
    vision_features = []
    if cnn_available and yolo_available:
        vision_features = [
            "  • Direct: 'image: path/to/image.jpg' (uses both CNN + YOLO)",
            "  • CNN only: 'What is in this image?' (single-object classification)",
            "  • YOLO only: 'Detect everything in this image' (multi-object with annotations)"
        ]
    elif cnn_available:
        vision_features = [
            "  • Direct: 'image: path/to/image.jpg' (CNN classification)",
            "  • CNN: 'What is in this image?' (single-object classification)",
            "  • Note: YOLO detection currently unavailable"
        ]
    elif yolo_available:
        vision_features = [
            "  • Direct: 'image: path/to/image.jpg' (YOLO detection)",
            "  • YOLO: 'Detect everything in this image' (multi-object with annotations)",
            "  • Note: CNN classification currently unavailable"
        ]
    else:
        vision_features = [
            "  • Image analysis currently unavailable (no vision models loaded)"
        ]
    
    vision_text = "\n".join(vision_features)
    
    # Welcome message based on debug mode
    if DEBUG_MODE:
        print(f"""
Welcome to the Kitchen Utensils Chatbot (Prototype) - DEBUG MODE

You can:
- Ask about kitchen utensils (e.g., 'What is a fishslice?', 'What is a ladle?')
- Check facts about utensils (e.g., 'Check that tongs are microwave safe')
- Tell the chatbot facts (e.g., 'I know that a tray is metal')
- Ask about utensil safety (e.g., 'Is a kitchen knife safe for children?')
- Identify utensils from images:
{vision_text}
- Type 'exit' or 'quit' to leave

Supported utensil classes:
{', '.join(classes[:8])}
{', '.join(classes[8:16])}
{', '.join(classes[16:])}
Supported fact properties: Metal, Plastic, Wood, Ceramic, Sharp, MicrowaveSafe, OvenSafe, DishwasherSafe, ChildSafe, RequiresCaution, etc.

[DEBUG MODE: Shows detailed routing decisions and logic engine steps]
""")
    else:
        print(f"""
Welcome to the Kitchen Utensils Chatbot!

You can:
- Ask about kitchen utensils (e.g., 'What is a fishslice?', 'What is a ladle?')
- Check facts about utensils (e.g., 'Check that tongs are microwave safe')
- Tell the chatbot facts (e.g., 'I know that a tray is metal')
- Ask about utensil safety (e.g., 'Is a kitchen knife safe for children?')
- Identify utensils from images:
{vision_text}
- Type 'exit' or 'quit' to leave

Supported utensil classes:
{', '.join(classes[:8])}
{', '.join(classes[8:16])}
{', '.join(classes[16:])}
""")
    while True:
        user_input_original = input("> ")
        if user_input_original.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        
        # Check if input is an image path (direct syntax)
        if user_input_original.lower().startswith("image:"):
            image_path = user_input_original[6:].strip()
            if DEBUG_MODE:
                print(f"\n🖼️ DEBUG: Processing image: {image_path}")
                print("─" * 50)
            
            vision_result = vision_reply(image_path, mode="both")
            if vision_result:
                if DEBUG_MODE:
                    print(f"4️⃣ Vision (BOTH): {vision_result.text}")
                    print("✅ USING VISION ANSWER")
                print(vision_result.text)
            else:
                if DEBUG_MODE:
                    print("4️⃣ Vision (BOTH): Failed to process image")
                print("Sorry, I couldn't process that image.")
            continue
            
        # Check if input is a natural language image query
        image_query_type = get_image_query_type(user_input_original)
        if image_query_type != "none":
            if DEBUG_MODE:
                if image_query_type == "cnn":
                    print(f"\n🖼️ DEBUG: Detected 'What is in this image?' - Opening file dialog for CNN classification")
                else:  # yolo
                    print(f"\n🖼️ DEBUG: Detected 'Detect everything in this image' - Opening file dialog for YOLO detection")
                print("─" * 50)
            
            image_path = prompt_for_image_path()
            if image_path:
                if DEBUG_MODE:
                    print(f"\n🖼️ DEBUG: Processing selected image with {image_query_type.upper()}: {image_path}")
                    print("─" * 50)
                
                vision_result = vision_reply(image_path, mode=image_query_type)
                if vision_result:
                    if DEBUG_MODE:
                        print(f"4️⃣ Vision ({image_query_type.upper()}): {vision_result.text}")
                        print("✅ USING VISION ANSWER")
                    print(vision_result.text)
                else:
                    if DEBUG_MODE:
                        print(f"4️⃣ Vision ({image_query_type.upper()}): Failed to process image")
                    print("Sorry, I couldn't process that image.")
            else:
                print("File selection cancelled.")
            continue
        
        user_input_normalized = normalize(user_input_original)
        if DEBUG_MODE:
            print("\n🔍 DEBUG: Processing normalized input: " + user_input_normalized + ", Original input: " + user_input_original)
            print("─" * 50)

        # DUAL PIPELINE ARCHITECTURE:
        # 1. Logic/Fuzzy Pipeline (Step 0): Runs first, no NLP fallback
        # 2. NLP Pipeline (Steps 1-3): Only runs if Logic/Fuzzy doesn't match
        
        # Step 0: Logic/Fuzzy Pipeline - Completely separate from NLP
        # Handles: fact assertions, fact checks, fuzzy safety queries
        # Returns result and stops (including "Unknown.") - no NLP fallback mixing
        logic_result_obj = logic_reply(user_input_original)
        
        if logic_result_obj:
            if DEBUG_MODE:
                print(f"0️⃣ Logic/Fuzzy: {logic_result_obj.text}")
                print("✅ USING LOGIC/FUZZY ANSWER (No NLP fallback)")
            print(logic_result_obj.text)
            continue # Logic pipeline complete - skip NLP pipeline
        else:
            if DEBUG_MODE:
                print(f"0️⃣ Logic/Fuzzy: No logic/fuzzy match found. Proceeding to NLP pipeline...")

        # NLP PIPELINE (Steps 1-3): Only runs if Logic/Fuzzy didn't match
        # Fallback chain: AIML → TF-IDF → Embedding
        
        # Step 1: AIML Pattern Matching
        aiml_result = aiml_reply(user_input_normalized)
        if aiml_result:
            if DEBUG_MODE:
                print(f"1️⃣ AIML: Found match → {aiml_result.text}")
                print("✅ USING AIML ANSWER")
            print(aiml_result.text)
            if aiml_result.end_conversation:
                print("Goodbye!")
                break
            continue
        else:
            if DEBUG_MODE:
                print(f"1️⃣ AIML: No match found for input: {user_input_normalized.upper()}")

        # Step 2: TF-IDF Similarity Matching (NLP Pipeline)
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
            if DEBUG_MODE:
                print(f"2️⃣ TF-IDF: Score={tfidf_score:.3f} (threshold={TFIDF_THRESHOLD})")
                print(f"   Top TF-IDF match → {tfidf_top_answer}")
                print(f"   Top TF-IDF score: {tfidf_score:.3f}")
            if tfidf_result:
                if DEBUG_MODE:
                    print(f"   Found match → {tfidf_result.text}")
                    print("✅ USING TF-IDF ANSWER")
                print(tfidf_result.text)
                answer_found = True
        else:
            if DEBUG_MODE:
                print("2️⃣ TF-IDF: Not available")
        
        # Step 3: Embedding Semantic Matching (NLP Pipeline Fallback)
        # Only runs if TF-IDF confidence is below threshold
        embed_score = None
        embed_top_answer = None
        if embed_sim and tfidf_sim:
            embed_score = embed_sim.get_best_similarity_score(user_input_normalized)
            user_embedding = embed_sim.nlp(user_input_normalized).vector.reshape(1, -1)
            similarities = cosine_similarity(user_embedding, embed_sim.question_embeddings)[0]
            best_idx = similarities.argmax()
            embed_top_answer = embed_sim.answers[best_idx]
            if DEBUG_MODE:
                print(f"3️⃣ Embedding: Score={embed_score:.3f}")
                print(f"   Top Embedding match → {embed_top_answer}")
                print(f"   Top Embedding score: {embed_score:.3f}")
            if not answer_found and tfidf_score is not None and tfidf_score < TFIDF_THRESHOLD:
                if DEBUG_MODE:
                    print(f"   TF-IDF not confident ({tfidf_score:.3f} < {TFIDF_THRESHOLD}) → Trying embedding...")
                embed_result = embed_reply(user_input_normalized) # Ensure this uses normalized
                if embed_result:
                    if DEBUG_MODE:
                        print(f"   Found match (≥0.6) → {embed_result.text}")
                        print("✅ USING EMBEDDING ANSWER")
                    print(embed_result.text)
                    answer_found = True
                else:
                    if DEBUG_MODE:
                        print(f"   No match (score < 0.6)")
            elif tfidf_score is not None:
                if DEBUG_MODE:
                    print(f"   TF-IDF confident ({tfidf_score:.3f} ≥ {TFIDF_THRESHOLD}) → Skipping embedding")
        else:
            if DEBUG_MODE:
                print("3️⃣ Embedding: Not available")
        
        # Step 4: Final Fallback (No pipeline matched)
        if not answer_found:
            if DEBUG_MODE:
                print("5️⃣ Fallback: No matches found in any pipeline")
            print("Sorry, I don't know that.")

if __name__ == "__main__":
    main() 