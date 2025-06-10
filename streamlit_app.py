import streamlit as st
import os
from io import StringIO
import sys

# Configure page
st.set_page_config(
    page_title="Kitchen Utensils Chatbot",
    page_icon="🍴",
    layout="wide"
)

# Import core functions from main.py
# We'll suppress output during import to avoid startup messages in Streamlit
old_stdout = sys.stdout
sys.stdout = StringIO()
try:
    from main import (
        aiml_reply, tfidf_reply, embed_reply, logic_reply, vision_reply,
        normalize, BotReply, cnn_available, yolo_available
    )
finally:
    sys.stdout = old_stdout

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# App header
st.title("🍴 Kitchen Utensils Chatbot")
st.markdown("*University Module Assessment - Multi-modal AI Demo*")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot can:
    - Answer questions about kitchen utensils
    - Check facts about utensils
    - Accept new facts about utensils
    - Assess utensil safety
    - Identify utensils from images
    """)
    
    st.header("Vision Features")
    if cnn_available and yolo_available:
        st.success("✅ CNN Classification")
        st.success("✅ YOLO Detection")
    elif cnn_available:
        st.success("✅ CNN Classification")
        st.warning("⚠️ YOLO Detection unavailable")
    elif yolo_available:
        st.warning("⚠️ CNN Classification unavailable")
        st.success("✅ YOLO Detection")
    else:
        st.error("❌ Image analysis unavailable")
    
    st.header("Example Queries")
    st.markdown("""
    - What is a fishslice?
    - What is a ladle?
    - Check that tongs are microwave safe
    - I know that a tray is metal
    - Is a kitchen knife safe for children?
    """)

# Main chat interface
st.header("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Image upload section
uploaded_file = st.file_uploader("Upload an image of a kitchen utensil", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process image when button is clicked
    if st.button("Analyse Image"):
        with st.spinner("Analysing image..."):
            vision_result = vision_reply(temp_path, mode="both")
            if vision_result:
                st.session_state.messages.append({"role": "assistant", "content": vision_result.text})
                with st.chat_message("assistant"):
                    st.markdown(vision_result.text)
            else:
                error_msg = "Sorry, I couldn't process that image."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_message(user_input: str) -> str:
    """Process user input through the same pipeline as main.py"""
    
    # Normalize input
    user_input_normalized = normalize(user_input)
    
    # Step 0: Logic/Fuzzy Pipeline
    logic_result = logic_reply(user_input)
    if logic_result:
        return logic_result.text
    
    # Step 1: AIML Pattern Matching
    aiml_result = aiml_reply(user_input_normalized)
    if aiml_result:
        return aiml_result.text
    
    # Step 2: TF-IDF Similarity Matching
    tfidf_result = tfidf_reply(user_input_normalized)
    if tfidf_result:
        return tfidf_result.text
    
    # Step 3: Embedding Semantic Matching
    embed_result = embed_reply(user_input_normalized)
    if embed_result:
        return embed_result.text
    
    # Fallback
    return "Sorry, I don't know that."

# Chat input
if prompt := st.chat_input("Ask me about kitchen utensils..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process the message through the same pipeline as main.py
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_message(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*This is a university module assessment project demonstrating multi-modal AI integration.*") 