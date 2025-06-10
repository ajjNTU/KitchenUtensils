# Streamlit Deployment Guide

## Quick Deployment to Streamlit Cloud

### Prerequisites
1. GitHub repository with the project
2. Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit web interface"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Configuration Files**
   - `streamlit_app.py` - Main application file
   - `requirements.txt` - Python dependencies (includes streamlit)
   - `packages.txt` - System dependencies for vision models
   - `.streamlit/config.toml` - Streamlit configuration

### Expected Result
- Working web demo at `https://your-app.streamlit.app`
- Interactive chat interface
- Image upload capability
- Same chatbot intelligence as CLI version
- Real-time model availability status

### Features
- ✅ **Chat Interface**: Interactive conversation with the chatbot
- ✅ **Image Upload**: Drag & drop or browse for kitchen utensil images
- ✅ **Vision Analysis**: CNN classification and YOLO detection
- ✅ **Model Status**: Visual indicators for available AI models
- ✅ **Responsive Design**: Works on desktop and mobile
- ✅ **Real-time Processing**: Instant responses and image analysis

### Model Files and GitHub Limitations

**Important**: Large model files are excluded from the GitHub repository due to size limits:
- CNN Model (`cnn_model.h5`) - ~96MB - excluded
- Embeddings (`qna_embeddings.npy`) - ~383KB - excluded  
- YOLO pretrained models - excluded

**✅ What Works Without Models**:
- Complete chat functionality (AIML, TF-IDF, Logic/Fuzzy reasoning)
- Text-based question answering about kitchen utensils
- Fact checking and logical assertions
- Safety assessments using fuzzy logic
- Professional web interface with clear status indicators

**⚠️ What's Limited Without Models**:
- CNN image classification features
- YOLO object detection features
- Semantic embedding fallback (uses TF-IDF instead)

### Troubleshooting

**If models don't load:**
- Vision features will show as unavailable (this is expected)
- Chat functionality will still work perfectly
- App is designed for graceful degradation
- Check Streamlit Cloud logs for specific errors

**If app is slow:**
- First load may take time for model initialization attempts
- Subsequent interactions should be faster
- Chat responses are immediate regardless of vision model status

**For Full Model Support** (optional):
- See `MODEL_DEPLOYMENT_SOLUTIONS.md` for advanced deployment options
- Consider hosting models externally and adding download functionality
- Use Git LFS for large file management (advanced)

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_md

# Run locally
streamlit run streamlit_app.py
```

### Architecture
The Streamlit app uses the same core logic as the CLI version:
- Imports functions from `main.py` without triggering CLI execution
- Maintains the same dual pipeline architecture (Logic/Fuzzy + NLP)
- Provides identical chatbot intelligence with a modern web interface
- Handles image processing through the same vision pipeline

This creates a minimal viable web demo that's ready for further enhancement while maintaining all the core functionality of the original CLI application. 