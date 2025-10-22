# JV Cinelytics 🎬

**Complete ML Training & Script Analysis Platform**

JV Cinelytics is a unified application that combines machine learning training capabilities with comprehensive script analysis tools. Train multitask models for sentiment, genre, and emotion classification, then use them to analyze movie scripts with character extraction, location detection, and automated summarization.

## 🚀 Features

### 🤖 ML Training
- **Multitask Model Training**: Train models for sentiment, genre, and emotion classification
- **Custom Data Support**: Upload your own JSONL training data
- **Flexible Architecture**: Transformer or BiLSTM encoders
- **Real-time Training**: Monitor training progress with live updates

### 📊 Script Analysis
- **Character Analysis**: Extract and rank characters by importance (dialogue count + presence)
- **Location Detection**: Find all scene locations from INT./EXT. headings
- **Script Summarization**: Generate intelligent summaries of plot points
- **Multi-format Support**: .docx and .txt file uploads

### 🔮 Genre Prediction
- **ML-Powered**: Use trained models for accurate genre classification
- **Keyword Fallback**: Simple keyword-based classification when no model is trained
- **Real-time Prediction**: Instant genre prediction for any text input

### 🎯 Unified Interface
- **Single App**: Everything in one Streamlit application
- **No APIs**: Fully local processing, no external dependencies
- **Modern UI**: Clean, responsive design with interactive visualizations

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│        JV Cinelytics App            │
│        (Streamlit)                  │
├─────────────────────────────────────┤
│  🤖 ML Training  │  📊 Script      │
│  - Data Upload   │  Analysis       │
│  - Model Config  │  - Character    │
│  - Training      │  - Locations    │
│  - Evaluation    │  - Summary      │
├─────────────────────────────────────┤
│  🔮 Genre Prediction  │  ⚙️ Settings │
│  - Text Input    │  - Model Path   │
│  - ML/Keyword    │  - System Info  │
└─────────────────────────────────────┘
```

## 📁 Project Structure

```
MINI PROJECT III/
├── app.py                          # Main unified Streamlit app
├── requirements.txt                # All dependencies
├── launch.bat                      # Windows launcher
├── sample_data.jsonl              # Sample training data
├── ml/                            # ML training modules
│   ├── src/
│   │   ├── models/
│   │   │   └── multitask_text_model.py
│   │   └── data/
│   │       └── datasets.py
│   ├── train.py                   # Training script
│   ├── predict.py                 # Prediction script
│   └── tools/
│       └── prepare_data.py        # Data preparation
├── script_analysis/
│   ├── script_analyzer.py         # Script analysis engine
│   └── README.md                  # Analysis documentation
├── frontend/                      # Legacy UI (optional)
└── docs/                          # Documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- pip package manager

### Quick Start
```bash
# 1. Clone and setup
git clone <repository-url>
cd MINI\ PROJECT\ III

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the app
streamlit run app.py
```

### Windows One-Click Launch
```bash
# Double-click launch.bat
# This will activate venv, install deps, and launch the app
```

## 🎯 Usage

### 1. ML Training
1. Go to "🤖 ML Training" tab
2. Upload your training data (JSONL format)
3. Configure model parameters
4. Click "🚀 Start Training"
5. Monitor training progress

### 2. Script Analysis
1. Go to "📊 Script Analysis" tab
2. Upload a .docx or .txt script file
3. Get instant analysis:
   - Character ranking by importance
   - Location extraction
   - Script summary
   - Genre prediction
4. Download analysis report

### 3. Genre Prediction
1. Go to "🔮 Genre Prediction" tab
2. Enter text for analysis
3. Get instant genre prediction
4. Uses trained model if available, otherwise keyword-based

## 📊 Training Data Format

Use JSONL format for training data:
```json
{"text": "He runs through the alley, chased by masked men.", "genre": "thriller", "sentiment": "negative", "emotion": "fear"}
{"text": "They laugh and hug after the show.", "genre": "comedy", "sentiment": "positive", "emotion": "joy"}
```

### Label Mapping
- **Sentiment**: {negative, neutral, positive}
- **Genre**: {action, drama, comedy, romance, thriller, sci-fi, horror}
- **Emotion**: {anger, joy, sadness, fear, disgust, surprise, neutral}

## 🔧 Model Configuration

### Default Parameters
- **Encoder**: Transformer (4 heads) or BiLSTM
- **Embedding Dim**: 128
- **Encoder Hidden**: 256
- **Layers**: 2
- **Max Length**: 256
- **Batch Size**: 32
- **Learning Rate**: 3e-4
- **Epochs**: 5

### Customization
All parameters can be adjusted in the ML Training interface.

## 📈 Script Analysis Features

### Character Analysis
- Extracts character names from ALL CAPS text
- Ranks by dialogue count + total mentions
- Shows importance score

### Location Detection
- Finds scene headings (INT./EXT. LOCATION - TIME)
- Counts mention frequency
- Sorts by importance

### Script Summarization
- Extractive summarization
- Focuses on dialogue and key action
- Configurable summary length

## 🎭 Genre Classification

### ML Model (Recommended)
- Uses trained multitask model
- Higher accuracy
- Requires training data

### Keyword Fallback
- Simple keyword matching
- Works without training
- Good for basic classification

## 🚀 Advanced Features

### Model Evaluation
- View training progress
- Model configuration details
- Performance metrics

### Settings Management
- Configure model paths
- Clear caches
- System information

## 🔮 Future Enhancements

- **Advanced Visualizations**: Training curves, confusion matrices
- **Model Comparison**: A/B testing different architectures
- **Batch Processing**: Analyze multiple scripts at once
- **Export Options**: PDF reports, Excel exports
- **Cloud Integration**: Optional cloud storage for models

## 📚 Documentation

- **Script Analysis**: See `script_analysis/README.md`
- **ML Training**: See `ml/` directory for training scripts
- **API Reference**: All functions are documented in the code

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **PyTorch**: Deep learning framework
- **Streamlit**: Web app framework
- **NLTK**: Natural language processing
- **Transformers**: Hugging Face model library

---

**JV Cinelytics** - Where machine learning meets cinematic storytelling.