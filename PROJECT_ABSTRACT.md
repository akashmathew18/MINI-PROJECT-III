# Project Abstract: JV Cinelytics - Intelligent Movie Script Analysis Platform

## 1. Problem Context

In the film industry, script analysis is a critical yet time-consuming task for filmmakers, scriptwriters, and producers. Manual analysis of movie scripts requires extensive effort to extract key elements such as character relationships, plot structure, emotional tone, and genre classification. Traditional methods lack automation and fail to provide data-driven insights that could inform creative and production decisions. This project addresses the need for an intelligent, automated script analysis system that can quickly process screenplay documents and provide comprehensive analytical insights including character importance ranking, location detection, plot summarization, and multi-dimensional text classification.

The choice of this domain is motivated by the growing intersection of artificial intelligence and creative industries, where machine learning can augment human creativity rather than replace it, enabling content creators to make informed decisions backed by quantitative analysis.

---

## 2. Objective

The primary objective of **JV Cinelytics** is to develop a unified machine learning platform that:

1. **Trains custom multitask deep learning models** for simultaneous sentiment analysis, genre classification, and emotion detection from textual content
2. **Analyzes movie scripts** by extracting characters, locations, generating intelligent summaries, and predicting genres
3. **Provides a user-friendly web interface** through Streamlit for seamless interaction with both training and analysis capabilities
4. **Delivers actionable insights** through visualizations and downloadable reports to support creative decision-making in scriptwriting and production

The system combines both extractive and abstractive NLP techniques to deliver comprehensive screenplay analysis while maintaining interpretability and accuracy.

---

## 3. Dataset Description


---

## 4. Methodology

### 4.1 Data Preprocessing
- **Text Cleaning**: Removal of script-specific formatting, extraction of dialogue and action lines
- **Tokenization**: Simple whitespace-based tokenization with lowercase normalization
- **Vocabulary Construction**: Dynamic vocabulary building with frequency thresholding and special tokens (`<pad>`, `<unk>`)
- **Label Encoding**: Multi-task label encoding for sentiment, genre, and emotion classes
- **Sequence Padding**: Fixed-length sequences (max_len=256) with attention masking

### 4.2 Feature Engineering
- **Positional Embeddings**: Learnable position encodings for sequence modeling
- **Attention Masking**: Binary masks to handle variable-length inputs
- **Task-Specific Masking**: Enables partial labeling across multiple tasks
- **Character Importance Scoring**: Weighted combination of dialogue count (×2) and total mentions
- **Location Frequency Analysis**: Regex-based extraction and ranking of scene locations

### 4.3 Model Architecture

**MultiTaskTextModel** - A unified deep learning architecture:

**Encoder Options:**
1. **Transformer Encoder** (Default)
   - Multi-head self-attention (4 heads)
   - Position-wise feedforward networks
   - Layer normalization and residual connections
   - Embedding dimension: 128
   - Hidden dimension: 256
   - Number of layers: 2

2. **Bidirectional LSTM Encoder** (Alternative)
   - Bidirectional sequence processing
   - Dropout regularization between layers
   - Hidden size: 256 (512 total with bidirectionality)

**Task-Specific Heads:**
- Three independent linear classifiers for sentiment, genre, and emotion
- Shared representation learning with task-specific fine-tuning

**Loss Function:**
- Weighted multitask cross-entropy loss
- Weights: Sentiment (1.0), Genre (1.0), Emotion (0.5)
- Task masking for handling missing labels

### 4.4 Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 3e-4
- **Batch Size**: 32
- **Epochs**: 5
- **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients
- **Validation**: Early stopping based on validation loss
- **Device**: CUDA-enabled GPU when available, CPU fallback

### 4.5 Script Analysis Pipeline

**Character Extraction:**
- Regex pattern matching for ALL CAPS character names
- Dialogue count and mention frequency tracking
- Importance ranking based on weighted scoring

**Location Detection:**
- Scene heading pattern recognition (INT./EXT. LOCATION - TIME)
- Frequency-based location ranking

**Summarization:**
- **Extractive Approach**: LexRank algorithm for sentence importance scoring
- **Abstractive Approach**: DistilBART-CNN-12-6 for sequence-to-sequence generation
- **Hybrid Pipeline**: Intelligent scene splitting, narrative element extraction, story arc analysis
- **Customizable Length**: Target sentence count (default: 25 for detailed synopsis)

**Genre Prediction:**
- Primary: Trained multitask model inference
- Fallback: Keyword-based classification when model unavailable

### 4.6 Evaluation Metrics
- **Cross-Entropy Loss**: Primary training objective
- **Validation Loss**: Model selection criterion
- **Task-Specific Accuracy**: Per-task classification performance (future enhancement)
- **Qualitative Analysis**: Summary coherence and completeness assessment

---

## 5. Results and Insights

### 5.1 Model Performance
The multitask learning approach successfully enables:
- **Shared Representation Learning**: Common text encoder captures general linguistic features
- **Task-Specific Specialization**: Independent classification heads for domain-specific predictions
- **Efficient Training**: Single model handles three related tasks simultaneously
- **Model Generalization**: Transferable to various screenplay analysis scenarios

### 5.2 Script Analysis Capabilities

**Character Analysis:**
- Automated extraction and ranking of characters by narrative importance
- Quantitative metrics (dialogue count + total mentions) provide objective character significance scores
- Identifies both major protagonists and supporting characters

**Location Insights:**
- Comprehensive scene location inventory with frequency statistics
- Helps production teams understand set requirements and location diversity
- Enables quick identification of primary shooting locations

**Intelligent Summarization:**
- Story-focused synopsis generation capturing plot progression, character arcs, and conflicts
- Configurable detail level (7-30+ sentences) for different use cases
- Combines extractive precision with abstractive fluency for comprehensive narrative summaries

**Genre Classification:**
- Multi-genre support covering major screenplay categories
- ML-powered predictions when trained model available
- Intelligent keyword-based fallback for robustness

### 5.3 Web Application Integration

The project features a **fully integrated Streamlit web application** enabling:
- **Unified Interface**: Single application for both ML training and script analysis
- **Real-time Processing**: Instant analysis upon script upload
- **Interactive Visualizations**: Character rankings, location distributions, and narrative insights
- **Download Reports**: Export analysis results for documentation
- **Model Management**: Configure paths, clear caches, view training progress
- **No External APIs**: Fully local processing ensuring privacy and offline capability

**Key Functionality:**
- Upload .docx/.txt scripts for instant analysis
- Train custom models with user-provided JSONL datasets
- Visualize analysis results with modern UI components
- Configure model parameters through intuitive interfaces
- Monitor training progress with live updates

---

## 6. Impact and Conclusion

**JV Cinelytics** delivers significant value to the film and entertainment industry by:

### 6.1 Actionable Insights
- **Character Development**: Writers can identify underutilized characters or dialogue imbalances
- **Story Structure**: Automated scene analysis reveals narrative pacing and arc progression
- **Production Planning**: Location lists and scene counts inform logistical requirements
- **Genre Optimization**: Classification feedback helps align scripts with target genres
- **Emotional Tone**: Sentiment and emotion analysis guides directorial interpretation

### 6.2 Data-Driven Decision Making
- Replaces subjective manual analysis with quantitative metrics
- Enables comparative analysis across multiple script versions
- Provides objective evidence for script revision discussions
- Supports pitch presentations with analytical credibility

### 6.3 Improved Efficiency
- **Time Savings**: Reduces hours of manual analysis to minutes of automated processing
- **Scalability**: Can process multiple scripts simultaneously
- **Accessibility**: User-friendly interface requires no technical expertise
- **Flexibility**: Customizable models adapt to specific creative domains

### 6.4 Technical Contributions
- **Modular Architecture**: Separation of ML training, analysis, and UI components
- **Multitask Learning**: Demonstrates effective knowledge sharing across related NLP tasks
- **Hybrid Summarization**: Combines extractive and abstractive approaches for superior results
- **Domain Specialization**: Tailored to screenplay-specific patterns and structures

### 6.5 Future Enhancements
- Advanced visualizations (training curves, confusion matrices)
- Batch processing for multiple scripts
- Character relationship network graphs
- Dialogue quality metrics and readability scores
- Integration with professional screenwriting software
- Cloud deployment for collaborative team usage

---

## Conclusion

The **JV Cinelytics** platform successfully bridges the gap between artificial intelligence and creative content analysis. By combining multitask deep learning with domain-specific NLP techniques, the system delivers comprehensive screenplay insights that enhance creative decision-making without replacing human intuition. The unified interface democratizes access to advanced text analytics, making sophisticated AI capabilities accessible to filmmakers regardless of technical background.

The project demonstrates that machine learning can serve as a powerful augmentation tool for creative industries, providing quantitative foundations for qualitative artistic decisions. With demonstrated success in character extraction, location detection, intelligent summarization, and multidimensional classification, JV Cinelytics represents a meaningful step toward data-driven storytelling and production optimization.

**Impact Summary**: Faster analysis, deeper insights, better decisions—empowering storytellers with the intelligence to craft compelling narratives backed by data.

---

## Technical Specifications Summary

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit (Python) |
| **ML Framework** | PyTorch 2.0+ |
| **NLP Libraries** | NLTK, Sumy, Transformers (Hugging Face) |
| **Model Architecture** | Transformer/BiLSTM Encoder + Multi-Head Classifier |
| **Summarization** | LexRank (Extractive) + DistilBART (Abstractive) |
| **Document Processing** | python-docx |
| **Data Format** | JSONL (JSON Lines) |
| **Deployment** | Local desktop application (Windows/macOS/Linux) |
| **License** | MIT |

---

**Project Name**: JV Cinelytics  
**Tagline**: Where Machine Learning Meets Cinematic Storytelling  
**Repository Structure**: Modular design with separate `ml/`, `script_analysis/`, and `frontend/` components  
**Launch Method**: One-click `launch.bat` script or `streamlit run app.py`  

**Key Achievement**: A complete end-to-end machine learning pipeline from data preprocessing to model training to production deployment with user-friendly interface—all in a single cohesive platform.
