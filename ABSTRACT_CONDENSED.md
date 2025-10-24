# Abstract: JV Cinelytics - Intelligent Movie Script Analysis Platform
## (Condensed Version for Academic Submission)

---

**Problem Context:**  
Manual movie script analysis is time-consuming and lacks automation, creating challenges for filmmakers and scriptwriters who need quick, data-driven insights into screenplay elements such as character importance, plot structure, emotional tone, and genre classification. This project addresses the need for an intelligent automated system that can process screenplay documents and extract comprehensive analytical insights.

**Objective:**  
To develop a unified machine learning platform that (1) trains custom multitask deep learning models for simultaneous sentiment analysis, genre classification, and emotion detection, (2) analyzes movie scripts by extracting characters, locations, and generating intelligent summaries, and (3) provides a user-friendly Streamlit web interface for seamless interaction.

**Dataset Description:**  
The system utilizes custom JSONL-formatted datasets with screenplay text labeled across three dimensions: sentiment (3 classes), genre (7 classes: action, drama, comedy, romance, thriller, sci-fi, horror), and emotion (7 classes). Script analysis accepts .docx and .txt files in standard screenplay format with character names, dialogue, and action descriptions.

**Methodology:**  
The project implements a **MultiTaskTextModel** with optional Transformer or Bidirectional LSTM encoders (embedding dim: 128, hidden: 256, 2 layers) and three task-specific classification heads. Data preprocessing includes tokenization, vocabulary construction with frequency filtering, and attention masking for variable-length sequences. Training uses AdamW optimizer (lr=3e-4, batch size=32, 5 epochs) with weighted multitask cross-entropy loss. Script analysis employs regex-based character extraction with importance scoring (dialogue count × 2 + mentions), location detection from scene headings, and hybrid summarization combining LexRank extractive methods with DistilBART abstractive generation.

**Results and Insights:**  
The multitask learning approach successfully enables shared representation learning across related NLP tasks. The integrated Streamlit application provides real-time script analysis with character rankings, location inventories, story-focused plot summaries (configurable 7-30+ sentences), and genre predictions. The system processes scripts in minutes, delivering quantitative metrics including character importance scores, location frequency statistics, and comprehensive narrative analysis.

**Impact/Conclusion:**  
JV Cinelytics delivers actionable insights for data-driven decision-making in film production by replacing subjective manual analysis with automated quantitative metrics. The platform enables writers to identify dialogue imbalances, reveals narrative pacing through story structure analysis, informs production logistics with location inventories, and provides emotional tone guidance. By democratizing access to advanced NLP capabilities through an intuitive interface, the system improves efficiency (reducing hours to minutes), enhances creative decision-making with objective evidence, and demonstrates that AI can effectively augment rather than replace human creativity in the entertainment industry. The modular architecture and local processing ensure privacy, scalability, and accessibility for filmmakers regardless of technical background.

---

**Word Count**: ~380 words  
**Technical Stack**: PyTorch, Streamlit, Transformers, NLTK  
**Key Innovation**: Unified multitask learning + hybrid summarization for comprehensive screenplay analysis
