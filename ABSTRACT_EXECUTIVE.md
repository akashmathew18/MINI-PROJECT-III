# Abstract: JV Cinelytics - Intelligent Movie Script Analysis Platform
## (Ultra-Condensed Version - 150 words)

---

**JV Cinelytics** addresses the time-consuming challenge of manual movie script analysis by developing an automated machine learning platform for comprehensive screenplay analytics. The system implements a multitask deep learning model using Transformer/BiLSTM encoders (PyTorch) to simultaneously classify text across sentiment (3 classes), genre (7 classes), and emotion (7 classes) dimensions.

The platform analyzes screenplay files (.docx/.txt) through regex-based character extraction with importance scoring, location detection from scene headings, and hybrid summarization combining LexRank extractive and DistilBART abstractive techniques. Training utilizes custom JSONL datasets with AdamW optimization and weighted multitask cross-entropy loss.

Deployed via Streamlit, the application delivers real-time analysis including character rankings, location inventories, and configurable plot summaries (7-30+ sentences). Results demonstrate significant efficiency improvements—reducing hours of manual work to minutes—while providing data-driven insights for creative decision-making. The system successfully demonstrates AI augmentation of creative processes, making sophisticated NLP accessible to filmmakers without technical expertise.

---

**Word Count**: 149 words
