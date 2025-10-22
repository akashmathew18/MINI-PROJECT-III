# JV Cinelytics - System Architecture & UML Diagrams

## System Overview

JV Cinelytics is a web-based application for intelligent movie script analysis using NLP and machine learning. The system consists of:

- **Frontend**: Streamlit-based web interface
- **Backend**: Django REST API with PyTorch-based NLP engine
- **NLP Engine**: Custom modules for script analysis
- **TTS Module**: Edge-TTS for voice narration

---

## 1. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    JV Cinelytics System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   Frontend      │    │    Backend      │    │   TTS       │ │
│  │   (Streamlit)   │◄──►│   (Django)      │◄──►│ (Edge-TTS)  │ │
│  │                 │    │                 │    │             │ │
│  │ • File Upload   │    │ • REST API      │    │ • Gender    │ │
│  │ • Results       │    │ • NLP Engine    │    │ • Emotion   │ │
│  │ • Visualization │    │ • PyTorch       │    │ • Audio     │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│           │                       │                             │
│           └───────────────────────┼─────────────────────────────┘
│                                   │
│  ┌─────────────────────────────────┼─────────────────────────────┐
│  │                    NLP Engine Modules                         │
│  │                                                                 │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │  │   Scene     │ │ Character   │ │ Sentiment   │ │ Genre   │ │
│  │  │ Breakdown   │ │ Extraction  │ │ Analysis    │ │ Class.  │ │
│  │  │             │ │             │ │             │ │         │ │
│  │  │ • INT/EXT   │ │ • ALL CAPS  │ │ • PyTorch   │ │ • ML    │ │
│  │  │ • Headings  │ │ • Names     │ │ • Labels    │ │ • 7 Genres│ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
│  └─────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        NLP Engine Classes                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    SentimentClassifier                      │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ + vocab_size: int                                          │ │
│  │ + embed_dim: int                                           │ │
│  │ + num_classes: int                                         │ │
│  │ + embedding: Embedding                                     │ │
│  │ + fc: Linear                                               │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ + forward(x)                                               │ │
│  │ + predict_sentiment(text: str) -> str                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▲                                 │
│                              │                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    GenreClassifier                          │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ + vocab_size: int                                          │ │
│  │ + embed_dim: int                                           │ │
│  │ + num_classes: int                                         │ │
│  │ + embedding: Embedding                                     │ │
│  │ + fc: Linear                                               │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ + forward(x)                                               │ │
│  │ + predict_genre(text: str) -> str                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    API Views                               │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ + SceneBreakdownView                                       │ │
│  │ + CharacterExtractionView                                  │ │
│  │ + SentimentAnalysisView                                    │ │
│  │ + GenreClassificationView                                  │ │
│  │ + TTSNarrationView                                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Sequence Diagram (API Request Flow)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Frontend  │    │   Django    │    │  NLP Engine │    │    TTS      │
│ (Streamlit) │    │    API      │    │             │    │ (Edge-TTS)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       │ 1. Upload Script │                   │                   │
       │─────────────────►│                   │                   │
       │                   │                   │                   │
       │                   │ 2. Process Script│                   │
       │                   │─────────────────►│                   │
       │                   │                   │                   │
       │                   │                   │ 3. Scene Analysis│
       │                   │                   │─────────────────►│
       │                   │                   │                   │
       │                   │                   │ 4. Character     │
       │                   │                   │    Extraction    │
       │                   │                   │─────────────────►│
       │                   │                   │                   │
       │                   │                   │ 5. Sentiment     │
       │                   │                   │    Analysis      │
       │                   │                   │─────────────────►│
       │                   │                   │                   │
       │                   │                   │ 6. Genre         │
       │                   │                   │    Classification│
       │                   │                   │─────────────────►│
       │                   │                   │                   │
       │                   │                   │ 7. TTS Narration │
       │                   │                   │─────────────────►│
       │                   │                   │                   │
       │                   │ 8. Return Results│                   │
       │                   │◄──────────────────│                   │
       │                   │                   │                   │
       │ 9. Display Results│                   │                   │
       │◄──────────────────│                   │                   │
       │                   │                   │                   │
```

---

## 4. Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    JV Cinelytics Components                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │   Frontend      │    │    Backend      │    │   External  │ │
│  │   Components    │    │   Components    │    │   Services  │ │
│  │                 │    │                 │    │             │ │
│  │ • File Upload   │    │ • Django Admin  │    │ • Edge-TTS  │ │
│  │ • Results       │    │ • REST API      │    │   API       │ │
│  │ • Visualization │    │ • Database      │    │ • PyTorch   │ │
│  │ • Navigation    │    │ • Media Files   │    │   Models    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│           │                       │                             │
│           └───────────────────────┼─────────────────────────────┘
│                                   │
│  ┌─────────────────────────────────┼─────────────────────────────┐
│  │                    Core NLP Modules                           │
│  │                                                                 │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │  │   Scene     │ │ Character   │ │ Sentiment   │ │ Genre   │ │
│  │  │ Breakdown   │ │ Extraction  │ │ Analysis    │ │ Class.  │ │
│  │  │             │ │             │ │             │ │         │ │
│  │  │ • Regex     │ │ • ALL CAPS  │ │ • PyTorch   │ │ • ML    │ │
│  │  │ • Headings  │ │ • Names     │ │ • Labels    │ │ • 7 Genres│ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
│  └─────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Use Case Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Use Cases                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                               │
│  │   User      │                                               │
│  │ (Filmmaker/│                                               │
│  │ Scriptwriter│                                               │
│  └─────────────┘                                               │
│           │                                                     │
│           │ 1. Upload Script                                   │
│           │─────────────────────────────────────────────────────┤
│           │                                                     │
│           │ 2. Analyze Scenes                                  │
│           │─────────────────────────────────────────────────────┤
│           │                                                     │
│           │ 3. Extract Characters                              │
│           │─────────────────────────────────────────────────────┤
│           │                                                     │
│           │ 4. Analyze Sentiment                               │
│           │─────────────────────────────────────────────────────┤
│           │                                                     │
│           │ 5. Classify Genre                                  │
│           │─────────────────────────────────────────────────────┤
│           │                                                     │
│           │ 6. Generate Narration                              │
│           │─────────────────────────────────────────────────────┤
│           │                                                     │
│           │ 7. View Results                                    │
│           │─────────────────────────────────────────────────────┤
│           │                                                     │
│           │ 8. Download Audio                                  │
│           │─────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints Summary

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/api/scene-breakdown/` | POST | `{"script": "..."}` | `{"scenes": [...]}` |
| `/api/character-extraction/` | POST | `{"script": "..."}` | `{"characters": [...]}` |
| `/api/sentiment-analysis/` | POST | `{"text": "..."}` | `{"sentiment": "..."}` |
| `/api/genre-classification/` | POST | `{"text": "..."}` | `{"genre": "..."}` |
| `/api/tts-narration/` | POST | `{"text": "...", "gender": "...", "emotion": "..."}` | `{"audio_file": "..."}` |

---

## Technology Stack

- **Frontend**: Streamlit (Python)
- **Backend**: Django + Django REST Framework
- **NLP**: PyTorch, Custom Models
- **TTS**: Edge-TTS (Microsoft)
- **Architecture**: Modular, API-first
- **Deployment**: Local development, scalable to cloud 