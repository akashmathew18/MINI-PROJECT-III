# Script Analysis Module

This module processes uploaded movie scripts (.docx/.txt) and provides comprehensive analysis including:

1. **Script Summary** - Extractive summarization of key plot points
2. **Character Analysis** - Characters ranked by importance (dialogue count + presence)
3. **Location Extraction** - All scene locations with mention counts
4. **Genre Prediction** - AI-powered genre classification

## Features

- **Multi-format Support**: .docx and .txt files
- **Character Ranking**: Based on dialogue count and total mentions
- **Location Detection**: Extracts from scene headings (INT./EXT.)
- **Smart Summarization**: Focuses on dialogue and key action
- **Genre Classification**: Uses ML model or keyword-based fallback
- **Streamlit UI**: User-friendly web interface

## Installation

```bash
pip install -r script_analysis/requirements.txt
```

## Usage

### Command Line
```python
from script_analyzer import ScriptAnalyzer

analyzer = ScriptAnalyzer()
results = analyzer.analyze_script("your_script.txt")

print(f"Summary: {results['summary']}")
print(f"Characters: {results['characters']['list']}")
print(f"Locations: {results['locations']['list']}")
print(f"Genre: {results['genre']}")
```

### Streamlit Web App
```bash
streamlit run script_analysis/streamlit_app.py
```

## Script Format

The analyzer expects standard screenplay format:

```
INT. OFFICE - DAY

JOHN sits at his desk, looking frustrated.

JOHN
I can't believe this is happening!

MARY enters the room.

MARY
(concerned)
What's wrong, John?

EXT. STREET - NIGHT

A car chase ensues through the city streets.
```

## Output Format

```json
{
  "summary": "Script summary text...",
  "characters": {
    "count": 2,
    "list": [
      ["JOHN", 2, 5],  // [name, dialogue_count, total_mentions]
      ["MARY", 1, 3]
    ]
  },
  "locations": {
    "count": 2,
    "list": [
      ["OFFICE", 1],  // [location, mention_count]
      ["STREET", 1]
    ]
  },
  "genre": "drama",
  "total_pages": 1
}
```

## Integration with ML Module

To use the trained multitask model for genre prediction:

```python
analyzer = ScriptAnalyzer(model_path="ml/runs/exp1/model.pt")
```

This will use your custom trained model instead of the keyword-based fallback.
