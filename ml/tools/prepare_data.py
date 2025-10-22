import argparse
import json
import os
import random
from typing import Dict, List


SENTIMENT_MAP_5_TO_3 = {
    "very negative": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "very positive": "positive",
}

GOEMOTIONS_TO_7 = {
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "disgust",
    "disgust": "disgust",
    "fear": "fear",
    "nervousness": "fear",
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "optimism": "joy",
    "pride": "joy",
    "admiration": "joy",
    "relief": "joy",
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    "embarrassment": "sadness",
    "surprise": "surprise",
    # default bucket
}

COARSE_GENRES = {
    "action": "action",
    "adventure": "action",
    "crime": "thriller",
    "thriller": "thriller",
    "mystery": "thriller",
    "drama": "drama",
    "romance": "romance",
    "comedy": "comedy",
    "sci-fi": "sci-fi",
    "science fiction": "sci-fi",
    "horror": "horror",
}


def to_jsonl(records: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_records(records: List[Dict], val_ratio: float = 0.1, seed: int = 42):
    random.Random(seed).shuffle(records)
    n_val = int(len(records) * val_ratio)
    return records[n_val:], records[:n_val]


def unify_record(text: str, sentiment=None, genre=None, emotion=None) -> Dict:
    rec: Dict = {"text": text}
    if sentiment is not None:
        rec["sentiment"] = sentiment
    if genre is not None:
        rec["genre"] = genre
    if emotion is not None:
        rec["emotion"] = emotion
    return rec


def main():
    parser = argparse.ArgumentParser(description="Prepare unified JSONL datasets")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file with fields: text, sentiment?, genre?, emotion?")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    # Load mixed JSON or JSONL
    records: List[Dict] = []
    if args.input.endswith(".jsonl"):
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                raise ValueError("Input JSON must be a list of objects")

    unified: List[Dict] = []
    for r in records:
        text = r.get("text") or r.get("sentence") or r.get("plot") or ""

        sent = r.get("sentiment")
        if sent in SENTIMENT_MAP_5_TO_3:
            sent = SENTIMENT_MAP_5_TO_3[sent]

        emo = r.get("emotion")
        if emo in GOEMOTIONS_TO_7:
            emo = GOEMOTIONS_TO_7[emo]

        gen = r.get("genre")
        if isinstance(gen, list) and gen:
            # take first mapped genre found
            mapped = None
            for g in gen:
                g_l = str(g).lower()
                if g_l in COARSE_GENRES:
                    mapped = COARSE_GENRES[g_l]
                    break
            gen = mapped
        elif isinstance(gen, str):
            gen = COARSE_GENRES.get(gen.lower())

        unified.append(unify_record(text, sent, gen, emo))

    train, val = split_records(unified, args.val_ratio)
    to_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    to_jsonl(val, os.path.join(args.output_dir, "val.jsonl"))
    print(f"Wrote {len(train)} train and {len(val)} val records to {args.output_dir}")


if __name__ == "__main__":
    main()


