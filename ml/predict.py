import argparse
import json
import os
from typing import Dict

import torch

from src.data.datasets import Vocab, encode_text
from src.models.multitask_text_model import MultiTaskTextModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--max_len", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})

    vocab_list = ckpt["vocab"]
    vocab = Vocab()
    # rebuild vocab indices to match saved list
    vocab.itos = list(vocab_list)
    vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}

    label_classes: Dict[str, list] = ckpt["label_classes"]

    model = MultiTaskTextModel(
        vocab_size=len(vocab),
        embed_dim=cfg.get("embed_dim", 128),
        encoder_hidden=cfg.get("encoder_hidden", 256),
        num_layers=cfg.get("num_layers", 2),
        sentiment_classes=len(label_classes["sentiment"]),
        genre_classes=len(label_classes["genre"]),
        emotion_classes=len(label_classes["emotion"]),
        encoder_type=cfg.get("encoder_type", "transformer"),
        max_len=cfg.get("max_len", args.max_len),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    input_ids, attention_mask = encode_text(args.text, vocab, cfg.get("max_len", args.max_len))
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    def decode(logits, classes):
        idx = int(logits.argmax(dim=-1).item())
        return classes[idx]

    result = {
        "sentiment": decode(outputs["sentiment"], label_classes["sentiment"]),
        "genre": decode(outputs["genre"], label_classes["genre"]),
        "emotion": decode(outputs["emotion"], label_classes["emotion"]),
    }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()


