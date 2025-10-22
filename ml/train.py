import argparse
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import MultiTaskJsonlDataset, build_vocab, LabelEncoder, collate_batch
from src.models.multitask_text_model import MultiTaskTextModel, multitask_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--encoder_hidden", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--encoder_type", type=str, default="transformer")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", type=str, default="runs/exp1")
    return p.parse_args()


def collect_texts(jsonl_path: str):
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json

            obj = json.loads(line)
            texts.append(obj.get("text", ""))
    return texts


def fit_label_encoders(train_path: str) -> Dict[str, LabelEncoder]:
    encs: Dict[str, LabelEncoder] = {"sentiment": LabelEncoder(), "genre": LabelEncoder(), "emotion": LabelEncoder()}
    labels = {"sentiment": [], "genre": [], "emotion": []}
    import json

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for k in labels.keys():
                v = obj.get(k)
                if v is not None:
                    labels[k].append(v)
    for k, enc in encs.items():
        enc.fit(labels[k])
    return encs


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            task_mask = {k: v.to(device) for k, v in batch["task_mask"].items()}
            outputs = model(input_ids, attention_mask)
            loss = multitask_loss(outputs, targets, task_mask)
            total_loss += loss.item()
            batches += 1
    return total_loss / max(batches, 1)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # vocab and labels
    vocab = build_vocab(collect_texts(args.train))
    label_encoders = fit_label_encoders(args.train)

    # datasets
    train_ds = MultiTaskJsonlDataset(args.train, vocab, label_encoders, max_len=args.max_len)
    val_ds = MultiTaskJsonlDataset(args.val, vocab, label_encoders, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = MultiTaskTextModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        encoder_hidden=args.encoder_hidden,
        num_layers=args.num_layers,
        sentiment_classes=len(label_encoders["sentiment"]),
        genre_classes=len(label_encoders["genre"]),
        emotion_classes=len(label_encoders["emotion"]),
        encoder_type=args.encoder_type,
        max_len=args.max_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            task_mask = {k: v.to(device) for k, v in batch["task_mask"].items()}

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = multitask_loss(outputs, targets, task_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss = evaluate(model, val_loader, device)
        print(f"val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": vocab.itos,
                    "label_classes": {k: enc.classes for k, enc in label_encoders.items()},
                    "config": vars(args),
                },
                os.path.join(args.output, "model.pt"),
            )


if __name__ == "__main__":
    main()


