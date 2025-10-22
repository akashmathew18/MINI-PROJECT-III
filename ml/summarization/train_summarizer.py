import argparse
import json
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


class JsonlSumDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_source_len: int = 1024, max_target_len: int = 256):
        self.examples: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "document" in obj and "summary" in obj:
                    self.examples.append({"document": obj["document"], "summary": obj["summary"]})
        self.tok = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        src = self.tok(
            ex["document"],
            truncation=True,
            max_length=self.max_source_len,
            padding="max_length",
            return_tensors="pt",
        )
        tgt = self.tok(
            ex["summary"],
            truncation=True,
            max_length=self.max_target_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": tgt["input_ids"].squeeze(0),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train.jsonl with fields: document, summary")
    p.add_argument("--val", required=True, help="Path to val.jsonl with fields: document, summary")
    p.add_argument("--model_name", default="sshleifer/distilbart-cnn-12-6")
    p.add_argument("--output", default="runs/summarizer")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_source_len", type=int, default=1024)
    p.add_argument("--max_target_len", type=int, default=256)
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total += float(out.loss.item())
            steps += 1
    return total / max(steps, 1)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    train_ds = JsonlSumDataset(args.train, tokenizer, args.max_source_len, args.max_target_len)
    val_ds = JsonlSumDataset(args.val, tokenizer, args.max_source_len, args.max_target_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, num_training_steps // 20), num_training_steps=num_training_steps
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss = evaluate(model, val_loader, device)
        print(f"val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()


