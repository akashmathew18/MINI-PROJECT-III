import json
from typing import Dict, List, Optional, Tuple

import orjson
import torch
from torch.utils.data import Dataset


def _read_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
            except Exception:
                obj = json.loads(line.decode("utf-8"))
            records.append(obj)
    return records


class Vocab:
    def __init__(self, specials: Optional[List[str]] = None):
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}
        specials = specials or ["<pad>", "<unk>"]
        for token in specials:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        if token not in self.stoi:
            idx = len(self.itos)
            self.stoi[token] = idx
            self.itos.append(token)
        return self.stoi[token]

    def __len__(self) -> int:
        return len(self.itos)

    def lookup(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])


def build_vocab(texts: List[str], min_freq: int = 2) -> Vocab:
    from collections import Counter

    counter: Counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    vocab = Vocab()
    for tok, freq in counter.items():
        if freq >= min_freq:
            vocab.add_token(tok)
    return vocab


def simple_tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


def encode_text(text: str, vocab: Vocab, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = simple_tokenize(text)
    ids = [vocab.lookup(tok) for tok in tokens][:max_len]
    attn = [1] * len(ids)
    # pad
    while len(ids) < max_len:
        ids.append(0)
        attn.append(0)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


class LabelEncoder:
    def __init__(self):
        self.classes: List[str] = []
        self.to_index: Dict[str, int] = {}

    def fit(self, labels: List[str]):
        uniq = []
        seen = set()
        for l in labels:
            if l is None:
                continue
            if l not in seen:
                uniq.append(l)
                seen.add(l)
        self.classes = uniq
        self.to_index = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.classes)

    def encode(self, label: Optional[str]) -> Optional[int]:
        if label is None:
            return None
        return self.to_index[label]


class MultiTaskJsonlDataset(Dataset):
    def __init__(self, path: str, vocab: Vocab, label_encoders: Dict[str, LabelEncoder], max_len: int = 256):
        self.records = _read_jsonl(path)
        self.vocab = vocab
        self.label_encoders = label_encoders
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        item = self.records[idx]
        text = item.get("text", "")
        input_ids, attention_mask = encode_text(text, self.vocab, self.max_len)

        y_sent = item.get("sentiment")
        y_genre = item.get("genre")
        y_emo = item.get("emotion")

        t_sent = (
            torch.tensor(self.label_encoders["sentiment"].encode(y_sent), dtype=torch.long)
            if y_sent is not None
            else None
        )
        t_genre = (
            torch.tensor(self.label_encoders["genre"].encode(y_genre), dtype=torch.long) if y_genre is not None else None
        )
        t_emo = (
            torch.tensor(self.label_encoders["emotion"].encode(y_emo), dtype=torch.long) if y_emo is not None else None
        )

        mask = {
            "sentiment": torch.tensor(1 if y_sent is not None else 0, dtype=torch.float),
            "genre": torch.tensor(1 if y_genre is not None else 0, dtype=torch.float),
            "emotion": torch.tensor(1 if y_emo is not None else 0, dtype=torch.float),
        }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": {"sentiment": t_sent, "genre": t_genre, "emotion": t_emo},
            "task_mask": mask,
        }


def collate_batch(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])

    targets = {}
    task_mask = {}
    for t in ["sentiment", "genre", "emotion"]:
        labels = [b["targets"][t] for b in batch]
        if any(l is None for l in labels):
            # replace Nones with zeros; mask will null them out
            labels = [torch.tensor(0, dtype=torch.long) if l is None else l for l in labels]
        targets[t] = torch.stack(labels)
        task_mask[t] = torch.stack([b["task_mask"][t] for b in batch])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "targets": targets, "task_mask": task_mask}


