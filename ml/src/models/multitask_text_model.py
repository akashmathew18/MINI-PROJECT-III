import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskTextModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder_hidden: int,
        num_layers: int,
        sentiment_classes: int,
        genre_classes: int,
        emotion_classes: int,
        encoder_type: str = "transformer",
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional = nn.Embedding(max_len, embed_dim)

        if encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=encoder_hidden,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            pooled_dim = embed_dim
        else:
            self.encoder = nn.LSTM(
                embed_dim,
                encoder_hidden,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            pooled_dim = encoder_hidden * 2

        self.encoder_type = encoder_type
        self.dropout = nn.Dropout(dropout)

        self.head_sentiment = nn.Linear(pooled_dim, sentiment_classes)
        self.head_genre = nn.Linear(pooled_dim, genre_classes)
        self.head_emotion = nn.Linear(pooled_dim, emotion_classes)

    def forward(self, input_ids, attention_mask=None, task_mask=None):
        batch_size, seq_len = input_ids.size()
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(input_ids) + self.positional(pos)

        if self.encoder_type == "transformer":
            src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            lengths = (
                attention_mask.sum(dim=1).cpu() if attention_mask is not None else torch.full((batch_size,), seq_len)
            )
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.encoder(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1))
        else:
            pooled = h.mean(dim=1)

        pooled = self.dropout(pooled)

        logits_sent = self.head_sentiment(pooled)
        logits_genre = self.head_genre(pooled)
        logits_emo = self.head_emotion(pooled)

        return {"sentiment": logits_sent, "genre": logits_genre, "emotion": logits_emo}


def multitask_loss(outputs, targets, task_mask=None, weights=(1.0, 1.0, 0.5)):
    loss_value = 0.0
    total_weight = 0.0
    sentiment_weight, genre_weight, emotion_weight = weights

    if targets.get("sentiment") is not None:
        loss_sent = F.cross_entropy(outputs["sentiment"], targets["sentiment"], reduction="none")
        if task_mask is not None and task_mask.get("sentiment") is not None:
            loss_sent = (loss_sent * task_mask["sentiment"]).sum() / task_mask["sentiment"].clamp_min(1).sum()
        else:
            loss_sent = loss_sent.mean()
        loss_value += sentiment_weight * loss_sent
        total_weight += sentiment_weight

    if targets.get("genre") is not None:
        loss_genre = F.cross_entropy(outputs["genre"], targets["genre"], reduction="none")
        if task_mask is not None and task_mask.get("genre") is not None:
            loss_genre = (loss_genre * task_mask["genre"]).sum() / task_mask["genre"].clamp_min(1).sum()
        else:
            loss_genre = loss_genre.mean()
        loss_value += genre_weight * loss_genre
        total_weight += genre_weight

    if targets.get("emotion") is not None:
        loss_emo = F.cross_entropy(outputs["emotion"], targets["emotion"], reduction="none")
        if task_mask is not None and task_mask.get("emotion") is not None:
            loss_emo = (loss_emo * task_mask["emotion"]).sum() / task_mask["emotion"].clamp_min(1).sum()
        else:
            loss_emo = loss_emo.mean()
        loss_value += emotion_weight * loss_emo
        total_weight += emotion_weight

    return loss_value / max(total_weight, 1e-8)


