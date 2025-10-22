import argparse
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to summarizer directory (from save_pretrained)")
    p.add_argument("--input_file", required=False, help="Path to .txt script to summarize")
    p.add_argument("--text", required=False, help="Raw text to summarize")
    p.add_argument("--max_len", type=int, default=220)
    p.add_argument("--min_len", type=int, default=80)
    return p.parse_args()


def summarize_text(model_dir: str, text: str, max_len: int = 220, min_len: int = 80) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

    # Chunk long text
    sentences = sent_tokenize(text)
    chunks: List[str] = []
    buf = ""
    max_tokens = 768
    for s in sentences:
        if len(tok.encode(buf + " " + s, add_special_tokens=True)) > max_tokens:
            if buf:
                chunks.append(buf)
                buf = s
        else:
            buf = (buf + " " + s).strip()
    if buf:
        chunks.append(buf)

    partial_summaries: List[str] = []
    for c in chunks:
        inputs = tok([c], max_length=1024, truncation=True, return_tensors="pt")
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=max_len,
                min_length=min_len,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
        s = tok.decode(summary_ids[0], skip_special_tokens=True)
        partial_summaries.append(s)

    stitched = " ".join(partial_summaries)
    # Second pass
    inputs = tok([stitched], max_length=1024, truncation=True, return_tensors="pt")
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
    final = tok.decode(summary_ids[0], skip_special_tokens=True)
    return final


def main():
    args = parse_args()
    if not args.text and not args.input_file:
        raise SystemExit("Provide either --text or --input_file")
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text
    print(summarize_text(args.checkpoint, text, args.max_len, args.min_len))


if __name__ == "__main__":
    main()


