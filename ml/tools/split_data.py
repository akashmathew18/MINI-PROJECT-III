"""
Data Splitting Utility
Splits a JSONL file into train and validation sets
"""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split JSONL data into train/val sets")
    p.add_argument("--input", required=True, help="Input JSONL file")
    p.add_argument("--output_dir", default="../data", help="Output directory")
    p.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--stratify", action="store_true", help="Stratify by genre")
    return p.parse_args()


def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def stratified_split(data, train_ratio, seed):
    """Split data with stratification by genre"""
    from collections import defaultdict
    
    # Group by genre
    genre_groups = defaultdict(list)
    for item in data:
        genre = item.get('genre', 'unknown')
        genre_groups[genre].append(item)
    
    train_data = []
    val_data = []
    
    # Split each genre group
    random.seed(seed)
    for genre, items in genre_groups.items():
        random.shuffle(items)
        split_idx = int(len(items) * train_ratio)
        train_data.extend(items[:split_idx])
        val_data.extend(items[split_idx:])
    
    # Shuffle again
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    return train_data, val_data


def simple_split(data, train_ratio, seed):
    """Simple random split"""
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def main():
    args = parse_args()
    
    print("=" * 70)
    print("DATA SPLITTING UTILITY")
    print("=" * 70)
    
    # Load data
    print(f"\n📁 Loading data from: {args.input}")
    data = load_jsonl(args.input)
    print(f"✅ Loaded {len(data)} samples")
    
    # Analyze data
    genres = [d.get('genre') for d in data if d.get('genre')]
    sentiments = [d.get('sentiment') for d in data if d.get('sentiment')]
    emotions = [d.get('emotion') for d in data if d.get('emotion')]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(data)}")
    print(f"   Samples with genre: {len(genres)}")
    print(f"   Samples with sentiment: {len(sentiments)}")
    print(f"   Samples with emotion: {len(emotions)}")
    
    if genres:
        from collections import Counter
        genre_counts = Counter(genres)
        print(f"\n🎬 Genre Distribution:")
        for genre, count in genre_counts.most_common():
            print(f"   {genre:15s}: {count:4d} ({count/len(genres)*100:.1f}%)")
    
    # Split data
    print(f"\n✂️  Splitting data (train: {args.train_ratio:.0%}, val: {1-args.train_ratio:.0%})")
    
    if args.stratify and genres:
        print(f"   Using stratified split by genre")
        train_data, val_data = stratified_split(data, args.train_ratio, args.seed)
    else:
        print(f"   Using simple random split")
        train_data, val_data = simple_split(data, args.train_ratio, args.seed)
    
    print(f"✅ Train samples: {len(train_data)}")
    print(f"✅ Val samples: {len(val_data)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    print(f"\n💾 Saving splits...")
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    
    print(f"✅ Train data saved to: {train_path}")
    print(f"✅ Val data saved to: {val_path}")
    
    # Verify splits
    if args.stratify and genres:
        train_genres = [d.get('genre') for d in train_data if d.get('genre')]
        val_genres = [d.get('genre') for d in val_data if d.get('genre')]
        
        print(f"\n📊 Genre Distribution in Splits:")
        from collections import Counter
        train_counter = Counter(train_genres)
        val_counter = Counter(val_genres)
        
        print(f"\n{'Genre':<15} {'Train':>10} {'Val':>10} {'Total':>10}")
        print("-" * 50)
        for genre in sorted(set(train_genres + val_genres)):
            t_count = train_counter.get(genre, 0)
            v_count = val_counter.get(genre, 0)
            total = t_count + v_count
            print(f"{genre:<15} {t_count:>10} {v_count:>10} {total:>10}")
    
    print(f"\n{'=' * 70}")
    print(f"✨ DATA SPLITTING COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
