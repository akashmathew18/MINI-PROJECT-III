#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Downloader for JV Cinelytics

Downloads a subset of the TurkuNLP/genre-6 dataset for genre classification training.
Saves the data in both JSONL and CSV formats.
"""

import os
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_dataset, Dataset
except ImportError:
    print("❌ Error: 'datasets' library not installed.")
    print("Please install it using: pip install datasets")
    exit(1)


def download_genre_dataset(num_samples=10000, output_dir="."):
    """
    Download and save a subset of the genre-6 dataset.
    
    Args:
        num_samples (int): Number of samples to download (default: 10000)
        output_dir (str): Directory to save the output files (default: current directory)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Print current working directory for debugging
        print(f"📂 Current working directory: {os.getcwd()}")
        print(f"📂 Output directory (absolute path): {output_path.resolve()}")
        
        print(f"📥 Loading TurkuNLP/genre-6 dataset (streaming mode)...")
        
        # 1. Load the dataset in streaming mode
        dataset = load_dataset("TurkuNLP/genre-6", split="train", streaming=True)
        
        print(f"📊 Taking first {num_samples:,} entries...")
        
        # 2. Take the first N entries with progress bar
        subset_list = []
        for i, item in enumerate(tqdm(dataset, total=num_samples, desc="Downloading")):
            if i >= num_samples:
                break
            subset_list.append(item)
        
        if len(subset_list) == 0:
            print("❌ Error: No data was downloaded.")
            return False
        
        print(f"✅ Downloaded {len(subset_list):,} entries")
        
        # 3. Save directly to JSONL (bypassing Dataset.from_list to avoid pickling issues)
        jsonl_path = output_path / "genre-6-subset-10k.jsonl"
        print(f"💾 Saving to {jsonl_path}...")
        
        import json
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in subset_list:
                f.write(json.dumps(item) + '\n')
        
        print(f"✅ Successfully saved JSONL: {jsonl_path}")
        
        # 4. Save as CSV
        csv_path = output_path / "genre-6-subset-10k.csv"
        print(f"💾 Saving to {csv_path}...")
        
        # Convert to pandas DataFrame for CSV
        import pandas as pd
        df = pd.DataFrame(subset_list)
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Successfully saved CSV: {csv_path}")
        
        # 5. Print dataset info
        print("\n📋 Dataset Information:")
        print(f"  - Total samples: {len(subset_list):,}")
        if subset_list:
            print(f"  - Features: {list(subset_list[0].keys())}")
        print(f"  - JSONL file size: {jsonl_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"  - CSV file size: {csv_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Print absolute paths
        print(f"\n💾 Files saved at:")
        print(f"  JSONL: {jsonl_path.resolve()}")
        print(f"  CSV:   {csv_path.resolve()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🎬 JV Cinelytics - Genre Dataset Downloader\n")
    
    # Download 10,000 samples
    success = download_genre_dataset(
        num_samples=10000,
        output_dir="."
    )
    
    if success:
        print("\n🎉 Dataset download completed successfully!")
        print("\nYou can now use these files for training:")
        print("  - genre-6-subset-10k.jsonl (for ML training)")
        print("  - genre-6-subset-10k.csv (for analysis/inspection)")
    else:
        print("\n❌ Dataset download failed. Please check the errors above.")
        exit(1)