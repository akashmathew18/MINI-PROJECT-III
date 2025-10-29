"""
Performance Metrics Evaluation Script
Generates Confusion Matrix, Accuracy, Precision, Recall for all tasks
"""

import argparse
import os
import json
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.datasets import MultiTaskJsonlDataset, build_vocab, LabelEncoder, collate_batch
from src.models.multitask_text_model import MultiTaskTextModel


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MultiTask Model Performance")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (model.pt)")
    p.add_argument("--val", required=True, help="Path to validation JSONL file")
    p.add_argument("--output_dir", default="evaluation_results", help="Directory to save results")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def collect_texts(jsonl_path: str) -> List[str]:
    """Collect all texts from JSONL file"""
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj.get("text", ""))
    return texts


def fit_label_encoders(train_path: str) -> Dict[str, LabelEncoder]:
    """Build label encoders from training data"""
    encs: Dict[str, LabelEncoder] = {
        "sentiment": LabelEncoder(),
        "genre": LabelEncoder(),
        "emotion": LabelEncoder()
    }
    labels = {"sentiment": [], "genre": [], "emotion": []}
    
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


def evaluate_model(model, loader, device, label_encoders):
    """
    Evaluate model on validation set
    Returns predictions and ground truth for each task
    """
    model.eval()
    
    # Storage for predictions and ground truth
    results = {
        "sentiment": {"preds": [], "targets": [], "masks": []},
        "genre": {"preds": [], "targets": [], "masks": []},
        "emotion": {"preds": [], "targets": [], "masks": []}
    }
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            task_mask = {k: v.to(device) for k, v in batch["task_mask"].items()}
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            
            # Process each task
            for task in ["sentiment", "genre", "emotion"]:
                # Get predicted labels (argmax)
                preds = outputs[task].argmax(dim=-1).cpu().numpy()
                targets_np = targets[task].cpu().numpy()
                masks_np = task_mask[task].cpu().numpy()
                
                results[task]["preds"].extend(preds)
                results[task]["targets"].extend(targets_np)
                results[task]["masks"].extend(masks_np)
    
    # Convert to numpy arrays
    for task in results:
        results[task]["preds"] = np.array(results[task]["preds"])
        results[task]["targets"] = np.array(results[task]["targets"])
        results[task]["masks"] = np.array(results[task]["masks"])
    
    return results


def compute_metrics(results, label_encoders, output_dir):
    """
    Compute and display metrics for all tasks
    Save confusion matrices as images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PERFORMANCE METRICS EVALUATION")
    print("=" * 80)
    
    all_metrics = {}
    
    for task in ["sentiment", "genre", "emotion"]:
        print(f"\n{'=' * 80}")
        print(f"TASK: {task.upper()}")
        print(f"{'=' * 80}")
        
        # Get data
        preds = results[task]["preds"]
        targets = results[task]["targets"]
        masks = results[task]["masks"]
        
        # Filter out masked (missing) labels
        valid_idx = masks > 0
        preds_valid = preds[valid_idx]
        targets_valid = targets[valid_idx]
        
        if len(targets_valid) == 0:
            print(f"⚠️  No valid samples for {task}")
            continue
        
        # Get class names
        class_names = label_encoders[task].classes
        
        # Compute metrics
        accuracy = accuracy_score(targets_valid, preds_valid)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets_valid, preds_valid, average=None, zero_division=0
        )
        
        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            targets_valid, preds_valid, average='macro', zero_division=0
        )
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            targets_valid, preds_valid, average='weighted', zero_division=0
        )
        
        # Store metrics
        all_metrics[task] = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1
        }
        
        # Display results
        print(f"\n📊 Overall Metrics:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n📈 Macro Averages:")
        print(f"   Precision: {macro_precision:.4f}")
        print(f"   Recall:    {macro_recall:.4f}")
        print(f"   F1-Score:  {macro_f1:.4f}")
        print(f"\n📉 Weighted Averages:")
        print(f"   Precision: {weighted_precision:.4f}")
        print(f"   Recall:    {weighted_recall:.4f}")
        print(f"   F1-Score:  {weighted_f1:.4f}")
        
        # Per-class metrics
        print(f"\n📋 Per-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 65)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                  f"{f1[i]:<12.4f} {support[i]:<10}")
        
        # Classification report
        print(f"\n📄 Detailed Classification Report:")
        print(classification_report(
            targets_valid, preds_valid, 
            target_names=class_names, 
            zero_division=0
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(targets_valid, preds_valid)
        
        # Display confusion matrix as text
        print(f"\n🔢 Confusion Matrix:")
        print(f"{'':>15}", end="")
        for name in class_names:
            print(f"{name[:10]:>12}", end="")
        print()
        print("-" * (15 + 12 * len(class_names)))
        for i, true_class in enumerate(class_names):
            print(f"{true_class[:15]:>15}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i][j]:>12}", end="")
            print()
        
        # Visualize and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            square=True,
            linewidths=1,
            linecolor='gray'
        )
        plt.title(f'Confusion Matrix - {task.capitalize()}', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        cm_path = os.path.join(output_dir, f'confusion_matrix_{task}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Confusion matrix saved to: {cm_path}")
        plt.close()
    
    # Save summary metrics to JSON
    summary_path = os.path.join(output_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n💾 Metrics summary saved to: {summary_path}")
    
    # Create comparison visualization
    create_comparison_plot(all_metrics, output_dir)
    
    return all_metrics


def create_comparison_plot(metrics, output_dir):
    """Create a bar plot comparing metrics across tasks"""
    tasks = list(metrics.keys())
    
    if not tasks:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy comparison
    accuracies = [metrics[task]['accuracy'] * 100 for task in tasks]
    axes[0].bar(tasks, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_title('Accuracy by Task', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Precision comparison
    precisions = [metrics[task]['weighted_precision'] * 100 for task in tasks]
    axes[1].bar(tasks, precisions, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1].set_title('Weighted Precision by Task', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Precision (%)', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Recall comparison
    recalls = [metrics[task]['weighted_recall'] * 100 for task in tasks]
    axes[2].bar(tasks, recalls, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[2].set_title('Weighted Recall by Task', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Recall (%)', fontsize=12)
    axes[2].set_ylim(0, 100)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Comparison plot saved to: {comparison_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        print(f"\n⚠️  BASELINE MODE: No trained model available")
        print(f"   Genre prediction would use keyword-based classification")
        print(f"   (No ML metrics applicable for baseline)")
        return
    
    print(f"🔍 Loading checkpoint from: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    
    # Rebuild vocab
    vocab_list = ckpt["vocab"]
    from src.data.datasets import Vocab
    vocab = Vocab()
    vocab.itos = list(vocab_list)
    vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
    
    # Get label classes
    label_classes = ckpt["label_classes"]
    
    # Rebuild label encoders
    label_encoders = {}
    for task in ["sentiment", "genre", "emotion"]:
        enc = LabelEncoder()
        enc.classes = label_classes[task]
        enc.to_index = {c: i for i, c in enumerate(enc.classes)}
        label_encoders[task] = enc
    
    print(f"📊 Label classes loaded:")
    for task, enc in label_encoders.items():
        print(f"   {task}: {len(enc.classes)} classes - {enc.classes}")
    
    # Create model
    model = MultiTaskTextModel(
        vocab_size=len(vocab),
        embed_dim=cfg.get("embed_dim", 128),
        encoder_hidden=cfg.get("encoder_hidden", 256),
        num_layers=cfg.get("num_layers", 2),
        sentiment_classes=len(label_encoders["sentiment"]),
        genre_classes=len(label_encoders["genre"]),
        emotion_classes=len(label_encoders["emotion"]),
        encoder_type=cfg.get("encoder_type", "transformer"),
        max_len=cfg.get("max_len", 256),
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Model loaded successfully")
    
    # Load validation dataset
    val_ds = MultiTaskJsonlDataset(
        args.val, 
        vocab, 
        label_encoders, 
        max_len=cfg.get("max_len", 256)
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_batch
    )
    
    print(f"📁 Validation samples: {len(val_ds)}")
    
    # Evaluate
    print(f"\n🔄 Running evaluation...")
    results = evaluate_model(model, val_loader, device, label_encoders)
    
    # Compute and display metrics
    metrics = compute_metrics(results, label_encoders, args.output_dir)
    
    print(f"\n{'=' * 80}")
    print(f"✨ EVALUATION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"📂 Results saved to: {args.output_dir}")
    print(f"   - Confusion matrices (PNG)")
    print(f"   - Metrics summary (JSON)")
    print(f"   - Comparison plot (PNG)")


if __name__ == "__main__":
    main()
