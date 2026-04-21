"""
Training script for RuBERT 3-class sentiment classifier.

Usage:
  python train.py --config config.yaml
  python train.py --config config.yaml --seed 42
"""

import argparse
import gc
import yaml
import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.preprocessing import load_dataset, preprocess_dataset
from src.model.rubert_classifier import RuBERTClassifier

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def evaluate(model, data_loader, device):
    """Evaluate model accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total if total > 0 else 0.0

def train(config_path: str = "config.yaml", seed: int = 42):
    # Set seed for reproducibility
    set_seed(seed)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model"]
    data_cfg = config["data"]
    
    # Load data
    print("Loading dataset...")
    train_df, test_df = load_dataset(
        train_path=data_cfg["train_path"],
        test_path=data_cfg.get("test_path"),
    )
    
    train_texts, train_labels = preprocess_dataset(train_df)
    
    print(f"Train samples: {len(train_texts)}")
    
    if test_df is not None:
        test_texts, test_labels = preprocess_dataset(test_df)
        print(f"Test samples: {len(test_texts)}")
    else:
        test_texts, test_labels = [], []
        print("No test data provided, skipping validation")
    
    # Model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    
    train_dataset = SentimentDataset(
        train_texts,
        train_labels,
        tokenizer,
        max_length=model_cfg["max_length"],
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # Create test loader if test data exists
    test_loader = None
    if test_texts:
        test_dataset = SentimentDataset(
            test_texts,
            test_labels,
            tokenizer,
            max_length=model_cfg["max_length"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=model_cfg["batch_size"],
            shuffle=False,
            num_workers=0,
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = RuBERTClassifier(
        model_name=model_cfg["name"],
        num_labels=model_cfg["num_labels"],
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
    )
    
    total_steps = len(train_loader) * model_cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )
    
    criterion = torch.nn.CrossEntropyLoss()

    # Create checkpoints directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    best_accuracy = 0.0
    
    # Training loop    
    for epoch in range(model_cfg["epochs"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_cfg['epochs']}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        # Validation after each epoch
        if test_loader:
            accuracy = evaluate(model, test_loader, device)
            print(f"  Test Accuracy: {accuracy:.4f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_path = checkpoint_dir / "rubert_sentiment_best.pt"
                torch.save(model.state_dict(), save_path)
                print(f"  Best model saved to {save_path} (acc={accuracy:.4f})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    final_save_path = checkpoint_dir / "rubert_sentiment.pt"
    torch.save(model.state_dict(), final_save_path)
    print(f"\nFinal model saved to {final_save_path}")

    # Save training config alongside model
    config_save_path = checkpoint_dir / "training_config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        # Add training metadata
        training_info = {
            "training_config": config,
            "best_accuracy": best_accuracy,
            "seed": seed,
            "device": device,
        }
        yaml.dump(training_info, f, allow_unicode=True, default_flow_style=False)
    print(f"Training config saved to {config_save_path}")
    
    if best_accuracy > 0:
        print(f"\nBest test accuracy: {best_accuracy:.4f}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RuBERT sentiment classifier")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    train(args.config, seed=args.seed)
