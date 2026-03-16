"""
Training script for RuBERT 3-class sentiment classifier.
"""

import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.preprocessing import load_dataset, preprocess_dataset
from src.model.rubert_classifier import RuBERTClassifier


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


def train(config_path: str = "config.yaml"):
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
    
    train_texts, train_labels = preprocess_dataset(
        train_df,
        max_length=model_cfg["max_length"],
    )
    
    print(f"Train samples: {len(train_texts)}")
    
    # Model and tokenizer
    from transformers import AutoTokenizer
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RuBERTClassifier(
        model_name=model_cfg["name"],
        num_labels=model_cfg["num_labels"],
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg["learning_rate"],
    )
    
    total_steps = len(train_loader) * model_cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    Path("checkpoints").mkdir(exist_ok=True)
    
    for epoch in range(model_cfg["epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
    
    # Save model
    save_path = Path("checkpoints/rubert_sentiment.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    train(args.config)
