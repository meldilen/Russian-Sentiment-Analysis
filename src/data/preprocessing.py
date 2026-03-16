"""
Dataset preprocessing for Russian Sentiment Analysis.
Handles the Russian Sentiment Analysis Dataset (Kaggle) with columns: text, label, src.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


def load_dataset(
    train_path: str = "data/train.csv",
    test_path: Optional[str] = "data/test.csv",
    text_column: str = "text",
    label_column: str = "label",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load Russian sentiment dataset from CSV files.
    
    Expected columns: text, label, src
    Labels: 0 (negative), 1 (neutral), 2 (positive)
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV (optional)
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Tuple of (train_df, test_df or None)
    """
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Please download the Russian Sentiment Analysis Dataset from Kaggle "
            "and place it in the data/ directory."
        )
    
    train_df = pd.read_csv(train_path)
    
    # Validate required columns
    if text_column not in train_df.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available: {list(train_df.columns)}")
    if label_column not in train_df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {list(train_df.columns)}")
    
    # Ensure labels are numeric (0, 1, 2)
    train_df[label_column] = pd.to_numeric(train_df[label_column], errors="coerce")
    train_df = train_df.dropna(subset=[label_column])
    train_df[label_column] = train_df[label_column].astype(int)
    
    test_df = None
    if test_path:
        test_path = Path(test_path)
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            if text_column in test_df.columns and label_column in test_df.columns:
                test_df[label_column] = pd.to_numeric(test_df[label_column], errors="coerce")
                test_df = test_df.dropna(subset=[label_column])
                test_df[label_column] = test_df[label_column].astype(int)
            else:
                test_df = None
    
    return train_df, test_df


def preprocess_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    max_length: Optional[int] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Preprocess dataset for model input.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        label_column: Name of label column
        max_length: Optional max text length (truncation)
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].values
    
    # Basic cleaning: strip whitespace
    texts = [t.strip() for t in texts]
    
    # Filter out empty texts and align labels
    valid_indices = [i for i, t in enumerate(texts) if t]
    texts = [texts[i] for i in valid_indices]
    labels = labels[valid_indices]
    
    if max_length:
        texts = [t[:max_length] if len(t) > max_length else t for t in texts]
    
    return texts, labels
