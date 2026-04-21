"""
Dataset preprocessing for Russian Sentiment Analysis.
Handles the Russian Sentiment Analysis Dataset (Kaggle) with columns: text, label, src.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


def load_dataset(
    train_path: str = "data/sentiment_dataset.csv",
    test_path: Optional[str] = None,
    text_column: str = "text",
    label_column: str = "label",
    val_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load Russian sentiment dataset from CSV files.
    
    Expected columns: text, label, src
    Labels: 0 (negative), 1 (neutral), 2 (positive)
    
    Args:
        train_path: Path to CSV file (or single dataset file)
        test_path: Path to test CSV (optional, if None, split train data)
        text_column: Name of text column
        label_column: Name of label column
        val_split: Fraction of data to use for validation (if test_path is None)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df or None)
    """
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Please download the Russian Sentiment Analysis Dataset from Kaggle "
            "and place it in the data/ directory."
        )
    
    df = pd.read_csv(train_path)
    
    # Validate required columns
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {list(df.columns)}")
    
    # Ensure labels are numeric (0, 1, 2)
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
    df = df.dropna(subset=[label_column])
    df[label_column] = df[label_column].astype(int)
    
    # If test_path is provided, use it as validation set
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
        else:
            test_df = None
        return df, test_df
    
    # Otherwise, split the single dataset into train/val
    print(f"Total samples: {len(df)}")
    print(f"Splitting into train/val with val_split={val_split}")
    
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=random_state,
        stratify=df[label_column],  # maintain class balance
    )
    
    print(f"Train size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    
    # Print class distribution
    print("\nTrain class distribution:")
    print(train_df[label_column].value_counts().sort_index().to_dict())
    print("Val class distribution:")
    print(val_df[label_column].value_counts().sort_index().to_dict())
    
    return train_df, val_df


def remove_duplicates(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    keep: str = "first",
) -> pd.DataFrame:
    """
    Remove duplicate texts from the dataset.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        label_column: Name of label column
        keep: Which duplicates to keep ('first', 'last', or False)
        
    Returns:
        DataFrame with duplicates removed
    """
    original_len = len(df)
    df_cleaned = df.drop_duplicates(subset=[text_column], keep=keep)
    removed_count = original_len - len(df_cleaned)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate texts ({removed_count/original_len*100:.1f}% of data)")
    
    return df_cleaned


def preprocess_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    remove_duplicate_texts: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Preprocess dataset for model input.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        label_column: Name of label column
        remove_duplicate_texts: Whether to remove duplicate texts
        
    Returns:
        Tuple of (texts, labels)
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Remove duplicates if requested
    if remove_duplicate_texts:
        df = remove_duplicates(df, text_column, label_column)
    
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].values
    
    # Basic cleaning: strip whitespace
    texts = [t.strip() for t in texts]
    
    # Filter out empty texts and align labels
    valid_indices = [i for i, t in enumerate(texts) if t]
    texts = [texts[i] for i in valid_indices]
    labels = labels[valid_indices]
    
    # Print statistics
    print(f"Final dataset size: {len(texts)} samples")
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    return texts, labels