"""
Create sample Russian sentiment data for testing when Kaggle dataset is unavailable.
Generates a small balanced dataset with synthetic Russian reviews.
"""

import pandas as pd
import random
from pathlib import Path
from typing import Optional

# Sample Russian texts with labels (0=negative, 1=neutral, 2=positive)
SAMPLE_DATA = [
    # Negative (0)
    ("Ужасный фильм, полная ерунда. Не рекомендую.", 0, "sample"),
    ("Сервис отвратительный, никогда больше не вернусь.", 0, "sample"),
    ("Товар пришёл сломанный, очень разочарован.", 0, "sample"),
    ("Плохое качество, деньги на ветер.", 0, "sample"),
    ("Самый худший опыт в моей жизни.", 0, "sample"),
    ("Не понравилось совсем.", 0, "sample"),
    ("Разочарован результатом.", 0, "sample"),
    ("Качество оставляет желать лучшего.", 0, "sample"),
    # Neutral (1)
    ("Обычный продукт, ничего особенного.", 1, "sample"),
    ("Средне по качеству, как ожидалось.", 1, "sample"),
    ("Нормально, без сюрпризов.", 1, "sample"),
    ("Приемлемо для данной цены.", 1, "sample"),
    ("Стандартный уровень сервиса.", 1, "sample"),
    ("Ничего примечательного.", 1, "sample"),
    ("Так себе, средне.", 1, "sample"),
    ("В целом неплохо.", 1, "sample"),
    # Positive (2)
    ("Отличный фильм! Очень рекомендую к просмотру.", 2, "sample"),
    ("Превосходное качество, доволен покупкой.", 2, "sample"),
    ("Замечательный сервис, буду заказывать снова.", 2, "sample"),
    ("Супер! Всё понравилось, спасибо.", 2, "sample"),
    ("Лучший продукт в этой категории.", 2, "sample"),
    ("Очень круто!", 2, "sample"),
    ("Потрясающе, в восторге!", 2, "sample"),
    ("Рекомендую всем!", 2, "sample"),
]

# Templates for generating more variations
NEGATIVE_TEMPLATES = [
    "Ужасный {}, не рекомендую.", "Плохой {}, разочарован.", "{} оставляет желать лучшего.",
    "Не стоит своих денег.", "Качество {} отвратительное.", "{} полная ерунда."
]

NEUTRAL_TEMPLATES = [
    "Обычный {}, ничего особенного.", "{} нормальный, без сюрпризов.", "{} приемлемого качества.",
    "Средний {}, как ожидалось.", "{} стандартного уровня."
]

POSITIVE_TEMPLATES = [
    "Отличный {}, рекомендую!", "Превосходный {}, очень доволен.", "{} высшего качества!",
    "Замечательный {}, буду заказывать снова.", "Лучший {} в этой категории.", "{} супер!"
]

PRODUCTS = ["товар", "продукт", "сервис", "фильм", "книга", "ресторан"]


def generate_sample_texts(label: int, count: int) -> list:
    """Generate synthetic texts for a given label."""
    texts = []
    
    if label == 0:  # negative
        for _ in range(count):
            product = random.choice(PRODUCTS)
            template = random.choice(NEGATIVE_TEMPLATES)
            texts.append(template.format(product))
    elif label == 1:  # neutral
        for _ in range(count):
            product = random.choice(PRODUCTS)
            template = random.choice(NEUTRAL_TEMPLATES)
            texts.append(template.format(product))
    else:  # positive
        for _ in range(count):
            product = random.choice(PRODUCTS)
            template = random.choice(POSITIVE_TEMPLATES)
            texts.append(template.format(product))
    
    return texts


def create_sample_data(sample_size: int = 100, output_dir: str = "data", random_state: int = 42):
    """
    Create sample dataset for quick testing.
    
    Args:
        sample_size: Total number of samples (approx, will be balanced)
        output_dir: Directory to save the CSV files
        random_state: Random seed for reproducibility
    """
    random.seed(random_state)
    
    # Calculate samples per class (balanced)
    samples_per_class = sample_size // 3
    remainder = sample_size - (samples_per_class * 3)
    
    # First, use base samples
    base_samples_by_label = {0: [], 1: [], 2: []}
    for text, label, src in SAMPLE_DATA:
        base_samples_by_label[label].append((text, label, src))
    
    # Build final dataset
    all_data = []
    
    for label in [0, 1, 2]:
        # Add base samples first
        base_count = len(base_samples_by_label[label])
        all_data.extend(base_samples_by_label[label])
        
        # Generate additional samples if needed
        needed = samples_per_class - base_count
        if label == 2 and remainder > 0 and label == 0:  # Add remainder to positive
            needed += remainder
        elif label == 2:  # positive gets extra
            needed += remainder
        
        if needed > 0:
            generated = generate_sample_texts(label, needed)
            for text in generated:
                all_data.append((text, label, "generated"))
    
    # Shuffle
    random.shuffle(all_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data, columns=["text", "label", "src"])
    
    # Save to CSV (single file, as expected by load_dataset)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "sentiment_dataset.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Created {output_path} ({len(df)} samples)")
    print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
    print(f"  (0=negative, 1=neutral, 2=positive)")
    
    return output_path


def create_sample_data_train_test(sample_size: int = 100, output_dir: str = "data", random_state: int = 42):
    """
    Create sample dataset split into train/test files.
    This is an alternative for workflows expecting separate files.
    """
    output_path = create_sample_data(sample_size, output_dir, random_state)
    
    # Split into train/test
    df = pd.read_csv(output_path)
    train_df = df.sample(frac=0.8, random_state=random_state)
    test_df = df.drop(train_df.index)
    
    train_path = Path(output_dir) / "train.csv"
    test_path = Path(output_dir) / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Also created train/test split:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Test: {test_path} ({len(test_df)} samples)")
    
    return train_path, test_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100, help="Sample size")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--split", action="store_true", help="Create train/test split")
    args = parser.parse_args()
    
    if args.split:
        create_sample_data_train_test(args.size, args.output_dir)
    else:
        create_sample_data(args.size, args.output_dir)