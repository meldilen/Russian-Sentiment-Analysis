"""
Create sample Russian sentiment data for testing when Kaggle dataset is unavailable.
Generates a small balanced dataset with synthetic Russian reviews.
"""

import pandas as pd
from pathlib import Path

# Sample Russian texts with labels (0=negative, 1=neutral, 2=positive)
SAMPLE_DATA = [
    # Negative
    ("Ужасный фильм, полная ерунда. Не рекомендую.", 0, "review"),
    ("Сервис отвратительный, никогда больше не вернусь.", 0, "review"),
    ("Товар пришёл сломанный, очень разочарован.", 0, "review"),
    ("Плохое качество, деньги на ветер.", 0, "review"),
    ("Самый худший опыт в моей жизни.", 0, "review"),
    # Neutral
    ("Обычный продукт, ничего особенного.", 1, "review"),
    ("Средне по качеству, как ожидалось.", 1, "review"),
    ("Нормально, без сюрпризов.", 1, "review"),
    ("Приемлемо для данной цены.", 1, "review"),
    ("Стандартный уровень сервиса.", 1, "review"),
    # Positive
    ("Отличный фильм! Очень рекомендую к просмотру.", 2, "review"),
    ("Превосходное качество, доволен покупкой.", 2, "review"),
    ("Замечательный сервис, буду заказывать снова.", 2, "review"),
    ("Супер! Всё понравилось, спасибо.", 2, "review"),
    ("Лучший продукт в этой категории.", 2, "review"),
]

# Extended samples for more robust training
EXTRA_SAMPLES = [
    ("Не понравилось совсем.", 0, "review"),
    ("Разочарован результатом.", 0, "review"),
    ("Качество оставляет желать лучшего.", 0, "review"),
    ("Ничего примечательного.", 1, "review"),
    ("Так себе, средне.", 1, "review"),
    ("В целом неплохо.", 1, "review"),
    ("Очень круто!", 2, "review"),
    ("Потрясающе, в восторге!", 2, "review"),
    ("Рекомендую всем!", 2, "review"),
]


def create_sample_data(output_dir: str = "data"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_data = SAMPLE_DATA + EXTRA_SAMPLES
    df = pd.DataFrame(all_data, columns=["text", "label", "src"])
    
    # Split 80/20 for train/test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    train_path = Path(output_dir) / "train.csv"
    test_path = Path(output_dir) / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Created {train_path} ({len(train_df)} samples)")
    print(f"Created {test_path} ({len(test_df)} samples)")
    print("Label distribution:", df["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    create_sample_data()
