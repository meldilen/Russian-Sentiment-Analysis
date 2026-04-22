"""
Interactive Web Application for Comparing Vanilla LIME vs Stability-Enhanced LIME
Russian Sentiment Analysis with RuBERT
Professional UI/UX implementation - Russian Language Interface
"""

from src.utils.metrics import (
    compute_stability_score,
    compute_faithfulness,
    compute_sparsity,
    compute_rank_correlation,
)
from src.xai.stability_lime import StabilityEnhancedLIME
from src.xai.lime_text import LimeTextExplainer, _tokenize_russian
from src.model.rubert_classifier import RuBERTPipeline
from datetime import datetime
import time
import os
import sys
from pathlib import Path
import yaml
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
import logging
import warnings
warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Page configuration
st.set_page_config(
    page_title="Сравнение LIME | Анализ тональности русских текстов",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background-color: #1a1a2e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #16213e;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.25rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.875rem;
        font-weight: 500;
        opacity: 0.8;
        letter-spacing: 0.5px;
    }
    .metric-card h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 0.875rem;
        opacity: 0.9;
    }
    
    /* Color variants */
    .metric-primary {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .metric-success {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    }
    .metric-warning {
        background: linear-gradient(135deg, #e94560 0%, #c73e56 100%);
    }
    
    /* Word highlighting */
    .word-span {
        display: inline-block;
        margin: 2px;
        padding: 2px 6px;
        border-radius: 4px;
        transition: all 0.2s ease;
        cursor: pointer;
        font-size: 1.1em;
    }
    .word-span:hover {
        transform: scale(1.02);
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #f5f5f5;
        padding: 0.25rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        color: #666;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8e8e8;
        color: #1a1a2e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a2e;
        color: white;
        border-bottom: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* Info box */
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #1a1a2e;
        font-size: 0.875rem;
        color: #555;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #1a1a2e;
    }
    
    /* Caption styling */
    .caption {
        font-size: 0.75rem;
        color: #888;
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("Vanilla LIME vs Stability-Enhanced LIME")
st.caption(
    "Интерактивное сравнение методов объяснения для анализа тональности русских текстов")
# Create tabs
tab_theory, tab_problem, tab_solution, tab_analyze = st.tabs([
    "Что такое LIME?",
    "Проблема нестабильности",
    "Наше решение",
    "Анализ текста"
])


with tab_theory:
    st.markdown(r"""
    
    LIME (Local Interpretable Model-agnostic Explanations) — метод локального интерпретируемого объяснения предсказаний любой модели машинного обучения.
    
    ### Основная идея
    
    Сложную модель невозможно объяснить глобально, но можно приблизить локально — в окрестности конкретного предсказания.
    
    ### Формальное определение
    
    LIME решает следующую оптимизационную задачу:
    
    $$
    \underset{g \in G}{\operatorname{argmin}} \; \mathcal{L}(f, g, \pi_x) + \Omega(g)
    $$
    
    где:
    - $f$ — исходная модель (чёрный ящик)
    - $g$ — интерпретируемая модель-суррогат (линейная регрессия)
    - $\pi_x$ — ядро близости к исходному объекту $x$
    - $\Omega(g)$ — регуляризация, контролирующая сложность модели
    - $G$ — класс интерпретируемых моделей
    
    ### Алгоритм за 3 шага
    
    **Шаг 1. Генерация синтетических объектов**
    
    Из исходного текста создаётся множество пертурбаций. Каждый пертурбированный текст кодируется бинарным вектором $z \in \{0,1\}^d$, где $d$ — размерность словаря, 1 означает присутствие слова.
    
    **Шаг 2. Получение меток от модели**
    
    Для каждого синтетического текста вычисляется предсказание модели $f(z)$. Для задач классификации это вектор вероятностей по классам.
    
    **Шаг 3. Обучение локальной модели**
    
    Взвешенная линейная регрессия минимизирует:
    
    $$
    \sum_{z} \pi_x(z) \cdot (f(z) - g(z))^2 + \alpha \|\mathbf{w}\|_2^2
    $$
    
    где:
    - $\pi_x(z) = \exp\left(-\frac{D(x,z)^2}{\sigma^2}\right)$ — вес, убывающий с расстоянием от исходного объекта
    - $D(x,z)$ — расстояние Хэмминга (доля изменённых признаков)
    - $\sigma$ — ширина ядра (`kernel_width` в реализации)
    - $\alpha$ — коэффициент регуляризации (L2-штраф)
    - $\mathbf{w}$ — вектор коэффициентов линейной модели
    
    ### Интерпретация результата
    
    Коэффициенты линейной модели $w_i$ интерпретируются как **вклад $i$-го слова** в предсказание:
    
    - $w_i > 0$ → слово способствует предсказанному классу
    - $w_i < 0$ → слово препятствует предсказанному классу
    - $|w_i|$ → сила влияния слова
    
    ### Пример работы
    
    ```
    Исходный текст:    "Фильм был отличный!"
           ↓
    Пертурбации:       "____ был отличный!"  → модель: положительный
                       "Фильм ____ отличный!" → модель: положительный  
                       "Фильм был ____"      → модель: нейтральный
                       "____ ____ отличный!" → модель: положительный
           ↓
    Взвешенная регрессия:
                       w_фильм = 0.3
                       w_отличный = 0.8
                       w_был ≈ 0.0
    ```
    
    ### Ссылка
    
    Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD*, 1135-1144. [arXiv:1602.04938](https://arxiv.org/abs/1602.04938)
    """)

with tab_problem:
    st.markdown(r"""    
    ### Эмпирическое наблюдение
    
    При многократном применении Vanilla LIME к одному и тому же объекту $x$ с различными seed-значениями генератора случайных чисел, объяснения $g^{(1)}(x), g^{(2)}(x), \dots, g^{(k)}(x)$ могут существенно различаться.
    
    **Формально:** Для фиксированного $x$ и модели $f$:
    
    $$
    \exists i,j : \quad \text{rank}(g^{(i)}(x)) \neq \text{rank}(g^{(j)}(x))
    $$
    
    где $\text{rank}(g(x))$ — упорядоченный список важности признаков.
    
    ### Причины нестабильности
    
    | Причина | Математическое обоснование |
    |---------|---------------------------|
    | Случайная генерация соседей | $Z \sim \text{Bernoulli}(p)^d$ — пертурбации независимы по признакам |
    | Отсутствие структурных ограничений | $\forall i,j: \text{Cov}(z_i, z_j) = 0$ — нет корреляции между признаками |
    | Высокая дисперсия оценок | $\mathbb{V}[\hat{w}] \propto \frac{1}{n_{\text{samples}}}$ — ограничена размером выборки |
    
    ### Следствия для практического использования
    
    | Проблема | Последствие |
    |----------|-------------|
    | Невоспроизводимость результатов | Невозможность верификации объяснений |
    | Различные топ-признаки при перезапуске | Противоречивые бизнес-решения |
    | Чувствительность к seed | Невозможность применения в production-средах |
    
    ### Количественная оценка
    
    Для количественного измерения нестабильности используется дисперсия важности признаков:
    
    $$
    \text{Stability}(f, x) = 1 - \frac{1}{1 + \frac{1}{k} \sum_{i=1}^k \mathbb{V}[w^{(i)}]}
    $$
    
    где $\mathbb{V}[w^{(i)}]$ — дисперсия коэффициентов $i$-го признака по $k$ запускам.
    
    ### Цель исследования
    
    Разработать модификацию LIME, обеспечивающую:
    
    $$
    \forall i,j: \quad \text{rank}(g^{(i)}(x)) \approx \text{rank}(g^{(j)}(x))
    $$
    
    при сохранении локальной точности аппроксимации.
    """)


def visualize_masking_strategies():
    """Interactive visualization of different masking strategies"""

    # Пример текста
    example_tokens = ["Фильм", "был", "отличный", ",",
                      "очень", "понравилась", "игра", "актёров", "!"]

    st.markdown("#### Выберите стратегию маскирования:")

    strategy = st.radio(
        "Стратегия",
        ["Графовая (распространение на соседей)", "Фразовая (группы токенов)"],
        horizontal=True,
        key="mask_viz_strategy"
    )

    if strategy == "Графовая (распространение на соседей)":
        seed = st.selectbox(
            "Выберите seed-токен (источник маскирования)", example_tokens, key="seed_token")
        prob = st.slider("Вероятность распространения на соседей",
                         0.0, 1.0, 0.3, 0.05, key="prop_prob_viz")

        seed_idx = example_tokens.index(seed)

        # Определяем соседей (window=2)
        neighbors = []
        for i in range(max(0, seed_idx - 2), min(len(example_tokens), seed_idx + 3)):
            if i != seed_idx:
                neighbors.append(i)

        # Случайное распространение
        np.random.seed(42)  # Для воспроизводимости
        masked = [1] * len(example_tokens)
        masked[seed_idx] = 0
        for nb in neighbors:
            if np.random.random() < prob:
                masked[nb] = 0

        # Визуализация
        cols = st.columns(len(example_tokens))
        for i, (token, m) in enumerate(zip(example_tokens, masked)):
            if m == 0:
                cols[i].markdown(
                    f"<div style='background-color:#F44336; padding:10px; border-radius:5px; text-align:center; color:white;'><s>{token}</s><br><span style='font-size:10px'>МАСКИРОВАН</span></div>", unsafe_allow_html=True)
            elif i == seed_idx:
                cols[i].markdown(
                    f"<div style='background-color:#FF9800; padding:10px; border-radius:5px; text-align:center;'><b>{token}</b><br><span style='font-size:10px'>SEED</span></div>", unsafe_allow_html=True)
            else:
                cols[i].markdown(
                    f"<div style='background-color:#4CAF50; padding:10px; border-radius:5px; text-align:center; color:white;'>{token}<br><span style='font-size:10px'>СОХРАНЁН</span></div>", unsafe_allow_html=True)

        st.caption(
            f"Маскирование распространилось на {sum(1 for m in masked if m == 0)} из {len(example_tokens)} токенов")

    else:  # Фразовая стратегия
        # Группы для примера
        groups = [
            {"name": "группа 1 (1 токен)", "indices": [
                0], "tokens": ["Фильм"]},
            {"name": "группа 2 (2 токена)", "indices": [
                1, 2], "tokens": ["был", "отличный"]},
            {"name": "группа 3 (3 токена)", "indices": [1, 2, 3], "tokens": [
                "был", "отличный", ","]},
            {"name": "группа 4 (1 токен)", "indices": [
                4], "tokens": ["очень"]},
            {"name": "группа 5 (3 токена)", "indices": [4, 5, 6], "tokens": [
                "очень", "понравилась", "игра"]},
            {"name": "группа 6 (2 токена)", "indices": [
                5, 6], "tokens": ["понравилась", "игра"]},
            {"name": "группа 7 (1 токен)", "indices": [
                7], "tokens": ["актёров"]},
            {"name": "группа 8 (1 токен)", "indices": [8], "tokens": ["!"]},
        ]

        selected_group = st.selectbox("Выберите фразовую группу для маскирования", [
                                      g["name"] for g in groups], key="phrase_group")
        group = next(g for g in groups if g["name"] == selected_group)

        # Визуализация
        cols = st.columns(len(example_tokens))
        for i, token in enumerate(example_tokens):
            if i in group["indices"]:
                cols[i].markdown(
                    f"<div style='background-color:#F44336; padding:10px; border-radius:5px; text-align:center; color:white;'><s>{token}</s><br><span style='font-size:10px'>МАСКИРОВАН</span></div>", unsafe_allow_html=True)
            else:
                cols[i].markdown(
                    f"<div style='background-color:#4CAF50; padding:10px; border-radius:5px; text-align:center; color:white;'>{token}<br><span style='font-size:10px'>СОХРАНЁН</span></div>", unsafe_allow_html=True)

        st.caption(
            f"Маскирована фраза: {' '.join(group['tokens'])} ({len(group['indices'])} токенов)")


with tab_solution:
    st.markdown("""    
    Предложенный метод расширяет Vanilla LIME четырьмя структурными улучшениями, вдохновлёнными MINDFUL-LIME.
                
    **Ссылка:** Zhu, R., et al. (2025). MINDFUL-LIME: Structure-aware perturbation for explanation stability. [arXiv:2503.20758](https://arxiv.org/abs/2503.20758)
    
    ---
    
    ### 1. Графовая структура токенов
    
    В Vanilla LIME каждый токен маскируется независимо, что приводит к нереалистичным пертурбациям. 
    Наш метод строит граф смежности токенов в скользящем окне:
    
    ```
    Токены:    t1 — t2 — t3 — t4 — t5 — t6
    Рёбра:     t1-t2, t1-t3, t2-t3, t2-t4, t3-t4, t3-t5, ...
    ```
    
    Граф задаёт пространственную структуру: маскирование seed-токена распространяется на соседей 
    с вероятностью `propagation_prob` (по умолчанию 0.3). Это аналог superpixel-графа из MINDFUL-LIME, 
    адаптированный для текстовой модальности.
    
    ---
    
    ### 2. Фразовые группы
    
    Метод автоматически извлекает перекрывающиеся группы смежных токенов длиной от 1 до `phrase_max_len` 
    (по умолчанию 3). Пример для текста "не мог оторваться":
    
    ```
    Группы длины 1: ["не"], ["мог"], ["оторваться"]
    Группы длины 2: ["не мог"], ["мог оторваться"]
    Группы длины 3: ["не мог оторваться"]
    ```
    
    При фразовой пертурбации выбирается случайная группа, и все её токены маскируются одновременно. 
    Это сохраняет лингвистическую целостность — фразы не разрываются.
    
    ---
    
    ### 3. Комбинированная стратегия маскирования
    
    Каждый запуск генерации пертурбаций делит `num_samples` пополам:
    
    - **Первая половина**: графовые маски с распространением на соседей
    - **Вторая половина**: фразовые маски целых групп токенов
    
    Такое комбинирование обеспечивает разнообразие пертурбаций: графовые маски сохраняют 
    локальную связность, фразовые — смысловую целостность.
    
    ---
    
    ### 4. Многозапусковая агрегация по медиане
    
    Метод выполняет `n_runs` независимых объяснений (по умолчанию 3-5). Для каждого токена 
    вычисляется медианная важность:
    
    ```
    Запуск 1: токен_A = 0.8, токен_B = 0.4, токен_C = 0.3
    Запуск 2: токен_A = 0.7, токен_B = 0.5, токен_C = 0.4
    Запуск 3: токен_A = 0.9, токен_B = 0.3, токен_C = 0.3
    
    Медиана:   токен_A = 0.8, токен_B = 0.4, токен_C = 0.3
    ```
    
    Медиана устойчива к выбросам в отличие от среднего. Дополнительно вычисляется дисперсия 
    для каждого токена — метрика стабильности объяснения.
    
    ---
    
    ### Сравнение с Vanilla LIME
    
    | Компонент | Vanilla LIME | Stability-Enhanced LIME |
    |-----------|--------------|------------------------|
    | Маскирование | Независимое (Bernoulli) | Графовое + фразовое |
    | Учёт структуры токенов | Нет | Граф смежности (window=2) |
    | Целостность фраз | Нет | Группы 1-3 токенов |
    | Агрегация запусков | Нет | Медиана по n_runs |
    | Дисперсия важности | Не вычисляется | Доступна в детальном выводе |
    
    ---
    
    ### Управляемые параметры
    
    Параметры доступны в боковой панели:
    
    - `n_runs`: количество запусков для агрегации (1-10). Больше значений = выше стабильность, но линейно растёт время вычислений.
    
    - `mask_rate`: доля маскируемых токенов (0.1-0.7). Более высокие значения создают более агрессивные пертурбации.
    
    - `propagation_prob`: вероятность распространения маски на соседей по графу (0.1-0.7). Низкие значения дают изолированные маски, высокие — связные области.
    """)

    st.markdown("---")
    st.markdown("### Интерактивная демонстрация")
    visualize_masking_strategies()

st.markdown("---")

with st.sidebar:
    st.markdown("### Настройки")

    # Model section
    st.markdown("#### Модель")
    model_path = st.text_input(
        "Путь к модели",
        value="checkpoints/rubert_sentiment.pt",
        help="Путь к файлу обученной модели RuBERT"
    )

    st.markdown("---")

    # LIME parameters
    st.markdown("#### Параметры LIME")

    num_samples = st.slider(
        "Количество пертурбаций",
        min_value=500,
        max_value=5000,
        value=2000,
        step=500,
        help="Больше значений = выше точность, но медленнее"
    )

    num_features = st.slider(
        "Количество топ-признаков",
        min_value=5,
        max_value=15,
        value=10,
        step=1,
        help="Количество наиболее важных слов для отображения"
    )

    st.markdown("---")

    # Enhanced LIME parameters
    st.markdown("#### Улучшенный LIME")

    n_runs = st.slider(
        "Количество запусков для агрегации",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Больше запусков = стабильнее, но медленнее"
    )

    mask_rate = st.slider(
        "Коэффициент маскирования",
        min_value=0.1,
        max_value=0.7,
        value=0.5,
        step=0.05,
        help="Доля токенов, которые маскируются при пертурбации"
    )

    propagation_prob = st.slider(
        "Вероятность распространения",
        min_value=0.1,
        max_value=0.7,
        value=0.5,
        step=0.05,
        help="Вероятность маскирования соседних токенов"
    )

    st.markdown("---")

    # Example texts
    st.markdown("#### Примеры текстов")

    EXAMPLES = {
        "Положительный - Отзыв о книге": "Потрясающая книга, не мог оторваться до последней страницы!",
        "Положительный - Отзыв о фильме": "Фильм был отличный, очень понравилась игра актёров!",
        "Отрицательный - Отзыв о сервисе": "Ужасный сервис, больше никогда не приду сюда.",
        "Нейтральный - Отзыв о товаре": "Обычный продукт, ничего особенного не заметил.",
        "Отрицательный - Отзыв о качестве": "Качество оставляет желать лучшего, разочарован покупкой.",
        "Смешанный - Сбалансированный отзыв": "Цена высокая, но качество хорошее. В целом нормально."
    }

    selected_example = st.selectbox("Выберите пример", list(EXAMPLES.keys()))
    example_text = EXAMPLES[selected_example]

    st.markdown("---")
    st.markdown('<div class="caption">Совет: наведите курсор на подсвеченные слова, чтобы увидеть точные значения</div>', unsafe_allow_html=True)

    @st.cache_resource
    def load_model(model_path_str):
        """Load model with caching for performance"""
        try:
            pipeline = RuBERTPipeline(
                model_name="DeepPavlov/rubert-base-cased",
                model_path=model_path_str,
                num_labels=3,
                max_length=128,
            )
            return pipeline
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            return None


    def get_sentiment_color(label):
        """Return color for sentiment class"""
        colors = {
            "positive": "#4CAF50",
            "negative": "#F44336",
            "neutral": "#9E9E9E"
        }
        return colors.get(label, "#666")


    def get_sentiment_rus(label):
        """Return Russian name for sentiment class"""
        names = {
            "positive": "положительный",
            "negative": "отрицательный",
            "neutral": "нейтральный"
        }
        return names.get(label, label)


    def highlight_words(text, word_scores, tokenizer):
        """Generate HTML with highlighted words - with subword token handling"""

        tokens = tokenizer(text)

        merged_words = []
        current_word = ""

        for token in tokens:
            if token.startswith('##') or token.startswith('#'):
                current_word += token.replace('##', '').replace('#', '')
            elif token.startswith('_') or token == '[UNK]' or token == '[PAD]':
                if current_word:
                    merged_words.append(current_word)
                    current_word = ""
                if token not in ['[UNK]', '[PAD]', '[CLS]', '[SEP]']:
                    merged_words.append(token.replace('_', ''))
            else:
                if current_word:
                    merged_words.append(current_word)
                current_word = token

        if current_word:
            merged_words.append(current_word)

        word_to_score = {}
        for word, score in word_scores:
            word_to_score[word.lower()] = score

        html_parts = []
        original_words = text.split()

        for original_word in original_words:
            clean_word = original_word.strip('.,!?;:()[]{}«»""\'').lower()

            score = word_to_score.get(clean_word, 0)

            if score == 0:
                for key in word_to_score:
                    if key in clean_word or clean_word in key:
                        score = word_to_score[key]
                        break

            if score > 0:
                intensity = min(0.9, abs(score) * 2.5)
                html_parts.append(
                    f'<span class="word-span" style="background-color: rgba(76, 175, 80, {intensity:.2f}); '
                    f'font-weight: 600; color: #1a1a1a;" title="Положительный вклад: {score:+.4f}">{original_word}</span>'
                )
            elif score < 0:
                intensity = min(0.85, abs(score) * 2.5)
                html_parts.append(
                    f'<span class="word-span" style="background-color: rgba(244, 67, 54, {intensity:.2f}); '
                    f'font-weight: 600; color: white;" title="Отрицательный вклад: {score:+.4f}">{original_word}</span>'
                )
            else:
                html_parts.append(
                    f'<span class="word-span" style="background-color: #e0e0e0; color: #555;" '
                    f'title="Нейтральный вклад">{original_word}</span>'
                )

        return " ".join(html_parts)


    def plot_comparison_bar_chart(vanilla_exp, enhanced_exp, num_features):
        """Create side-by-side bar chart comparison"""
        vanilla_dict = {w: s for w, s in vanilla_exp[:num_features]}
        enhanced_dict = {w: s for w, s in enhanced_exp[:num_features]}

        all_words = list(set(vanilla_dict.keys()) | set(enhanced_dict.keys()))
        all_words.sort(key=lambda x: abs(vanilla_dict.get(x, 0)) +
                    abs(enhanced_dict.get(x, 0)), reverse=True)
        all_words = all_words[:10]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Vanilla LIME", "Улучшенный LIME"),
            shared_yaxes=True,
            horizontal_spacing=0.12
        )

        # Vanilla LIME
        vanilla_scores = [vanilla_dict.get(w, 0) for w in all_words]
        vanilla_colors = ['#4CAF50' if s >
                        0 else '#F44336' for s in vanilla_scores]

        fig.add_trace(
            go.Bar(
                x=vanilla_scores,
                y=all_words,
                orientation='h',
                marker_color=vanilla_colors,
                text=[f'{s:.4f}' for s in vanilla_scores],
                textposition='outside',
                textfont=dict(size=10),
                hovertemplate='<b>%{y}</b><br>Важность: %{x:+.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Enhanced LIME
        enhanced_scores = [enhanced_dict.get(w, 0) for w in all_words]
        enhanced_colors = ['#4CAF50' if s >
                        0 else '#F44336' for s in enhanced_scores]

        fig.add_trace(
            go.Bar(
                x=enhanced_scores,
                y=all_words,
                orientation='h',
                marker_color=enhanced_colors,
                text=[f'{s:.4f}' for s in enhanced_scores],
                textposition='outside',
                textfont=dict(size=10),
                hovertemplate='<b>%{y}</b><br>Важность: %{x:+.4f}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=500,
            showlegend=False,
            template="plotly_white",
            font=dict(family="Segoe UI", size=12)
        )

        fig.update_xaxes(title_text="Важность признака", row=1, col=1)
        fig.update_xaxes(title_text="Важность признака", row=1, col=2)

        return fig


    def plot_metrics_radar(vanilla_metrics, enhanced_metrics):
        """Create radar chart for metrics comparison"""
        categories = ['Стабильность', 'Достоверность',
                    'Разреженность', 'Ранговая корреляция']

        vanilla_values = [
            vanilla_metrics.get('stability', 0),
            vanilla_metrics.get('faithfulness', 0),
            vanilla_metrics.get('sparsity', 0),
            vanilla_metrics.get('rank_correlation', 0)
        ]

        enhanced_values = [
            enhanced_metrics.get('stability', 0),
            enhanced_metrics.get('faithfulness', 0),
            enhanced_metrics.get('sparsity', 0),
            enhanced_metrics.get('rank_correlation', 0)
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=vanilla_values,
            theta=categories,
            fill='toself',
            name='Vanilla LIME',
            line_color='#1a1a2e',
            fillcolor='rgba(26, 26, 46, 0.2)'
        ))

        fig.add_trace(go.Scatterpolar(
            r=enhanced_values,
            theta=categories,
            fill='toself',
            name='Улучшенный LIME',
            line_color='#e94560',
            fillcolor='rgba(233, 69, 96, 0.2)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            height=450,
            showlegend=True,
            legend=dict(x=0.85, y=1.1),
            template="plotly_white"
        )

        return fig

with tab_analyze:
    st.caption("Введите русский текст, и модель определит его тональность с объяснением от LIME")

    # Text input area
    col1, col2 = st.columns([4, 1])

    with col1:
        user_text = st.text_area(
            "Введите текст",
            value=example_text,
            height=100,
            help="Введите русский текст для анализа тональности"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Загрузить пример", width='stretch'):
            user_text = example_text
            st.rerun()

    # Analysis button
    analyze_button = st.button("Анализировать текст",
                            type="primary", width='stretch')


    if analyze_button and user_text:

        # Load model
        with st.spinner("Загрузка модели..."):
            pipeline = load_model(model_path)

        if pipeline is None:
            st.error("Не удалось загрузить модель. Проверьте путь к файлу.")
            st.stop()

        # Prediction
        with st.spinner("Анализ тональности..."):
            preds, probs = pipeline.predict([user_text], return_probs=True)
            pred_class = int(preds[0])
            label_names = ["neutral", "positive", "negative"]
            pred_label = label_names[pred_class]
            pred_conf = float(probs[0][pred_class])

        # Results display
        st.markdown("---")
        st.markdown("### Результаты анализа")

        col1, col2, col3 = st.columns(3)

        sentiment_color = get_sentiment_color(pred_label)
        sentiment_rus = get_sentiment_rus(pred_label)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Предсказанная тональность</h3>
                <h1 style="color: {sentiment_color};">{sentiment_rus.upper()}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card metric-success">
                <h3>Уверенность</h3>
                <h1>{pred_conf:.1%}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card metric-warning">
                <h3>Распределение</h3>
                <p>НЕЙТ: {probs[0][0]:.0%} | ПОЗ: {probs[0][1]:.0%} | НЕГ: {probs[0][2]:.0%}</p>
            </div>
            """, unsafe_allow_html=True)

        # Probability chart
        proba_df = pd.DataFrame({
            'Тональность': ['Нейтральная', 'Положительная', 'Отрицательная'],
            'Вероятность': [probs[0][0], probs[0][1], probs[0][2]]
        })

        fig_proba = px.bar(
            proba_df,
            x='Тональность',
            y='Вероятность',
            color='Тональность',
            color_discrete_map={'Нейтральная': '#9E9E9E',
                                'Положительная': '#4CAF50', 'Отрицательная': '#F44336', },
            title="Распределение вероятностей тональности"
        )
        fig_proba.update_layout(
            height=350,
            showlegend=False,
            template="plotly_white",
            font=dict(family="Segoe UI")
        )
        st.plotly_chart(fig_proba, width='stretch')

        # LIME Explanations
        st.markdown("---")
        st.markdown("### LIME объяснения")

        info_text = f"Обработка {num_samples} пертурбаций с {n_runs} запусками агрегации. Примерное время: 60-180 секунд."
        st.markdown(
            f'<div class="info-box">{info_text}</div>', unsafe_allow_html=True)

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prediction function
        def predict_fn(texts):
            return pipeline.predict_proba(texts)

        tokenizer = pipeline.tokenizer.tokenize

        # Vanilla LIME
        status_text.text("Vanilla LIME: генерация объяснений...")
        progress_bar.progress(20)

        start_time = time.time()

        vanilla_explainer = LimeTextExplainer(
            predict_fn=predict_fn,
            num_samples=num_samples,
            num_features=num_features,
            kernel_width=25.0,
        )
        vanilla_exp = vanilla_explainer.explain_instance(
            user_text, class_idx=pred_class)

        progress_bar.progress(45)

        # Enhanced LIME
        status_text.text("Улучшенный LIME: генерация объяснений...")

        enhanced_explainer = StabilityEnhancedLIME(
            predict_fn=predict_fn,
            num_samples=num_samples,
            num_features=num_features,
            kernel_width=25.0,
            phrase_max_len=3,
            adjacency_window=2,
            mask_rate=mask_rate,
            propagation_prob=propagation_prob,
            n_runs=n_runs,
        )
        enhanced_detail = enhanced_explainer.explain_instance_detailed(
            user_text, class_idx=pred_class)
        enhanced_exp = enhanced_detail["aggregated"]

        progress_bar.progress(70)

        # Metrics computation
        status_text.text("Вычисление метрик оценки...")

        vanilla_runs = []
        for i in range(min(n_runs, 5)):
            vanilla_explainer.rng = np.random.default_rng(i * 42)
            vanilla_runs.append(vanilla_explainer.explain_instance(
                user_text, class_idx=pred_class))

        vanilla_metrics = {
            "stability": compute_stability_score(vanilla_runs) if len(vanilla_runs) > 1 else 1.0,
            "faithfulness": compute_faithfulness(user_text, tokenizer, predict_fn, vanilla_exp, top_k=5),
            "sparsity": compute_sparsity(vanilla_exp),
            "rank_correlation": compute_rank_correlation(vanilla_runs) if len(vanilla_runs) > 1 else 1.0,
        }

        enhanced_metrics = {
            "stability": compute_stability_score(enhanced_detail["per_run"]),
            "faithfulness": compute_faithfulness(user_text, tokenizer, predict_fn, enhanced_exp, top_k=5),
            "sparsity": compute_sparsity(enhanced_exp),
            "rank_correlation": compute_rank_correlation(enhanced_detail["per_run"]),
        }

        elapsed_time = time.time() - start_time
        progress_bar.progress(100)
        status_text.text(f"Анализ завершён за {elapsed_time:.1f} секунд")

        # Metrics display
        st.markdown("---")
        st.markdown("### Количественные метрики")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Vanilla LIME**")
            m1 = vanilla_metrics
            st.metric("Стабильность", f"{m1['stability']:.4f}")
            st.metric("Достоверность", f"{m1['faithfulness']:.4f}")
            st.metric("Разреженность", f"{m1['sparsity']:.4f}")
            st.metric("Ранговая корреляция", f"{m1['rank_correlation']:.4f}")

        with col2:
            st.markdown("**Улучшенный LIME**")
            m2 = enhanced_metrics
            delta_stab = m2['stability'] - m1['stability']
            delta_faith = m2['faithfulness'] - m1['faithfulness']
            delta_sparse = m2['sparsity'] - m1['sparsity']
            delta_rank = m2['rank_correlation'] - m1['rank_correlation']

            st.metric("Стабильность",
                    f"{m2['stability']:.4f}", delta=f"{delta_stab:+.4f}")
            st.metric("Достоверность",
                    f"{m2['faithfulness']:.4f}", delta=f"{delta_faith:+.4f}")
            st.metric("Разреженность",
                    f"{m2['sparsity']:.4f}", delta=f"{delta_sparse:+.4f}")
            st.metric("Ранговая корреляция",
                    f"{m2['rank_correlation']:.4f}", delta=f"{delta_rank:+.4f}")
        
        # Radar chart
        st.plotly_chart(plot_metrics_radar(vanilla_metrics,
                        enhanced_metrics), width='stretch')

        # Word importance charts
        st.markdown("---")
        st.markdown("### Сравнение важности слов")
        st.plotly_chart(plot_comparison_bar_chart(
            vanilla_exp, enhanced_exp, num_features), width='stretch')

        # Highlighted text visualization
        st.markdown("---")
        st.markdown("### Визуализация подсветки слов")
        st.caption(
            "Зелёный = положительный вклад | Красный = отрицательный вклад | Серый = нейтральный")

        tab1, tab2 = st.tabs([" Vanilla LIME", " Улучшенный LIME"])

        with tab1:
            highlighted_html = highlight_words(user_text, vanilla_exp, tokenizer)
            st.markdown(
                f'<div style="background: #fafafa; padding: 20px; border-radius: 10px; font-size: 1.15em; line-height: 1.8; border: 1px solid #e0e0e0;">'
                f'{highlighted_html}</div>',
                unsafe_allow_html=True
            )

            with st.expander("Посмотреть полную таблицу важности слов"):
                df_vanilla = pd.DataFrame(
                    vanilla_exp, columns=["Слово", "Важность"])
                st.dataframe(df_vanilla, width='stretch')

        with tab2:
            highlighted_html = highlight_words(user_text, enhanced_exp, tokenizer)
            st.markdown(
                f'<div style="background: #fafafa; padding: 20px; border-radius: 10px; font-size: 1.15em; line-height: 1.8; border: 1px solid #e0e0e0;">'
                f'{highlighted_html}</div>',
                unsafe_allow_html=True
            )

            with st.expander("Посмотреть полную таблицу важности слов"):
                df_enhanced = pd.DataFrame(
                    enhanced_exp, columns=["Слово", "Важность"])
                st.dataframe(df_enhanced, width='stretch')

            if enhanced_detail.get("stability"):
                with st.expander("Посмотреть анализ стабильности (дисперсия по запускам)"):
                    stability_df = pd.DataFrame([
                        {"Слово": w, "Дисперсия": f"{v:.6f}"}
                        for w, v in sorted(enhanced_detail["stability"].items(), key=lambda x: x[1])
                    ])
                    st.dataframe(stability_df, width='stretch')
                    st.caption(
                        "Меньшая дисперсия означает более стабильные объяснения")

        # Download report
        st.markdown("---")
        st.markdown("### Экспорт отчёта")

        # Create HTML report
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Отчёт о сравнении LIME</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #1a1a2e; border-bottom: 2px solid #1a1a2e; padding-bottom: 10px; }}
                h2 {{ color: #16213e; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-top: 30px; }}
                h3 {{ color: #333; margin-top: 20px; }}
                .highlight {{ padding: 20px; background: #f5f5f5; border-radius: 8px; margin: 20px 0; font-size: 1.1em; line-height: 1.8; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #f0f0f0; font-weight: 600; }}
                .metric-box {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 8px; min-width: 150px; }}
                .metric-value {{ font-size: 1.5em; font-weight: bold; color: #1a1a2e; }}
                .positive {{ background-color: rgba(76, 175, 80, 0.3); }}
                .negative {{ background-color: rgba(244, 67, 54, 0.3); }}
                .footer {{ margin-top: 50px; text-align: center; color: #888; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <h1>Отчёт о сравнении LIME</h1>
            <p><strong>Дата генерации:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Модель:</strong> {model_path}</p>
            
            <h2>Входной текст</h2>
            <p><strong>{user_text}</strong></p>
            
            <h2>Результаты предсказания</h2>
            <p><strong>Предсказанная тональность:</strong> {sentiment_rus.upper()} (уверенность: {pred_conf:.2%})</p>
            <p><strong>Распределение вероятностей:</strong> Нейтральная: {probs[0][0]:.2%} | Положительная: {probs[0][1]:.2%} | Негативная: {probs[0][2]:.2%}</p>
            
            <h2>Сравнение метрик</h2>
            <table>
                <tr><th>Метрика</th><th>Vanilla LIME</th><th>Улучшенный LIME</th><th>Разница</th></tr>
                <tr><td>Стабильность</dt><td>{vanilla_metrics['stability']:.4f}</td>
                <td>{enhanced_metrics['stability']:.4f}</dt><td>{enhanced_metrics['stability']-vanilla_metrics['stability']:+.4f}</dt></tr>
                <tr><td>Достоверность</dt><td>{vanilla_metrics['faithfulness']:.4f}</dt>
                <td>{enhanced_metrics['faithfulness']:.4f}</dt><td>{enhanced_metrics['faithfulness']-vanilla_metrics['faithfulness']:+.4f}</dt></tr>
                <tr><td>Разреженность</dt><td>{vanilla_metrics['sparsity']:.4f}</dt>
                <td>{enhanced_metrics['sparsity']:.4f}</dt><td>{enhanced_metrics['sparsity']-vanilla_metrics['sparsity']:+.4f}</dt></tr>
                <tr><td>Ранговая корреляция</dt><td>{vanilla_metrics['rank_correlation']:.4f}</dt>
                <td>{enhanced_metrics['rank_correlation']:.4f}</dt><td>{enhanced_metrics['rank_correlation']-vanilla_metrics['rank_correlation']:+.4f}</dt></tr>
            </table>
            
            <h2>Объяснение Vanilla LIME</h2>
            <div class="highlight">{highlight_words(user_text, vanilla_exp, tokenizer)}</div>
            
            <h2>Объяснение улучшенного LIME</h2>
            <div class="highlight">{highlight_words(user_text, enhanced_exp, tokenizer)}</div>
            
            <h2>Таблицы важности слов</h2>
            <h3>Vanilla LIME - Топ 15 слов</h3>
            <table><th>Слово</th><th>Важность</th></tr>
            {"".join(f"<tr><td>{w}</dt><td>{s:+.4f}</dt></tr>" for w, s in vanilla_exp[:15])}
            </table>
            
            <h3>Улучшенный LIME - Топ 15 слов</h3>
            <table><th>Слово</th><th>Важность</th></tr>
            {"".join(f"<tr><td>{w}</dt><td>{s:+.4f}</dt></tr>" for w, s in enhanced_exp[:15])}
            </table>
            
            <div class="footer">
                <p>Сгенерировано инструментом сравнения LIME | Stability-Enhanced LIME для анализа тональности русских текстов</p>
            </div>
        </body>
        </html>
        """

        st.download_button(
            label="Скачать HTML отчёт",
            data=report_html,
            file_name=f"lime_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            width='stretch'
        )

        # Completion message
        st.success("Анализ завершён")

    elif analyze_button and not user_text:
        st.warning("Пожалуйста, введите текст для анализа")

st.markdown(
    """
    <div style="text-align: center; color: #888; padding: 20px; font-size: 0.8rem;">
        Stability-Enhanced LIME для анализа тональности русских текстов<br>
        Основано на трансформерной модели RuBERT и структурированных пертурбациях MINDFUL-LIME
    </div>
    """,
    unsafe_allow_html=True
)
