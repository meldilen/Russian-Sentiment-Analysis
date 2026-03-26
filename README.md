# Stability-Enhanced LIME for Russian Sentiment Analysis

An XAI course project implementing a stability-enhanced LIME framework inspired by **MINDFUL-LIME** to explain predictions of a **RuBERT**-based transformer model for Russian-language sentiment classification.

## Project Overview

| Component | Details |
|---|---|
| **Domain** | Russian Sentiment Analysis (3-class: negative / neutral / positive) |
| **Model** | RuBERT (`DeepPavlov/rubert-base-cased`) fine-tuned for classification |
| **Baseline XAI** | Vanilla LIME (Ribeiro et al., 2016) |
| **Enhanced XAI** | Stability-Enhanced LIME with structure-aware perturbations |
| **Dataset** | [Russian Sentiment Dataset (Kaggle)](https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset) |

## What Makes This Different

Standard LIME generates explanations by **randomly** removing individual words, which leads to **unstable** explanations (different runs produce different results) and **incoherent** perturbations.

Our **Stability-Enhanced LIME** adapts MINDFUL-LIME's principles to text:

- **Phrase-level perturbations** — adjacent words are masked together, preserving linguistic structure
- **Graph-based neighborhood sampling** — a token adjacency graph guides which tokens get masked together (analogous to superpixel graphs in MINDFUL-LIME)
- **Controlled masking** — seed tokens propagate masking to neighbors with tunable probability
- **Multi-run aggregation** — multiple explanation passes are averaged for stability

## Project Structure

```
├── config.yaml                  # All configuration (model, LIME, evaluation)
├── requirements.txt             # Python dependencies
├── train.py                     # RuBERT fine-tuning script
├── explain.py                   # Explain single predictions (vanilla / enhanced / both)
├── evaluate.py                  # Full evaluation pipeline with metrics & visualizations
├── case_studies.py              # Generate 5 qualitative case study reports
├── scripts/
│   └── create_sample_data.py    # Generate sample data for testing
└── src/
    ├── data/
    │   └── preprocessing.py     # Dataset loading & preprocessing
    ├── model/
    │   └── rubert_classifier.py # RuBERT classifier + inference pipeline
    ├── xai/
    │   ├── lime_text.py         # Vanilla LIME implementation
    │   └── stability_lime.py    # Stability-Enhanced LIME (MINDFUL-LIME inspired)
    └── utils/
        ├── metrics.py           # Stability, faithfulness, sparsity, rank correlation
        └── visualization.py     # Bar charts, heatmaps, deletion curves, comparisons
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

**Option A: Sample data** (for quick testing)

```bash
python scripts/create_sample_data.py
```

**Option B: Kaggle dataset** (for real results)

Download the [Russian Sentiment Dataset](https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset) and place `train.csv` and `test.csv` in `data/`. Expected columns: `text`, `label`, `src`.

### 3. Train the model

```bash
python train.py --config config.yaml
```

This fine-tunes RuBERT for 3-class sentiment classification and saves the checkpoint to `checkpoints/rubert_sentiment.pt`.

### 4. Explain a prediction

```bash
# Both methods side by side (default)
python explain.py --text "Фильм был отличный, очень понравилась игра актёров!"

# Vanilla LIME only
python explain.py --text "Ужасный сервис." --method vanilla

# Enhanced LIME only
python explain.py --text "Обычный продукт." --method enhanced

# Save visualization
python explain.py --text "Супер!" --method both --save-plot output.png

# Save HTML report
python explain.py --text "Супер!" --method both --save-html output.html
```

### 5. Run full evaluation

```bash
python evaluate.py --config config.yaml --output-dir results/
```

This runs both methods on multiple texts and produces:
- `results/evaluation_results.json` — quantitative metrics
- `results/metrics_comparison.png` — grouped bar chart
- `results/case_*_comparison.png` — per-text side-by-side
- `results/case_*_stability.png` — stability heatmaps
- `results/case_*_deletion.png` — faithfulness deletion curves

### 6. Generate case studies

```bash
python case_studies.py --config config.yaml --output-dir results/case_studies
```

Produces 5 detailed case studies with:
- HTML reports with highlighted words (`case_N/report.html`)
- Comparison plots and stability heatmaps
- Index page (`index.html`) linking all cases
- JSON summary (`summary.json`)

## How It Works

### Vanilla LIME (Baseline)

The LIME objective function:

```
argmin_g  L(f, g, π_x) + Ω(g)
```

Where `f` is RuBERT, `g` is a Ridge regression surrogate, `π_x` is an exponential locality kernel, and `Ω(g)` is L2 regularization.

Steps:
1. Tokenize input text into words
2. Generate perturbations by **randomly** masking individual words
3. Get model predictions on perturbed texts
4. Compute locality weights using exponential kernel on Hamming distance
5. Fit weighted Ridge regression
6. Extract word importance coefficients

### Stability-Enhanced LIME

Extends vanilla LIME with structure-aware perturbations:

1. **Build adjacency graph** — connect tokens within a sliding window
2. **Extract phrase groups** — identify contiguous spans of 1–3 tokens
3. **Graph-based masking** — select seed tokens, propagate masking to graph neighbors
4. **Phrase-level masking** — mask entire phrase groups for coherent perturbations
5. **Combine both** — 50% graph-based + 50% phrase-level perturbations
6. **Multi-run aggregation** — repeat N times, average per-word importance

This produces **more stable** (lower variance across runs) and **more faithful** (better captures what the model relies on) explanations.

## Evaluation Metrics

| Metric | Description | Better |
|---|---|---|
| **Stability** | `1 / (1 + mean_variance)` across runs | Higher = more stable |
| **Faithfulness** | Confidence drop when removing top-k words | Higher = more faithful |
| **Sparsity** | Gini coefficient of absolute importance weights | Higher = sparser |
| **Rank Correlation** | Spearman ρ of feature rankings across runs | Higher = more consistent |

## Configuration

All settings are in `config.yaml`:

```yaml
# Model
model:
  name: "DeepPavlov/rubert-base-cased"
  num_labels: 3
  max_length: 256

# Vanilla LIME
lime:
  num_samples: 5000
  num_features: 10
  kernel_width: 25.0

# Stability-Enhanced LIME
enhanced_lime:
  num_samples: 5000
  num_features: 10
  phrase_max_len: 3       # max tokens per phrase group
  adjacency_window: 2     # token adjacency window size
  mask_rate: 0.4          # fraction of tokens masked per perturbation
  n_runs: 5               # aggregation runs for stability
```

Key parameters to tune:
- `mask_rate`: Higher = more aggressive perturbations. 0.3–0.5 recommended.
- `n_runs`: More runs = more stable but slower. 3–10 recommended.
- `phrase_max_len`: Longer phrases = coarser explanations. 2–4 recommended.
- `adjacency_window`: Wider window = more connected graph. 1–3 recommended.

## References

- Ribeiro et al. (2016) — [LIME: Local Interpretable Model-agnostic Explanations](https://arxiv.org/abs/1602.04938)
- MINDFUL-LIME — Structure-aware perturbations for explanation stability
- [ModernBERT-XAI](https://arxiv.org/abs/2503.20758) — Transformer interpretability
- [Russian Sentiment Dataset](https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset)

## Authors

- Vladimir Rublev (v.rublev@innopolis.university)
- Dziyana Melnikava (dz.melnikava@innopolis.university)
- Anton Korotkov (a.korotkov@innopolis.university)
