# XAI Project: Stability-Enhanced LIME for Russian Sentiment Analysis

Baseline implementation for the XAI course project. Implements vanilla LIME for explaining RuBERT-based sentiment predictions on Russian text.

## Project Overview

- **Domain**: Russian sentiment analysis (3-class: negative / neutral / positive)
- **Model**: RuBERT (DeepPavlov/rubert-base-cased) fine-tuned for classification
- **XAI**: Vanilla LIME (Ribeiro et al., 2016) for word-level explanations

## Structure

```
xAI/
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
├── train.py              # RuBERT fine-tuning
├── explain.py            # LIME explanation script
├── data/                 # Dataset (create via scripts/create_sample_data.py)
├── checkpoints/          # Saved models
├── scripts/
│   └── create_sample_data.py  # Generate sample data
└── src/
    ├── data/             # Dataset loading & preprocessing
    ├── model/            # RuBERT classifier
    ├── xai/              # LIME implementation
    └── utils/            # Visualization & metrics
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create sample data (or use Kaggle dataset)

For testing without the full Kaggle dataset:

```bash
python scripts/create_sample_data.py
```

For production, download the [Russian Sentiment Analysis Dataset](https://www.kaggle.com/datasets) and place `train.csv` and `test.csv` in `data/` with columns: `text`, `label`, `src`.

### 3. Train the model

```bash
python train.py --config config.yaml
```

### 4. Explain predictions

```bash
python explain.py --text "Фильм был отличный, очень понравилась игра актёров!"
```

## LIME Implementation

The vanilla LIME (`src/xai/lime_text.py`) implements:

- **Perturbation**: Random word removal (masking)
- **Neighborhood**: Configurable number of perturbed samples
- **Surrogate**: Weighted Ridge regression with locality kernel
- **Output**: Word importance scores for the predicted class

## Evaluation Metrics

- **Stability score**: Variance of explanations across multiple runs
- **Faithfulness**: Confidence drop when removing top-k important words

## Next Steps (Stability-Enhanced LIME)

The proposal outlines a MINDFUL-LIME inspired extension:

- Phrase-level perturbations (adjacent word grouping)
- Local neighborhood constraints for contextual coherence
- Structure-aware masking respecting linguistic relationships

## References

- Ribeiro et al. (2016) — LIME: Local Interpretable Model-agnostic Explanations
- MINDFUL-LIME — Structure-aware perturbations
- ModernBERT-XAI — Transformer interpretability
