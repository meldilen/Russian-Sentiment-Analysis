"""
Evaluation metrics for LIME explanations.
- Stability score: variance across runs
- Faithfulness: confidence drop after removing top-k words
"""

import numpy as np
from typing import List, Tuple, Callable


def compute_stability_score(
    explanations: List[List[Tuple[str, float]]],
) -> float:
    """
    Compute explanation stability as inverse of variance across runs.
    
    For each word, compute variance of its importance score across runs.
    Lower variance = more stable.
    
    Returns:
        Stability score (0-1, higher is better). 
        Based on 1 / (1 + mean_variance) so that 0 variance -> 1.0
    """
    if len(explanations) < 2:
        return 1.0
    
    # Build word -> list of scores
    word_scores = {}
    for exp in explanations:
        for word, score in exp:
            if word not in word_scores:
                word_scores[word] = []
            word_scores[word].append(score)
    
    variances = []
    for word, scores in word_scores.items():
        if len(scores) >= 2:
            variances.append(np.var(scores))
    
    if not variances:
        return 1.0
    
    mean_var = np.mean(variances)
    return 1.0 / (1.0 + mean_var)


def compute_faithfulness(
    text: str,
    tokenizer: Callable[[str], List[str]],
    predict_fn: Callable[[List[str]], np.ndarray],
    explanation: List[Tuple[str, float]],
    top_k: int = 5,
) -> float:
    """
    Faithfulness: measure of how much removing top-k important words
    drops the model's confidence in the predicted class.
    
    Higher drop = more faithful (explanation captures important words).
    
    Returns:
        Confidence drop (0-1, higher means more faithful)
    """
    tokens = tokenizer(text)
    if not tokens or not explanation:
        return 0.0
    
    # Get top-k words by absolute importance
    top_words = set(w for w, _ in sorted(explanation, key=lambda x: abs(x[1]), reverse=True)[:top_k])
    
    # Original prediction
    orig_probs = predict_fn([text])[0]
    pred_class = int(np.argmax(orig_probs))
    orig_conf = orig_probs[pred_class]
    
    # Create perturbed text (remove top-k words)
    perturbed_tokens = [t for t in tokens if t not in top_words]
    perturbed_text = " ".join(perturbed_tokens).strip() or " "
    
    # Prediction on perturbed
    new_probs = predict_fn([perturbed_text])[0]
    new_conf = new_probs[pred_class]
    
    # Confidence drop
    drop = orig_conf - new_conf
    return max(0.0, min(1.0, drop))
