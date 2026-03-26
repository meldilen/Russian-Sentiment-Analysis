"""
Evaluation metrics for LIME explanations.
- Stability score: variance across runs
- Faithfulness: confidence drop after removing top-k words
- Sparsity: gini coefficient of explanation weights
- Rank correlation: consistency of feature rankings across runs
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Optional
from collections import defaultdict


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

    word_scores: Dict[str, List[float]] = defaultdict(list)
    for exp in explanations:
        for word, score in exp:
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
    Faithfulness: confidence drop when removing top-k important words.
    Higher drop = more faithful explanation.

    Returns:
        Confidence drop (0-1, higher means more faithful)
    """
    tokens = tokenizer(text)
    if not tokens or not explanation:
        return 0.0

    top_words = set(
        w for w, _ in sorted(explanation, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    )

    orig_probs = predict_fn([text])[0]
    pred_class = int(np.argmax(orig_probs))
    orig_conf = orig_probs[pred_class]

    perturbed_tokens = [t for t in tokens if t not in top_words]
    perturbed_text = " ".join(perturbed_tokens).strip() or " "

    new_probs = predict_fn([perturbed_text])[0]
    new_conf = new_probs[pred_class]

    drop = orig_conf - new_conf
    return max(0.0, min(1.0, drop))


def compute_sparsity(
    explanation: List[Tuple[str, float]],
) -> float:
    """
    Compute explanation sparsity using the Gini coefficient of absolute weights.

    A Gini coefficient close to 1 means a very sparse explanation (few features
    dominate). Close to 0 means weights are uniformly distributed.

    Returns:
        Gini coefficient (0-1, higher = sparser)
    """
    if not explanation:
        return 0.0

    scores = np.array([abs(s) for _, s in explanation])
    if scores.sum() == 0:
        return 0.0

    scores = np.sort(scores)
    n = len(scores)
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * scores).sum() / (n * scores.sum()))


def compute_rank_correlation(
    explanations: List[List[Tuple[str, float]]],
) -> float:
    """
    Compute average pairwise Spearman rank correlation of word rankings
    across multiple explanation runs.

    High correlation = consistent feature ranking = more stable.

    Returns:
        Mean Spearman rho (-1 to 1, higher is more consistent)
    """
    if len(explanations) < 2:
        return 1.0

    all_words = set()
    for exp in explanations:
        all_words.update(w for w, _ in exp)

    if not all_words:
        return 1.0

    word_list = sorted(all_words)
    word_to_idx = {w: i for i, w in enumerate(word_list)}
    n_words = len(word_list)

    rank_matrices = []
    for exp in explanations:
        score_vec = np.zeros(n_words)
        for w, s in exp:
            score_vec[word_to_idx[w]] = abs(s)
        ranks = np.zeros(n_words)
        order = np.argsort(-score_vec)
        for rank, idx in enumerate(order):
            ranks[idx] = rank
        rank_matrices.append(ranks)

    correlations = []
    for i in range(len(rank_matrices)):
        for j in range(i + 1, len(rank_matrices)):
            r1, r2 = rank_matrices[i], rank_matrices[j]
            d = r1 - r2
            rho = 1 - (6 * np.sum(d ** 2)) / (n_words * (n_words ** 2 - 1))
            correlations.append(rho)

    return float(np.mean(correlations)) if correlations else 1.0


def compute_incremental_faithfulness(
    text: str,
    tokenizer: Callable[[str], List[str]],
    predict_fn: Callable[[List[str]], np.ndarray],
    explanation: List[Tuple[str, float]],
    max_k: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """
    Compute faithfulness at each removal step k=1..max_k.

    Returns list of (k, confidence_drop) for plotting deletion curves.
    """
    tokens = tokenizer(text)
    if not tokens or not explanation:
        return []

    sorted_exp = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)
    if max_k is None:
        max_k = len(sorted_exp)

    orig_probs = predict_fn([text])[0]
    pred_class = int(np.argmax(orig_probs))
    orig_conf = orig_probs[pred_class]

    results = []
    removed = set()
    for k in range(1, max_k + 1):
        if k - 1 < len(sorted_exp):
            removed.add(sorted_exp[k - 1][0])

        remaining = [t for t in tokens if t not in removed]
        perturbed = " ".join(remaining).strip() or " "
        new_probs = predict_fn([perturbed])[0]
        drop = orig_conf - new_probs[pred_class]
        results.append((k, max(0.0, min(1.0, drop))))

    return results


def compute_all_metrics(
    text: str,
    tokenizer: Callable[[str], List[str]],
    predict_fn: Callable[[List[str]], np.ndarray],
    explanations: List[List[Tuple[str, float]]],
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics at once.

    Args:
        text: Input text
        tokenizer: Tokenizer function
        predict_fn: Prediction function
        explanations: List of explanations from multiple runs
        top_k: Number of top features for faithfulness

    Returns:
        Dict with stability, faithfulness, sparsity, rank_correlation
    """
    stability = compute_stability_score(explanations)
    faithfulness = compute_faithfulness(
        text, tokenizer, predict_fn, explanations[0], top_k
    ) if explanations else 0.0
    sparsity = compute_sparsity(explanations[0]) if explanations else 0.0
    rank_corr = compute_rank_correlation(explanations)

    return {
        "stability": stability,
        "faithfulness": faithfulness,
        "sparsity": sparsity,
        "rank_correlation": rank_corr,
    }
