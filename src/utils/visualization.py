"""
Visualization utilities for LIME explanations.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def plot_word_importance(
    word_scores: List[Tuple[str, float]],
    title: str = "Word Importance (LIME)",
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
):
    """
    Plot horizontal bar chart of word importance scores.
    
    Args:
        word_scores: List of (word, score) tuples
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if not word_scores:
        return
    
    words = [w for w, _ in word_scores]
    scores = [s for _, s in word_scores]
    colors = ["green" if s > 0 else "red" for s in scores]
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(words, scores, color=colors, alpha=0.7)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_explanation_comparison(
    explanations: List[List[Tuple[str, float]]],
    labels: List[str],
    title: str = "Explanation Comparison",
    save_path: str = None,
):
    """
    Compare word importance across multiple runs (stability analysis).
    """
    # Flatten and aggregate
    all_words = set()
    for exp in explanations:
        all_words.update(w for w, _ in exp)
    
    # Could add variance visualization here for stability-enhanced LIME
    fig, axes = plt.subplots(1, len(explanations), figsize=(5 * len(explanations), 5))
    if len(explanations) == 1:
        axes = [axes]
    
    for ax, exp, label in zip(axes, explanations, labels):
        if exp:
            words = [w for w, _ in exp]
            scores = [s for _, s in exp]
            colors = ["green" if s > 0 else "red" for s in scores]
            ax.barh(words, scores, color=colors, alpha=0.7)
        ax.set_title(label)
        ax.axvline(x=0, color="black", linewidth=0.5)
    
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
