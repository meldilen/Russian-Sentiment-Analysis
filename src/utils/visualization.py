"""
Visualization utilities for LIME explanations.
Supports vanilla LIME, stability-enhanced LIME, and comparison views.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


def plot_word_importance(
    word_scores: List[Tuple[str, float]],
    title: str = "Word Importance (LIME)",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot horizontal bar chart of word importance scores."""
    if not word_scores:
        return

    words = [w for w, _ in word_scores]
    scores = [s for _, s in word_scores]
    colors = ["#4CAF50" if s > 0 else "#F44336" for s in scores]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(words, scores, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_explanation_comparison(
    explanations: List[List[Tuple[str, float]]],
    labels: List[str],
    title: str = "Explanation Comparison",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Compare word importance across multiple explanation methods or runs."""
    n = len(explanations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, exp, label in zip(axes, explanations, labels):
        if exp:
            words = [w for w, _ in exp]
            scores = [s for _, s in exp]
            colors = ["#4CAF50" if s > 0 else "#F44336" for s in scores]
            ax.barh(words, scores, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_stability_heatmap(
    per_run_explanations: List[List[Tuple[str, float]]],
    title: str = "Explanation Stability Across Runs",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Heatmap showing word importance scores across multiple runs.
    Consistent colors across rows indicate stable explanations.
    """
    all_words = []
    seen = set()
    for exp in per_run_explanations:
        for w, _ in exp:
            if w not in seen:
                all_words.append(w)
                seen.add(w)

    if not all_words:
        return

    n_runs = len(per_run_explanations)
    n_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    matrix = np.zeros((n_runs, n_words))
    for r, exp in enumerate(per_run_explanations):
        for w, s in exp:
            matrix[r, word_to_idx[w]] = s

    fig, ax = plt.subplots(figsize=(max(8, n_words * 0.8), max(4, n_runs * 0.6)))
    vmax = np.abs(matrix).max() or 1
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_words))
    ax.set_xticklabels(all_words, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_runs))
    ax.set_yticklabels([f"Run {i+1}" for i in range(n_runs)], fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Importance Score")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Vanilla LIME vs Stability-Enhanced LIME",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Grouped bar chart comparing metric values between methods.
    metrics: {"Vanilla LIME": {"stability": 0.8, ...}, "Enhanced LIME": {...}}
    """
    methods = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    n_methods = len(methods)
    n_metrics = len(metric_names)

    x = np.arange(n_metrics)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for i, method in enumerate(methods):
        values = [metrics[method].get(m, 0) for m in metric_names]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ").title() for n in metric_names], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_deletion_curve(
    curves: Dict[str, List[Tuple[int, float]]],
    title: str = "Faithfulness Deletion Curve",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot deletion curves for multiple methods.
    curves: {"Vanilla LIME": [(1, 0.1), (2, 0.3), ...], "Enhanced": [...]}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, (method, data) in enumerate(curves.items()):
        ks = [k for k, _ in data]
        drops = [d for _, d in data]
        ax.plot(ks, drops, marker="o", linewidth=2, markersize=5,
                color=colors[i % len(colors)], label=method)

    ax.set_xlabel("# Words Removed (top-k)", fontsize=12)
    ax.set_ylabel("Confidence Drop", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
