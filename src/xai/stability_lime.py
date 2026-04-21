"""
Stability-Enhanced LIME inspired by MINDFUL-LIME (adapted to text domain).

Instead of purely random word removal (vanilla LIME), this module introduces
structure-aware perturbations that respect linguistic relationships:

1. Phrase-level perturbations: adjacent word groups are masked together
2. Local neighborhood constraints: contextual coherence is preserved
3. Structure-aware masking: perturbation respects token adjacency graphs

Reference:
- MINDFUL-LIME: structure-aware perturbation for improved stability
- Ribeiro et al. (2016): original LIME
"""

import numpy as np
from typing import Callable, List, Optional, Tuple, Dict
from collections import defaultdict

from .lime_text import _tokenize_russian


def _build_adjacency_graph(tokens: List[str], window_size: int = 2) -> Dict[int, List[int]]:
    """
    Build a token adjacency graph where edges connect tokens within
    a sliding window. This is the text-domain analogue of the superpixel
    graph in MINDFUL-LIME.
    """
    graph = defaultdict(list)
    n = len(tokens)
    for i in range(n):
        for j in range(max(0, i - window_size), min(n, i + window_size + 1)):
            if i != j:
                graph[i].append(j)
    return dict(graph)


def _extract_phrase_groups(tokens: List[str], max_phrase_len: int = 3) -> List[List[int]]:
    """
    Group tokens into overlapping phrase-level units (adjacent word groups).
    Each group is a contiguous span of 1..max_phrase_len tokens.
    Returns list of index groups sorted by position.
    """
    n = len(tokens)
    groups = []
    for start in range(n):
        for length in range(1, min(max_phrase_len + 1, n - start + 1)):
            groups.append(list(range(start, start + length)))
    return groups


class StabilityEnhancedLIME:
    """
    Stability-Enhanced LIME for text explanations.

    Adapts MINDFUL-LIME's structural perturbation principles to the text
    domain using:
    - Phrase-level perturbation groups (adjacent tokens masked together)
    - Graph-based neighborhood sampling (respecting adjacency structure)
    - Controlled masking rate for better local fidelity
    - Multiple-run aggregation for stable explanations
    """

    def __init__(
        self,
        predict_fn: Callable[[List[str]], np.ndarray],
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        num_samples: int = 5000,
        num_features: int = 10,
        kernel_width: float = 25.0,
        phrase_max_len: int = 3,
        adjacency_window: int = 2,
        mask_rate: float = 0.4,
        propagation_prob: float = 0.3,
        n_runs: int = 5,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            predict_fn: Function mapping list of strings to probability array (n, n_classes)
            tokenizer: Text tokenizer function
            num_samples: Perturbed samples per explanation run
            num_features: Max features in output explanation
            kernel_width: Exponential kernel width for locality weighting
            phrase_max_len: Max contiguous token span for phrase-level groups
            adjacency_window: Window size for the adjacency graph
            mask_rate: Target fraction of tokens to mask per perturbation
            propagation_prob: Probability of masking a neighbor (lower = more conservative)
            n_runs: Number of aggregated runs for stability
            random_state: Seed for reproducibility
        """
        self.predict_fn = predict_fn
        self.tokenizer = tokenizer or _tokenize_russian
        self.num_samples = num_samples
        self.num_features = num_features
        self.kernel_width = kernel_width
        self.phrase_max_len = phrase_max_len
        self.adjacency_window = adjacency_window
        self.mask_rate = mask_rate
        self.propagation_prob = propagation_prob
        self.n_runs = n_runs
        self.rng = np.random.default_rng(random_state)

    def _structure_aware_masks(
        self,
        tokens: List[str],
        graph: Dict[int, List[int]],
    ) -> np.ndarray:
        """
        Generate perturbation masks using graph-guided sampling.

        Instead of flipping each token independently (vanilla LIME),
        we select seed tokens and propagate masking to neighbors
        with controlled probability, producing coherent perturbations.

        Propagation_prob controls neighbor masking separately from mask_rate.
        """
        n = len(tokens)
        masks = np.ones((self.num_samples, n), dtype=np.int32)
        masks[0] = 1  # first sample is always unperturbed

        for i in range(1, self.num_samples):
            n_seeds = max(1, int(n * self.mask_rate))
            seeds = self.rng.choice(n, size=min(n_seeds, n), replace=False)

            for seed in seeds:
                masks[i, seed] = 0
                neighbors = graph.get(seed, [])
                for nb in neighbors:
                    if self.rng.random() < self.propagation_prob:
                        masks[i, nb] = 0

        return masks
    
    def _phrase_level_masks(
        self,
        tokens: List[str],
        groups: List[List[int]],
    ) -> np.ndarray:
        """
        Generate perturbation masks at the phrase level.

        Randomly select phrase groups to mask entirely, maintaining
        linguistic coherence of the perturbation.
        """
        n = len(tokens)
        masks = np.ones((self.num_samples, n), dtype=np.int32)
        masks[0] = 1

        n_groups = len(groups)
        for i in range(1, self.num_samples):
            n_mask = max(1, int(n_groups * self.mask_rate))
            chosen = self.rng.choice(n_groups, size=min(n_mask, n_groups), replace=False)
            for g_idx in chosen:
                for tok_idx in groups[g_idx]:
                    masks[i, tok_idx] = 0

        return masks

    def _perturb_text(self, tokens: List[str], mask: np.ndarray) -> str:
        kept = [tok for tok, m in zip(tokens, mask) if m == 1]
        return " ".join(kept).strip() or " "

    def _kernel(self, distances: np.ndarray) -> np.ndarray:
        return np.sqrt(np.exp(-(distances ** 2) / (self.kernel_width ** 2)))

    def _single_run(
        self,
        text: str,
        tokens: List[str],
        class_idx: int,
        graph: Dict[int, List[int]],
        groups: List[List[int]],
    ) -> List[Tuple[str, float]]:
        """Run one explanation pass, combining graph-based and phrase-level perturbations."""
        n = len(tokens)
        half = self.num_samples // 2

        graph_masks = self._structure_aware_masks(tokens, graph)[:half]
        phrase_masks = self._phrase_level_masks(tokens, groups)[: self.num_samples - half]
        masks = np.vstack([graph_masks, phrase_masks])

        masks[0] = 1

        perturbed = [self._perturb_text(tokens, masks[i]) for i in range(len(masks))]
        labels = self.predict_fn(perturbed)

        if labels.ndim == 2:
            y = labels[:, class_idx]
        else:
            y = labels

        distances = np.sum(masks != 1, axis=1).astype(float)
        weights = self._kernel(distances)

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(masks, y, sample_weight=weights)

        scores = [(tokens[i], float(model.coef_[i])) for i in range(n)]
        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return scores[:self.num_features]

    def explain_instance(
        self,
        text: str,
        class_idx: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Explain a prediction with stability-enhanced LIME.

        Aggregates multiple runs and returns the median importance per word.

        Returns:
            List of (word, median_importance) sorted by absolute importance.
        """
        tokens = self.tokenizer(text)
        if not tokens:
            return []

        orig_probs = self.predict_fn([text])[0]
        if class_idx is None:
            class_idx = int(np.argmax(orig_probs))

        graph = _build_adjacency_graph(tokens, self.adjacency_window)
        groups = _extract_phrase_groups(tokens, self.phrase_max_len)
        all_runs: List[List[Tuple[str, float]]] = []
        for _ in range(self.n_runs):
            run = self._single_run(text, tokens, class_idx, graph, groups)
            all_runs.append(run)
        aggregated = self._aggregate_runs(all_runs, tokens)
        return aggregated

    def explain_instance_detailed(
        self,
        text: str,
        class_idx: Optional[int] = None,
    ) -> Dict:
        """
        Return detailed explanation with per-run data for analysis.

        Returns dict with:
            - aggregated: final (word, score) list
            - per_run: list of per-run explanations
            - tokens: input tokens
            - class_idx: explained class
            - stability: per-word variance dict
        """
        tokens = self.tokenizer(text)
        if not tokens:
            return {"aggregated": [], "per_run": [], "tokens": tokens, "class_idx": None, "stability": {}}

        orig_probs = self.predict_fn([text])[0]
        if class_idx is None:
            class_idx = int(np.argmax(orig_probs))

        graph = _build_adjacency_graph(tokens, self.adjacency_window)
        groups = _extract_phrase_groups(tokens, self.phrase_max_len)

        all_runs = []
        for _ in range(self.n_runs):
            run = self._single_run(text, tokens, class_idx, graph, groups)
            all_runs.append(run)
        aggregated = self._aggregate_runs(all_runs, tokens)

        word_scores_map: Dict[str, List[float]] = defaultdict(list)
        for run in all_runs:
            for w, s in run:
                word_scores_map[w].append(s)

        stability = {}
        for w, scores in word_scores_map.items():
            stability[w] = float(np.var(scores)) if len(scores) > 1 else 0.0

        return {
            "aggregated": aggregated,
            "per_run": all_runs,
            "tokens": tokens,
            "class_idx": class_idx,
            "stability": stability,
        }

    def _aggregate_runs(
        self,
        all_runs: List[List[Tuple[str, float]]],
        tokens: List[str],
    ) -> List[Tuple[str, float]]:
        """Aggregate multiple runs by computing median importance per word."""
        word_scores: Dict[str, List[float]] = defaultdict(list)
        for run in all_runs:
            for w, s in run:
                word_scores[w].append(s)

        aggregated = []
        for word in tokens:
            if word in word_scores:
                median_score = float(np.median(word_scores[word]))
                aggregated.append((word, median_score))

        seen = set()
        deduped = []
        for w, s in aggregated:
            if w not in seen:
                deduped.append((w, s))
                seen.add(w)

        deduped.sort(key=lambda x: abs(x[1]), reverse=True)
        return deduped[:self.num_features]

    def explain_instance_as_html(
        self,
        text: str,
        class_idx: Optional[int] = None,
        labels: Tuple[str, str, str] = ("negative", "neutral", "positive"),
    ) -> str:
        """Return explanation as HTML with color-coded words."""
        exp = self.explain_instance(text, class_idx)
        tokens = self.tokenizer(text)
        word_to_score = {w: s for w, s in exp}

        html_parts = []
        for token in tokens:
            score = word_to_score.get(token, 0)
            if score > 0:
                intensity = min(0.6, abs(score) * 2)
                html_parts.append(
                    f'<span style="background-color: rgba(0, 200, 83, {intensity:.2f}); '
                    f'padding: 2px 4px; border-radius: 3px;" '
                    f'title="{score:+.4f}">{token}</span>'
                )
            elif score < 0:
                intensity = min(0.6, abs(score) * 2)
                html_parts.append(
                    f'<span style="background-color: rgba(244, 67, 54, {intensity:.2f}); '
                    f'padding: 2px 4px; border-radius: 3px;" '
                    f'title="{score:+.4f}">{token}</span>'
                )
            else:
                html_parts.append(f'<span>{token}</span>')

        return " ".join(html_parts)
