"""
Vanilla LIME implementation for text (Ribeiro et al., 2016).
Explains individual predictions by approximating the model locally with a sparse linear surrogate.
"""

import numpy as np
from typing import Callable, List, Optional, Tuple, Dict
from collections import defaultdict
import re


def _tokenize_russian(text: str) -> List[str]:
    """
    Simple tokenizer for Russian text.
    Splits on whitespace and punctuation while preserving words.
    """
    # Split on whitespace, keep words
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class LimeTextExplainer:
    """
    Vanilla LIME for text explanations.
    
    Perturbs input by removing words (masking), generates neighborhood samples,
    fits a sparse linear surrogate, and extracts word importance scores.
    
    Implements the LIME objective:
    argmin_g L(f, g, π_x) + Ω(g)
    where f is the black-box model, g is the interpretable surrogate,
    π_x is the locality weighting, and Ω(g) penalizes complexity.
    """
    
    def __init__(
        self,
        predict_fn: Callable[[List[str]], np.ndarray],
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        num_samples: int = 5000,
        num_features: int = 10,
        num_splits: int = 50,
        kernel_width: float = 25.0,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            predict_fn: Function that takes list of strings and returns
                       probability array of shape (n_samples, n_classes)
            tokenizer: Function to split text into tokens. Default: simple word tokenizer
            num_samples: Number of perturbed samples to generate
            num_features: Max number of features in explanation
            num_splits: Number of splits for feature selection
            kernel_width: Width of exponential kernel for locality weighting
            random_state: Random seed for reproducibility
        """
        self.predict_fn = predict_fn
        self.tokenizer = tokenizer or _tokenize_russian
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_splits = num_splits
        self.kernel_width = kernel_width
        self.rng = np.random.default_rng(random_state)
        
    def _perturb_text(self, text: str, tokens: List[str], mask: np.ndarray) -> str:
        """Create perturbed text by masking words according to binary mask (1=keep, 0=remove)."""
        if len(tokens) == 0:
            return ""
        
        # Reconstruct text with masked tokens replaced by empty string
        perturbed_tokens = [
            tok if mask[i] == 1 else ""
            for i, tok in enumerate(tokens)
        ]
        # Join with spaces, collapse multiple spaces
        return " ".join(t for t in perturbed_tokens if t).strip() or " "
    
    def _generate_neighborhood(
        self,
        text: str,
        tokens: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate neighborhood samples by randomly masking words.
        
        Returns:
            data: Binary matrix (num_samples, num_tokens), 1=word present
            labels: Model predictions for perturbed samples
        """
        n_tokens = len(tokens)
        if n_tokens == 0:
            return np.zeros((self.num_samples, 1)), np.zeros((self.num_samples, 1))
        
        # Random binary masks - favor keeping ~50% of words on average
        data = self.rng.integers(0, 2, size=(self.num_samples, n_tokens))
        
        # Ensure at least one sample is unperturbed (all 1s)
        data[0] = 1
        
        # Get model predictions for perturbed samples
        perturbed_texts = [
            self._perturb_text(text, tokens, data[i])
            for i in range(self.num_samples)
        ]
        
        # Batch prediction
        labels = self.predict_fn(perturbed_texts)
        
        return data, labels
    
    def _kernel(self, distances: np.ndarray) -> np.ndarray:
        """Exponential kernel for locality weighting: π_x(z) = exp(-D(x,z)^2 / σ^2)"""
        return np.sqrt(np.exp(-(distances ** 2) / (self.kernel_width ** 2)))
    
    def explain_instance(
        self,
        text: str,
        class_idx: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Explain a single prediction.
        
        Args:
            text: Input text to explain
            class_idx: Class to explain (default: predicted class)
            
        Returns:
            List of (word, importance_score) tuples, sorted by absolute importance
        """
        tokens = self.tokenizer(text)
        if len(tokens) == 0:
            return []
        
        # Get prediction for original text
        orig_probs = self.predict_fn([text])[0]
        if class_idx is None:
            class_idx = int(np.argmax(orig_probs))
        
        # Generate neighborhood
        data, labels = self._generate_neighborhood(text, tokens)
        
        # Labels for the class of interest
        if labels.ndim == 2:
            y = labels[:, class_idx]
        else:
            y = labels
        
        # Distance = number of different features (Hamming)
        distances = np.sum(data != 1, axis=1)
        weights = self._kernel(distances.astype(float))
        
        # Fit weighted linear regression: minimize L = Σ π_x(z) * (f(z) - g(z))^2
        from sklearn.linear_model import Ridge
        
        # Ridge with high regularization for sparsity
        model = Ridge(alpha=1.0)
        model.fit(data, y, sample_weight=weights)
        
        coefficients = model.coef_
        
        # Build word importance list
        word_scores = [
            (tokens[i], float(coefficients[i]))
            for i in range(len(tokens))
        ]
        
        # Sort by absolute importance
        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return top features
        return word_scores[: self.num_features]
    
    def explain_instance_multiple_runs(
        self,
        text: str,
        class_idx: Optional[int] = None,
        n_runs: int = 5,
        aggregation: str = "median",  # 'mean' or 'median'
    ) -> List[Tuple[str, float]]:
        """
        Run LIME multiple times and aggregate results for stability.
        
        Args:
            text: Input text to explain
            class_idx: Class to explain
            n_runs: Number of runs
            aggregation: 'mean' or 'median'
            
        Returns:
            Aggregated list of (word, importance_score)
        """
        tokens = self.tokenizer(text)
        if not tokens:
            return []
        
        all_runs = []
        for i in range(n_runs):
            # Save and restore RNG state to ensure independence
            saved_state = self.rng.bit_generator.state
            exp = self.explain_instance(text, class_idx)
            all_runs.append(exp)
            self.rng.bit_generator.state = saved_state
            # Advance RNG for next run
            self.rng.random()
        
        return self._aggregate_runs(all_runs, tokens, aggregation)
    
    def explain_instance_detailed(
        self,
        text: str,
        class_idx: Optional[int] = None,
        n_runs: int = 5,
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
        
        all_runs = []
        for i in range(n_runs):
            saved_state = self.rng.bit_generator.state
            exp = self.explain_instance(text, class_idx)
            all_runs.append(exp)
            self.rng.bit_generator.state = saved_state
            self.rng.random()
        
        aggregated = self._aggregate_runs(all_runs, tokens, aggregation="median")
        
        # Calculate per-word variance for stability
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
        aggregation: str = "median",
    ) -> List[Tuple[str, float]]:
        """Aggregate multiple runs by computing mean or median importance per word."""
        word_scores: Dict[str, List[float]] = defaultdict(list)
        for run in all_runs:
            for w, s in run:
                word_scores[w].append(s)
        
        aggregated = []
        for word in tokens:
            if word in word_scores:
                if aggregation == "median":
                    score = float(np.median(word_scores[word]))
                else:
                    score = float(np.mean(word_scores[word]))
                aggregated.append((word, score))
        
        # Deduplicate and sort
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
        """
        Return explanation as HTML with highlighted words.
        Green = positive contribution, Red = negative contribution.
        """
        exp = self.explain_instance(text, class_idx)
        return self._to_html(exp, text)

    def explain_instance_as_html_detailed(
        self,
        text: str,
        class_idx: Optional[int] = None,
        n_runs: int = 5,
    ) -> str:
        """
        Return HTML explanation with aggregated results from multiple runs.
        """
        detail = self.explain_instance_detailed(text, class_idx, n_runs)
        return self._to_html(detail["aggregated"], text)
    
    def _to_html(self, exp: List[Tuple[str, float]], text: str) -> str:
        """Convert explanation to HTML with color-coded words."""
        tokens = self.tokenizer(text)
        word_to_score = {w: s for w, s in exp}
        
        html_parts = []
        for token in tokens:
            score = word_to_score.get(token, 0)
            if score > 0:
                # Green for positive contribution
                intensity = min(0.6, abs(score) * 2)
                html_parts.append(
                    f'<span style="background-color: rgba(0, 200, 83, {intensity:.2f}); '
                    f'padding: 2px 4px; border-radius: 3px;" '
                    f'title="{score:+.4f}">{token}</span>'
                )
            elif score < 0:
                # Red for negative contribution
                intensity = min(0.6, abs(score) * 2)
                html_parts.append(
                    f'<span style="background-color: rgba(244, 67, 54, {intensity:.2f}); '
                    f'padding: 2px 4px; border-radius: 3px;" '
                    f'title="{score:+.4f}">{token}</span>'
                )
            else:
                html_parts.append(f'<span>{token}</span>')
        
        return " ".join(html_parts)