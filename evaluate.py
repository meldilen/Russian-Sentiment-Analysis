"""
Comprehensive evaluation pipeline comparing Vanilla LIME vs Stability-Enhanced LIME.

Produces:
  - Stability scores (variance across runs)
  - Faithfulness metrics (confidence drop)
  - Sparsity comparison (Gini coefficient)
  - Rank correlation (Spearman rho)
  - Deletion curves
  - Visualization outputs

Usage:
  python evaluate.py --config config.yaml
  python evaluate.py --config config.yaml --output-dir results/
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

from src.model.rubert_classifier import RuBERTPipeline
from src.xai.lime_text import LimeTextExplainer, _tokenize_russian
from src.xai.stability_lime import StabilityEnhancedLIME
from src.utils.metrics import (
    compute_stability_score,
    compute_faithfulness,
    compute_sparsity,
    compute_rank_correlation,
    compute_incremental_faithfulness,
    compute_all_metrics,
)
from src.utils.visualization import (
    plot_word_importance,
    plot_explanation_comparison,
    plot_stability_heatmap,
    plot_metrics_comparison,
    plot_deletion_curve,
)


def run_vanilla_lime_multiple(
    predict_fn,
    text: str,
    class_idx: int,
    n_runs: int,
    num_samples: int = 5000,
    num_features: int = 10,
    kernel_width: float = 25.0,
) -> List[List[Tuple[str, float]]]:
    """Run vanilla LIME multiple times for stability analysis."""
    results = []
    for i in range(n_runs):
        explainer = LimeTextExplainer(
            predict_fn=predict_fn,
            num_samples=num_samples,
            num_features=num_features,
            kernel_width=kernel_width,
            random_state=i * 42,
        )
        exp = explainer.explain_instance(text, class_idx=class_idx)
        results.append(exp)
    return results


def run_enhanced_lime_detailed(
    predict_fn,
    text: str,
    class_idx: int,
    enhanced_cfg: dict,
) -> Dict:
    """Run enhanced LIME and return detailed results."""
    explainer = StabilityEnhancedLIME(
        predict_fn=predict_fn,
        num_samples=enhanced_cfg.get("num_samples", 5000),
        num_features=enhanced_cfg.get("num_features", 10),
        kernel_width=enhanced_cfg.get("kernel_width", 25.0),
        phrase_max_len=enhanced_cfg.get("phrase_max_len", 3),
        adjacency_window=enhanced_cfg.get("adjacency_window", 2),
        mask_rate=enhanced_cfg.get("mask_rate", 0.4),
        n_runs=enhanced_cfg.get("n_runs", 5),
    )
    return explainer.explain_instance_detailed(text, class_idx=class_idx)


def evaluate_text(
    predict_fn,
    text: str,
    label_names: List[str],
    lime_cfg: dict,
    enhanced_cfg: dict,
    stability_runs: int = 10,
    faithfulness_top_k: int = 5,
) -> Dict:
    """Full evaluation on a single text."""
    probs = predict_fn([text])[0]
    pred_class = int(np.argmax(probs))
    pred_label = label_names[pred_class]
    pred_conf = float(probs[pred_class])

    print(f"  Text: {text[:80]}...")
    print(f"  Prediction: {pred_label} ({pred_conf:.2%})")

    # Vanilla LIME: multiple runs
    print("  Running Vanilla LIME...")
    vanilla_runs = run_vanilla_lime_multiple(
        predict_fn, text, pred_class,
        n_runs=stability_runs,
        num_samples=lime_cfg.get("num_samples", 5000),
        num_features=lime_cfg.get("num_features", 10),
        kernel_width=lime_cfg.get("kernel_width", 25.0),
    )

    vanilla_metrics = compute_all_metrics(
        text, _tokenize_russian, predict_fn, vanilla_runs, top_k=faithfulness_top_k
    )

    # Enhanced LIME
    print("  Running Enhanced LIME...")
    enhanced_detail = run_enhanced_lime_detailed(predict_fn, text, pred_class, enhanced_cfg)

    enhanced_metrics = compute_all_metrics(
        text, _tokenize_russian, predict_fn, enhanced_detail["per_run"], top_k=faithfulness_top_k
    )

    # Deletion curves
    vanilla_deletion = compute_incremental_faithfulness(
        text, _tokenize_russian, predict_fn, vanilla_runs[0]
    )
    enhanced_deletion = compute_incremental_faithfulness(
        text, _tokenize_russian, predict_fn, enhanced_detail["aggregated"]
    )

    return {
        "text": text,
        "prediction": pred_label,
        "confidence": pred_conf,
        "vanilla": {
            "explanation": vanilla_runs[0],
            "all_runs": vanilla_runs,
            "metrics": vanilla_metrics,
            "deletion_curve": vanilla_deletion,
        },
        "enhanced": {
            "explanation": enhanced_detail["aggregated"],
            "all_runs": enhanced_detail["per_run"],
            "metrics": enhanced_metrics,
            "deletion_curve": enhanced_deletion,
            "per_word_stability": enhanced_detail["stability"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LIME methods")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-path", default="checkpoints/rubert_sentiment.pt")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--texts", nargs="*", help="Override evaluation texts")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    label_names = config["data"]["label_names"]
    lime_cfg = config.get("lime", {})
    enhanced_cfg = config.get("enhanced_lime", {})
    eval_cfg = config.get("evaluation", {})

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found at {model_path}. Run train.py first.")
        return

    print("Loading model...")
    pipeline = RuBERTPipeline(
        model_name=model_cfg["name"],
        model_path=str(model_path),
        num_labels=model_cfg["num_labels"],
        max_length=model_cfg["max_length"],
    )

    def predict_fn(texts):
        return pipeline.predict_proba(texts)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = args.texts or eval_cfg.get("case_study_texts", [
        "Фильм был отличный, очень понравилась игра актёров!",
        "Ужасный сервис, больше никогда не приду сюда.",
        "Обычный продукт, ничего особенного не заметил.",
    ])

    stability_runs = eval_cfg.get("stability_runs", 10)
    top_k = eval_cfg.get("faithfulness_top_k", 5)

    all_results = []
    all_vanilla_metrics = []
    all_enhanced_metrics = []

    for idx, text in enumerate(texts):
        print(f"\n{'='*60}")
        print(f"Case Study {idx + 1}/{len(texts)}")
        print(f"{'='*60}")

        result = evaluate_text(
            predict_fn, text, label_names,
            lime_cfg, enhanced_cfg,
            stability_runs=stability_runs,
            faithfulness_top_k=top_k,
        )
        all_results.append(result)
        all_vanilla_metrics.append(result["vanilla"]["metrics"])
        all_enhanced_metrics.append(result["enhanced"]["metrics"])

        # Per-text visualizations
        plot_explanation_comparison(
            [result["vanilla"]["explanation"], result["enhanced"]["explanation"]],
            ["Vanilla LIME", "Enhanced LIME"],
            title=f"Case {idx+1}: {text[:50]}...",
            save_path=str(output_dir / f"case_{idx+1}_comparison.png"),
        )

        plot_stability_heatmap(
            result["vanilla"]["all_runs"][:5],
            title=f"Case {idx+1}: Vanilla LIME Stability",
            save_path=str(output_dir / f"case_{idx+1}_vanilla_stability.png"),
        )

        plot_stability_heatmap(
            result["enhanced"]["all_runs"],
            title=f"Case {idx+1}: Enhanced LIME Stability",
            save_path=str(output_dir / f"case_{idx+1}_enhanced_stability.png"),
        )

        if result["vanilla"]["deletion_curve"] and result["enhanced"]["deletion_curve"]:
            plot_deletion_curve(
                {
                    "Vanilla LIME": result["vanilla"]["deletion_curve"],
                    "Enhanced LIME": result["enhanced"]["deletion_curve"],
                },
                title=f"Case {idx+1}: Deletion Curve",
                save_path=str(output_dir / f"case_{idx+1}_deletion.png"),
            )

    # Aggregate metrics
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    avg_vanilla = {}
    avg_enhanced = {}
    for key in ["stability", "faithfulness", "sparsity", "rank_correlation"]:
        avg_vanilla[key] = float(np.mean([m[key] for m in all_vanilla_metrics]))
        avg_enhanced[key] = float(np.mean([m[key] for m in all_enhanced_metrics]))

    print("\nVanilla LIME (average):")
    for k, v in avg_vanilla.items():
        print(f"  {k}: {v:.4f}")

    print("\nEnhanced LIME (average):")
    for k, v in avg_enhanced.items():
        print(f"  {k}: {v:.4f}")

    improvement = {}
    for key in avg_vanilla:
        diff = avg_enhanced[key] - avg_vanilla[key]
        improvement[key] = diff
        sign = "+" if diff >= 0 else ""
        print(f"  Improvement in {key}: {sign}{diff:.4f}")

    # Summary visualization
    plot_metrics_comparison(
        {"Vanilla LIME": avg_vanilla, "Enhanced LIME": avg_enhanced},
        save_path=str(output_dir / "metrics_comparison.png"),
    )

    # Save results JSON
    def _to_native(obj):
        """Convert numpy types to Python native for JSON serialization."""
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    serializable_results = []
    for r in all_results:
        sr = {
            "text": r["text"],
            "prediction": r["prediction"],
            "confidence": float(r["confidence"]),
            "vanilla_metrics": _to_native(r["vanilla"]["metrics"]),
            "enhanced_metrics": _to_native(r["enhanced"]["metrics"]),
            "vanilla_explanation": [(w, round(float(s), 6)) for w, s in r["vanilla"]["explanation"]],
            "enhanced_explanation": [(w, round(float(s), 6)) for w, s in r["enhanced"]["explanation"]],
        }
        serializable_results.append(sr)

    summary = {
        "aggregate": {
            "vanilla_lime": avg_vanilla,
            "enhanced_lime": avg_enhanced,
            "improvement": improvement,
        },
        "case_studies": serializable_results,
    }

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
