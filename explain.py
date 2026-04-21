"""
Explain predictions using Vanilla LIME or Stability-Enhanced LIME.

Usage:
  python explain.py --text "Фильм был отличный!" --method vanilla
  python explain.py --text "Ужасный сервис." --method enhanced
  python explain.py --text "Обычный товар." --method both --save-html output.html
"""

import argparse
import yaml
from pathlib import Path

from src.model.rubert_classifier import RuBERTPipeline
from src.xai.lime_text import LimeTextExplainer
from src.xai.stability_lime import StabilityEnhancedLIME
from src.utils.visualization import plot_word_importance, plot_explanation_comparison


def main():
    parser = argparse.ArgumentParser(description="Explain RuBERT sentiment predictions")
    parser.add_argument("--text", type=str, help="Text to explain")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--model-path", default="checkpoints/rubert_sentiment.pt")
    parser.add_argument(
        "--method", choices=["vanilla", "enhanced", "both"],
        default="both", help="Explanation method"
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Override num_samples")
    parser.add_argument("--num-features", type=int, default=None, help="Override num_features")
    parser.add_argument("--n-runs", type=int, default=None, help="Override n_runs for enhanced")
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot to file")
    parser.add_argument("--save-html", type=str, default=None, help="Save HTML explanation")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    label_names = config["data"]["label_names"]
    lime_cfg = config.get("lime", {})
    enhanced_cfg = config.get("enhanced_lime", {})

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found at {model_path}. Run train.py first.")
        return

    pipeline = RuBERTPipeline(
        model_name=model_cfg["name"],
        model_path=str(model_path),
        num_labels=model_cfg["num_labels"],
        max_length=model_cfg["max_length"],
    )

    def predict_fn(texts):
        return pipeline.predict_proba(texts)

    text = args.text or "Фильм был отличный, очень понравилась игра актёров!"
    pred, probs = pipeline.predict([text], return_probs=True)
    pred_class = int(pred[0])
    pred_label = label_names[pred_class]
    pred_conf = probs[0][pred_class]
    
    print(f"\n{'='*60}")
    print(f"Input: {text}")
    print(f"Prediction: {pred_label} ({pred_conf:.2%})")
    print(f"{'='*60}")

    results = {}
    vanilla_explainer = None  # store for HTML
    enhanced_explainer = None  # store for HTML

    if args.method in ("vanilla", "both"):
        print("\n--- Vanilla LIME ---")
        vanilla_explainer = LimeTextExplainer(
            predict_fn=predict_fn,
            num_samples=args.num_samples or lime_cfg.get("num_samples", 5000),
            num_features=args.num_features or lime_cfg.get("num_features", 10),
            kernel_width=lime_cfg.get("kernel_width", 25.0),
        )
        exp_vanilla = vanilla_explainer.explain_instance(text, class_idx=pred_class)
        results["Vanilla LIME"] = exp_vanilla

        print("Word importance:")
        for word, score in exp_vanilla:
            print(f"  {word}: {score:+.4f}")

    if args.method in ("enhanced", "both"):
        print("\n--- Stability-Enhanced LIME ---")
        enhanced_explainer = StabilityEnhancedLIME(
            predict_fn=predict_fn,
            num_samples=args.num_samples or enhanced_cfg.get("num_samples", 5000),
            num_features=args.num_features or enhanced_cfg.get("num_features", 10),
            kernel_width=enhanced_cfg.get("kernel_width", 25.0),
            phrase_max_len=enhanced_cfg.get("phrase_max_len", 3),
            adjacency_window=enhanced_cfg.get("adjacency_window", 2),
            mask_rate=enhanced_cfg.get("mask_rate", 0.4),
            propagation_prob=enhanced_cfg.get("propagation_prob", 0.3),
            n_runs=args.n_runs or enhanced_cfg.get("n_runs", 5),
        )
        detail = enhanced_explainer.explain_instance_detailed(text, class_idx=pred_class)
        results["Enhanced LIME"] = detail["aggregated"]

        print("Word importance (aggregated):")
        for word, score in detail["aggregated"]:
            print(f"  {word}: {score:+.4f}")

        if detail.get("stability"):
            print("\nPer-word stability (variance, lower = more stable):")
            for word, var in sorted(detail["stability"].items(), key=lambda x: x[1]):
                print(f"  {word}: {var:.6f}")

    # Save plot
    if args.save_plot and results:
        if len(results) == 1:
            name, exp = next(iter(results.items()))
            plot_word_importance(exp, title=name, save_path=args.save_plot)
        else:
            plot_explanation_comparison(
                list(results.values()),
                list(results.keys()),
                save_path=args.save_plot,
            )
        print(f"\nPlot saved to {args.save_plot}")

    # Save HTML - use stored explainers instead of creating new ones
    if args.save_html:
        html_parts = [
            "<html><head><meta charset='utf-8'>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".explanation { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }",
            ".word { display: inline-block; margin: 2px; padding: 2px 4px; border-radius: 3px; }",
            ".positive { background-color: rgba(0, 200, 83, 0.4); }",
            ".negative { background-color: rgba(244, 67, 54, 0.4); }",
            "</style>",
            "</head><body>",
            f"<h1>Explanation for: {text}</h1>",
            f"<h2>Prediction: {pred_label} ({pred_conf:.2%})</h2>",
        ]
        
        if "Vanilla LIME" in results and vanilla_explainer:
            html_parts.append("<div class='explanation'>")
            html_parts.append("<h3>Vanilla LIME</h3>")
            html_parts.append(f"<p>{vanilla_explainer.explain_instance_as_html(text, pred_class)}</p>")
            html_parts.append("</div>")
        
        if "Enhanced LIME" in results and enhanced_explainer:
            html_parts.append("<div class='explanation'>")
            html_parts.append("<h3>Stability-Enhanced LIME</h3>")
            html_parts.append(f"<p>{enhanced_explainer.explain_instance_as_html(text, pred_class)}</p>")
            html_parts.append("</div>")
        
        html_parts.append("</body></html>")
        Path(args.save_html).write_text("\n".join(html_parts), encoding="utf-8")
        print(f"HTML saved to {args.save_html}")


if __name__ == "__main__":
    main()