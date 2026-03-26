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
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-features", type=int, default=None)
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
    print(f"\nInput: {text}")
    print(f"Prediction: {label_names[pred_class]} ({probs[0][pred_class]:.2%})")

    results = {}

    if args.method in ("vanilla", "both"):
        print("\n--- Vanilla LIME ---")
        vanilla = LimeTextExplainer(
            predict_fn=predict_fn,
            num_samples=args.num_samples or lime_cfg.get("num_samples", 5000),
            num_features=args.num_features or lime_cfg.get("num_features", 10),
            kernel_width=lime_cfg.get("kernel_width", 25.0),
        )
        exp_vanilla = vanilla.explain_instance(text, class_idx=pred_class)
        results["Vanilla LIME"] = exp_vanilla

        print("Word importance:")
        for word, score in exp_vanilla:
            print(f"  {word}: {score:+.4f}")

    if args.method in ("enhanced", "both"):
        print("\n--- Stability-Enhanced LIME ---")
        enhanced = StabilityEnhancedLIME(
            predict_fn=predict_fn,
            num_samples=args.num_samples or enhanced_cfg.get("num_samples", 5000),
            num_features=args.num_features or enhanced_cfg.get("num_features", 10),
            kernel_width=enhanced_cfg.get("kernel_width", 25.0),
            phrase_max_len=enhanced_cfg.get("phrase_max_len", 3),
            adjacency_window=enhanced_cfg.get("adjacency_window", 2),
            mask_rate=enhanced_cfg.get("mask_rate", 0.4),
            n_runs=enhanced_cfg.get("n_runs", 5),
        )
        detail = enhanced.explain_instance_detailed(text, class_idx=pred_class)
        results["Enhanced LIME"] = detail["aggregated"]

        print("Word importance (aggregated):")
        for word, score in detail["aggregated"]:
            print(f"  {word}: {score:+.4f}")

        print("\nPer-word stability (variance):")
        for word, var in sorted(detail["stability"].items(), key=lambda x: x[1]):
            print(f"  {word}: {var:.6f}")

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

    if args.save_html:
        html_parts = ["<html><head><meta charset='utf-8'></head><body>"]
        html_parts.append(f"<h2>Input: {text}</h2>")
        html_parts.append(f"<h3>Prediction: {label_names[pred_class]} ({probs[0][pred_class]:.2%})</h3>")

        if "Vanilla LIME" in results:
            vanilla_exp = LimeTextExplainer(predict_fn=predict_fn, num_samples=100, num_features=10)
            html_parts.append("<h3>Vanilla LIME</h3>")
            html_parts.append(f"<p>{vanilla_exp.explain_instance_as_html(text, pred_class)}</p>")

        if "Enhanced LIME" in results:
            enhanced_exp = StabilityEnhancedLIME(predict_fn=predict_fn, num_samples=100, num_features=10, n_runs=2)
            html_parts.append("<h3>Stability-Enhanced LIME</h3>")
            html_parts.append(f"<p>{enhanced_exp.explain_instance_as_html(text, pred_class)}</p>")

        html_parts.append("</body></html>")
        Path(args.save_html).write_text("\n".join(html_parts), encoding="utf-8")
        print(f"HTML saved to {args.save_html}")


if __name__ == "__main__":
    main()
