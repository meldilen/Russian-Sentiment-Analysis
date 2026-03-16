"""
Explain predictions using vanilla LIME.
"""

import argparse
import yaml
from pathlib import Path

from src.model.rubert_classifier import RuBERTPipeline
from src.xai.lime_text import LimeTextExplainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to explain")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--model-path", default="checkpoints/rubert_sentiment.pt")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--num-features", type=int, default=10)
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model"]
    label_names = config["data"]["label_names"]
    
    # Load model
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
    
    # Predict function for LIME (returns probabilities)
    def predict_fn(texts):
        return pipeline.predict_proba(texts)
    
    explainer = LimeTextExplainer(
        predict_fn=predict_fn,
        num_samples=args.num_samples,
        num_features=args.num_features,
    )
    
    text = args.text or "Фильм был отличный, очень понравилась игра актёров!"
    print(f"Input: {text}")
    
    pred, probs = pipeline.predict([text], return_probs=True)
    pred_class = int(pred[0])
    print(f"Prediction: {label_names[pred_class]} ({probs[0][pred_class]:.2%})")
    
    print("\nWord importance (LIME):")
    exp = explainer.explain_instance(text, class_idx=pred_class)
    for word, score in exp:
        sign = "+" if score > 0 else ""
        print(f"  {word}: {sign}{score:.4f}")


if __name__ == "__main__":
    main()
