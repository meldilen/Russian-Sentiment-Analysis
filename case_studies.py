"""
Generate 3-5 qualitative case studies comparing Vanilla LIME vs Stability-Enhanced LIME.

Each case study includes:
  - Input text and prediction
  - Word importance from both methods
  - Stability analysis
  - Visual comparison
  - HTML report with highlighted words

Usage:
  python case_studies.py --config config.yaml
  python case_studies.py --output-dir results/case_studies
"""

import argparse
import yaml
import json
from pathlib import Path
from typing import List
import numpy as np

from src.model.rubert_classifier import RuBERTPipeline
from src.xai.lime_text import LimeTextExplainer, _tokenize_russian
from src.xai.stability_lime import StabilityEnhancedLIME
from src.utils.metrics import (
    compute_stability_score,
    compute_faithfulness,
    compute_sparsity,
    compute_rank_correlation,
)
from src.utils.visualization import (
    plot_word_importance,
    plot_explanation_comparison,
    plot_stability_heatmap,
)


CASE_STUDY_TEXTS = [
    {
        "text": "Фильм был отличный, очень понравилась игра актёров!",
        "expected": "positive",
        "description": "Positive movie review with strong sentiment words",
    },
    {
        "text": "Ужасный сервис, больше никогда не приду сюда.",
        "expected": "negative",
        "description": "Negative service review with negation and strong negative word",
    },
    {
        "text": "Обычный продукт, ничего особенного не заметил.",
        "expected": "neutral",
        "description": "Neutral review with hedging language",
    },
    {
        "text": "Потрясающая книга, не мог оторваться до последней страницы!",
        "expected": "positive",
        "description": "Positive book review with superlative and engagement markers",
    },
    {
        "text": "Качество оставляет желать лучшего, разочарован покупкой.",
        "expected": "negative",
        "description": "Negative product review with idiomatic expression",
    },
]


def generate_case_study(
    idx: int,
    case: dict,
    predict_fn,
    label_names: List[str],
    lime_cfg: dict,
    enhanced_cfg: dict,
    output_dir: Path,
) -> dict:
    """Generate a complete case study for one text."""
    text = case["text"]
    print(f"\n{'='*60}")
    print(f"Case Study {idx}: {case['description']}")
    print(f"{'='*60}")
    print(f"Text: {text}")
    print(f"Expected: {case['expected']}")

    probs = predict_fn([text])[0]
    pred_class = int(np.argmax(probs))
    pred_label = label_names[pred_class]
    pred_conf = float(probs[pred_class])
    print(f"Predicted: {pred_label} ({pred_conf:.2%})")

    # Vanilla LIME (5 runs for stability)
    n_stability_runs = 5
    vanilla_runs = []
    for i in range(n_stability_runs):
        exp = LimeTextExplainer(
            predict_fn=predict_fn,
            num_samples=lime_cfg.get("num_samples", 5000),
            num_features=lime_cfg.get("num_features", 10),
            kernel_width=lime_cfg.get("kernel_width", 25.0),
            random_state=i * 17,
        )
        vanilla_runs.append(exp.explain_instance(text, class_idx=pred_class))

    # Enhanced LIME
    enhanced = StabilityEnhancedLIME(
        predict_fn=predict_fn,
        num_samples=enhanced_cfg.get("num_samples", 5000),
        num_features=enhanced_cfg.get("num_features", 10),
        kernel_width=enhanced_cfg.get("kernel_width", 25.0),
        phrase_max_len=enhanced_cfg.get("phrase_max_len", 3),
        adjacency_window=enhanced_cfg.get("adjacency_window", 2),
        mask_rate=enhanced_cfg.get("mask_rate", 0.4),
        n_runs=enhanced_cfg.get("n_runs", 5),
    )
    enhanced_detail = enhanced.explain_instance_detailed(text, class_idx=pred_class)

    # Metrics
    v_stability = compute_stability_score(vanilla_runs)
    e_stability = compute_stability_score(enhanced_detail["per_run"])

    v_faithfulness = compute_faithfulness(text, _tokenize_russian, predict_fn, vanilla_runs[0], top_k=5)
    e_faithfulness = compute_faithfulness(text, _tokenize_russian, predict_fn, enhanced_detail["aggregated"], top_k=5)

    v_sparsity = compute_sparsity(vanilla_runs[0])
    e_sparsity = compute_sparsity(enhanced_detail["aggregated"])

    v_rank_corr = compute_rank_correlation(vanilla_runs)
    e_rank_corr = compute_rank_correlation(enhanced_detail["per_run"])

    print(f"\n  Vanilla LIME:  stability={v_stability:.4f}, faithfulness={v_faithfulness:.4f}, "
          f"sparsity={v_sparsity:.4f}, rank_corr={v_rank_corr:.4f}")
    print(f"  Enhanced LIME: stability={e_stability:.4f}, faithfulness={e_faithfulness:.4f}, "
          f"sparsity={e_sparsity:.4f}, rank_corr={e_rank_corr:.4f}")

    # Visualizations
    case_dir = output_dir / f"case_{idx}"
    case_dir.mkdir(parents=True, exist_ok=True)

    plot_explanation_comparison(
        [vanilla_runs[0], enhanced_detail["aggregated"]],
        ["Vanilla LIME", "Enhanced LIME"],
        title=f"Case {idx}: {case['description']}",
        save_path=str(case_dir / "comparison.png"),
    )

    plot_stability_heatmap(
        vanilla_runs,
        title=f"Case {idx}: Vanilla LIME Stability ({n_stability_runs} runs)",
        save_path=str(case_dir / "vanilla_stability.png"),
    )

    plot_stability_heatmap(
        enhanced_detail["per_run"],
        title=f"Case {idx}: Enhanced LIME Stability",
        save_path=str(case_dir / "enhanced_stability.png"),
    )

    plot_word_importance(
        vanilla_runs[0],
        title=f"Case {idx}: Vanilla LIME",
        save_path=str(case_dir / "vanilla_importance.png"),
    )

    plot_word_importance(
        enhanced_detail["aggregated"],
        title=f"Case {idx}: Enhanced LIME",
        save_path=str(case_dir / "enhanced_importance.png"),
    )

    # HTML report
    html = _generate_html_report(
        idx, case, pred_label, pred_conf, probs, label_names,
        vanilla_runs[0], enhanced_detail,
        {
            "vanilla": {"stability": v_stability, "faithfulness": v_faithfulness,
                        "sparsity": v_sparsity, "rank_correlation": v_rank_corr},
            "enhanced": {"stability": e_stability, "faithfulness": e_faithfulness,
                         "sparsity": e_sparsity, "rank_correlation": e_rank_corr},
        },
    )
    (case_dir / "report.html").write_text(html, encoding="utf-8")

    return {
        "text": text,
        "description": case["description"],
        "expected": case["expected"],
        "predicted": pred_label,
        "confidence": pred_conf,
        "vanilla_metrics": {
            "stability": v_stability, "faithfulness": v_faithfulness,
            "sparsity": v_sparsity, "rank_correlation": v_rank_corr,
        },
        "enhanced_metrics": {
            "stability": e_stability, "faithfulness": e_faithfulness,
            "sparsity": e_sparsity, "rank_correlation": e_rank_corr,
        },
        "vanilla_explanation": [(w, round(s, 6)) for w, s in vanilla_runs[0]],
        "enhanced_explanation": [(w, round(s, 6)) for w, s in enhanced_detail["aggregated"]],
    }


def _generate_html_report(
    idx, case, pred_label, pred_conf, probs, label_names,
    vanilla_exp, enhanced_detail, metrics,
):
    tokens = _tokenize_russian(case["text"])

    def highlight(exp):
        scores = {w: s for w, s in exp}
        parts = []
        for t in tokens:
            s = scores.get(t, 0)
            if s > 0:
                intensity = min(0.7, abs(s) * 3)
                parts.append(f'<span style="background:rgba(76,175,80,{intensity:.2f});padding:2px 4px;border-radius:3px" title="{s:+.4f}">{t}</span>')
            elif s < 0:
                intensity = min(0.7, abs(s) * 3)
                parts.append(f'<span style="background:rgba(244,67,54,{intensity:.2f});padding:2px 4px;border-radius:3px" title="{s:+.4f}">{t}</span>')
            else:
                parts.append(t)
        return " ".join(parts)

    probs_str = ", ".join(f"{label_names[i]}: {probs[i]:.2%}" for i in range(len(label_names)))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 10px; }}
h2 {{ color: #283593; }}
.prediction {{ background: #e8eaf6; padding: 15px; border-radius: 8px; margin: 15px 0; }}
.highlight {{ font-size: 1.2em; line-height: 2; padding: 15px; background: #fafafa; border-radius: 8px; margin: 10px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
th {{ background: #e8eaf6; font-weight: 600; }}
.better {{ color: #2e7d32; font-weight: bold; }}
img {{ max-width: 100%; margin: 10px 0; border: 1px solid #eee; border-radius: 8px; }}
</style>
</head><body>
<h1>Case Study {idx}: {case['description']}</h1>

<div class="prediction">
<strong>Input:</strong> {case['text']}<br>
<strong>Expected:</strong> {case['expected']}<br>
<strong>Predicted:</strong> {pred_label} ({pred_conf:.2%})<br>
<strong>All probabilities:</strong> {probs_str}
</div>

<h2>Vanilla LIME Explanation</h2>
<div class="highlight">{highlight(vanilla_exp)}</div>

<h2>Stability-Enhanced LIME Explanation</h2>
<div class="highlight">{highlight(enhanced_detail['aggregated'])}</div>

<h2>Quantitative Comparison</h2>
<table>
<tr><th>Metric</th><th>Vanilla LIME</th><th>Enhanced LIME</th><th>Winner</th></tr>"""

    for metric_name in ["stability", "faithfulness", "sparsity", "rank_correlation"]:
        v = metrics["vanilla"][metric_name]
        e = metrics["enhanced"][metric_name]
        winner = "Enhanced" if e > v else ("Vanilla" if v > e else "Tie")
        v_cls = ' class="better"' if v > e else ""
        e_cls = ' class="better"' if e > v else ""
        html += f"\n<tr><td>{metric_name.replace('_', ' ').title()}</td>"
        html += f"<td{v_cls}>{v:.4f}</td><td{e_cls}>{e:.4f}</td><td>{winner}</td></tr>"

    html += """
</table>

<h2>Visualizations</h2>
<h3>Side-by-Side Comparison</h3>
<img src="comparison.png" alt="Comparison">

<h3>Vanilla LIME Stability (across runs)</h3>
<img src="vanilla_stability.png" alt="Vanilla stability">

<h3>Enhanced LIME Stability (across runs)</h3>
<img src="enhanced_stability.png" alt="Enhanced stability">

</body></html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate case studies")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-path", default="checkpoints/rubert_sentiment.pt")
    parser.add_argument("--output-dir", default="results/case_studies")
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

    print("Loading model...")
    pipeline = RuBERTPipeline(
        model_name=model_cfg["name"],
        model_path=str(model_path),
        num_labels=model_cfg["num_labels"],
        max_length=model_cfg["max_length"],
    )

    predict_fn = pipeline.predict_proba
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_texts = config.get("evaluation", {}).get("case_study_texts", None)
    cases = CASE_STUDY_TEXTS
    if eval_texts:
        cases = [{"text": t, "expected": "unknown", "description": f"Text {i+1}"}
                 for i, t in enumerate(eval_texts)]

    all_results = []
    for idx, case in enumerate(cases, 1):
        result = generate_case_study(
            idx, case, predict_fn, label_names,
            lime_cfg, enhanced_cfg, output_dir,
        )
        all_results.append(result)

    # Summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Index HTML
    index_html = _generate_index_html(all_results)
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    print(f"Index page saved to {output_dir / 'index.html'}")


def _generate_index_html(results):
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body { font-family: 'Segoe UI', sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #333; }
h1 { color: #1a237e; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #ddd; }
th { background: #e8eaf6; }
a { color: #1565c0; }
.better { color: #2e7d32; font-weight: bold; }
</style>
</head><body>
<h1>Stability-Enhanced LIME: Case Study Results</h1>
<p>Comparative analysis of Vanilla LIME vs Stability-Enhanced LIME on Russian sentiment classification.</p>

<table>
<tr><th>#</th><th>Text</th><th>Predicted</th>
<th>V-Stability</th><th>E-Stability</th>
<th>V-Faithfulness</th><th>E-Faithfulness</th>
<th>Report</th></tr>"""

    for i, r in enumerate(results, 1):
        vm = r["vanilla_metrics"]
        em = r["enhanced_metrics"]
        s_v_cls = ' class="better"' if vm["stability"] > em["stability"] else ""
        s_e_cls = ' class="better"' if em["stability"] > vm["stability"] else ""
        f_v_cls = ' class="better"' if vm["faithfulness"] > em["faithfulness"] else ""
        f_e_cls = ' class="better"' if em["faithfulness"] > vm["faithfulness"] else ""

        html += f"""
<tr>
<td>{i}</td>
<td>{r['text'][:50]}...</td>
<td>{r['predicted']} ({r['confidence']:.0%})</td>
<td{s_v_cls}>{vm['stability']:.3f}</td>
<td{s_e_cls}>{em['stability']:.3f}</td>
<td{f_v_cls}>{vm['faithfulness']:.3f}</td>
<td{f_e_cls}>{em['faithfulness']:.3f}</td>
<td><a href="case_{i}/report.html">View</a></td>
</tr>"""

    html += """
</table>
</body></html>"""
    return html


if __name__ == "__main__":
    main()
