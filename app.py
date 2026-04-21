"""
Interactive Web Application for Comparing Vanilla LIME vs Stability-Enhanced LIME
Russian Sentiment Analysis with RuBERT
Professional UI/UX implementation
"""

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model.rubert_classifier import RuBERTPipeline
from src.xai.lime_text import LimeTextExplainer, _tokenize_russian
from src.xai.stability_lime import StabilityEnhancedLIME
from src.utils.metrics import (
    compute_stability_score,
    compute_faithfulness,
    compute_sparsity,
    compute_rank_correlation,
)

# Page configuration
st.set_page_config(
    page_title="LIME Comparison | Russian Sentiment Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background-color: #1a1a2e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #16213e;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.25rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.875rem;
        font-weight: 500;
        opacity: 0.8;
        letter-spacing: 0.5px;
    }
    .metric-card h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 0.875rem;
        opacity: 0.9;
    }
    
    /* Color variants */
    .metric-primary {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .metric-success {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    }
    .metric-warning {
        background: linear-gradient(135deg, #e94560 0%, #c73e56 100%);
    }
    
    /* Word highlighting */
    .word-span {
        display: inline-block;
        margin: 2px;
        padding: 2px 6px;
        border-radius: 4px;
        transition: all 0.2s ease;
        cursor: pointer;
        font-size: 1.1em;
    }
    .word-span:hover {
        transform: scale(1.02);
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #f5f5f5;
        padding: 0.25rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        color: #666;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8e8e8;
        color: #1a1a2e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a2e;
        color: white;
        border-bottom: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* Info box */
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #1a1a2e;
        font-size: 0.875rem;
        color: #555;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #1a1a2e;
    }
    
    /* Caption styling */
    .caption {
        font-size: 0.75rem;
        color: #888;
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("Vanilla LIME vs Stability-Enhanced LIME")
st.caption("Interactive comparison of explanation methods for Russian sentiment analysis")

st.markdown("---")

with st.sidebar:
    st.markdown("### Configuration")
    
    # Model section
    st.markdown("#### Model")
    model_path = st.text_input(
        "Model path",
        value="checkpoints/rubert_sentiment_best.pt",
        help="Path to trained RuBERT model checkpoint"
    )
    
    st.markdown("---")
    
    # LIME parameters
    st.markdown("#### LIME Parameters")
    
    num_samples = st.slider(
        "Number of perturbations",
        min_value=500,
        max_value=5000,
        value=2000,
        step=500,
        help="More samples increase accuracy but slow down computation"
    )
    
    num_features = st.slider(
        "Number of top features",
        min_value=5,
        max_value=15,
        value=10,
        step=1,
        help="Number of most important words to display"
    )
    
    st.markdown("---")
    
    # Enhanced LIME parameters
    st.markdown("#### Enhanced LIME")
    
    n_runs = st.slider(
        "Aggregation runs",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="More runs improve stability but increase computation time"
    )
    
    mask_rate = st.slider(
        "Mask rate",
        min_value=0.1,
        max_value=0.7,
        value=0.4,
        step=0.05,
        help="Fraction of tokens to mask during perturbation"
    )
    
    propagation_prob = st.slider(
        "Propagation probability",
        min_value=0.1,
        max_value=0.7,
        value=0.3,
        step=0.05,
        help="Probability of masking neighboring tokens"
    )
    
    st.markdown("---")
    
    # Example texts
    st.markdown("#### Example Texts")
    
    EXAMPLES = {
        "Positive - Movie Review": "Фильм был отличный, очень понравилась игра актёров!",
        "Negative - Service Review": "Ужасный сервис, больше никогда не приду сюда.",
        "Neutral - Product Review": "Обычный продукт, ничего особенного не заметил.",
        "Positive - Book Review": "Потрясающая книга, не мог оторваться до последней страницы!",
        "Negative - Quality Review": "Качество оставляет желать лучшего, разочарован покупкой.",
        "Mixed - Balanced Review": "Цена высокая, но качество хорошее. В целом нормально."
    }
    
    selected_example = st.selectbox("Select example", list(EXAMPLES.keys()))
    example_text = EXAMPLES[selected_example]
    
    st.markdown("---")
    st.markdown('<div class="caption">Tip: Hover over highlighted words to see scores</div>', unsafe_allow_html=True)

# Text input area
col1, col2 = st.columns([4, 1])

with col1:
    user_text = st.text_area(
        "Input Text",
        value=example_text,
        height=100,
        help="Enter Russian text for sentiment analysis"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Load Example", width='stretch'):
        user_text = example_text
        st.rerun()

# Analysis button
analyze_button = st.button("Analyze Text", type="primary", width='stretch')

# ============================================================
# CACHED FUNCTIONS
# ============================================================

@st.cache_resource
def load_model(model_path_str):
    """Load model with caching for performance"""
    try:
        pipeline = RuBERTPipeline(
            model_name="DeepPavlov/rubert-base-cased",
            model_path=model_path_str,
            num_labels=3,
            max_length=128,
        )
        return pipeline
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

def get_sentiment_color(label):
    """Return color for sentiment class"""
    colors = {
        "positive": "#4CAF50",
        "negative": "#F44336",
        "neutral": "#9E9E9E"
    }
    return colors.get(label, "#666")

def highlight_words(text, word_scores, tokenizer):
    """Generate HTML with highlighted words - brighter colors"""
    tokens = tokenizer(text)
    word_to_score = {w: s for w, s in word_scores}
    
    html_parts = []
    for token in tokens:
        score = word_to_score.get(token, 0)
        if score > 0:
            intensity = min(0.9, abs(score) * 2.5)
            html_parts.append(
                f'<span class="word-span" style="background-color: rgba(76, 175, 80, {intensity:.2f}); '
                f'font-weight: 600; color: #1a1a1a;" title="Positive contribution: {score:+.4f}">{token}</span>'
            )
        elif score < 0:
            intensity = min(0.85, abs(score) * 2.5)
            html_parts.append(
                f'<span class="word-span" style="background-color: rgba(244, 67, 54, {intensity:.2f}); '
                f'font-weight: 600; color: white;" title="Negative contribution: {score:+.4f}">{token}</span>'
            )
        else:
            html_parts.append(
                f'<span class="word-span" style="background-color: #e0e0e0; color: #555;" '
                f'title="Neutral contribution">{token}</span>'
            )
    
    return " ".join(html_parts)

def plot_comparison_bar_chart(vanilla_exp, enhanced_exp, num_features):
    """Create side-by-side bar chart comparison"""
    vanilla_dict = {w: s for w, s in vanilla_exp[:num_features]}
    enhanced_dict = {w: s for w, s in enhanced_exp[:num_features]}
    
    all_words = list(set(vanilla_dict.keys()) | set(enhanced_dict.keys()))
    all_words.sort(key=lambda x: abs(vanilla_dict.get(x, 0)) + abs(enhanced_dict.get(x, 0)), reverse=True)
    all_words = all_words[:10]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Vanilla LIME", "Enhanced LIME"),
        shared_yaxes=True,
        horizontal_spacing=0.12
    )
    
    # Vanilla LIME
    vanilla_scores = [vanilla_dict.get(w, 0) for w in all_words]
    vanilla_colors = ['#4CAF50' if s > 0 else '#F44336' for s in vanilla_scores]
    
    fig.add_trace(
        go.Bar(
            x=vanilla_scores,
            y=all_words,
            orientation='h',
            marker_color=vanilla_colors,
            text=[f'{s:.4f}' for s in vanilla_scores],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate='<b>%{y}</b><br>Score: %{x:+.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Enhanced LIME
    enhanced_scores = [enhanced_dict.get(w, 0) for w in all_words]
    enhanced_colors = ['#4CAF50' if s > 0 else '#F44336' for s in enhanced_scores]
    
    fig.add_trace(
        go.Bar(
            x=enhanced_scores,
            y=all_words,
            orientation='h',
            marker_color=enhanced_colors,
            text=[f'{s:.4f}' for s in enhanced_scores],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate='<b>%{y}</b><br>Score: %{x:+.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        template="plotly_white",
        font=dict(family="Segoe UI", size=12)
    )
    
    fig.update_xaxes(title_text="Importance Score", row=1, col=1)
    fig.update_xaxes(title_text="Importance Score", row=1, col=2)
    
    return fig

def plot_metrics_radar(vanilla_metrics, enhanced_metrics):
    """Create radar chart for metrics comparison"""
    categories = ['Stability', 'Faithfulness', 'Sparsity', 'Rank Correlation']
    
    vanilla_values = [
        vanilla_metrics.get('stability', 0),
        vanilla_metrics.get('faithfulness', 0),
        vanilla_metrics.get('sparsity', 0),
        vanilla_metrics.get('rank_correlation', 0)
    ]
    
    enhanced_values = [
        enhanced_metrics.get('stability', 0),
        enhanced_metrics.get('faithfulness', 0),
        enhanced_metrics.get('sparsity', 0),
        enhanced_metrics.get('rank_correlation', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=vanilla_values,
        theta=categories,
        fill='toself',
        name='Vanilla LIME',
        line_color='#1a1a2e',
        fillcolor='rgba(26, 26, 46, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=enhanced_values,
        theta=categories,
        fill='toself',
        name='Enhanced LIME',
        line_color='#e94560',
        fillcolor='rgba(233, 69, 96, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        height=450,
        showlegend=True,
        legend=dict(x=0.85, y=1.1),
        template="plotly_white"
    )
    
    return fig

if analyze_button and user_text:
    
    # Load model
    with st.spinner("Loading model..."):
        pipeline = load_model(model_path)
    
    if pipeline is None:
        st.error("Failed to load model. Please verify the model path.")
        st.stop()
    
    # Prediction
    with st.spinner("Processing sentiment..."):
        preds, probs = pipeline.predict([user_text], return_probs=True)
        pred_class = int(preds[0])
        label_names = ["negative", "neutral", "positive"]
        pred_label = label_names[pred_class]
        pred_conf = float(probs[0][pred_class])
    
    # Results display
    st.markdown("---")
    st.markdown("### Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    sentiment_color = get_sentiment_color(pred_label)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Sentiment</h3>
            <h1 style="color: {sentiment_color};">{pred_label.upper()}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-success">
            <h3>Confidence</h3>
            <h1>{pred_conf:.1%}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-warning">
            <h3>Distribution</h3>
            <p>NEG: {probs[0][0]:.0%} | NEU: {probs[0][1]:.0%} | POS: {probs[0][2]:.0%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability chart
    proba_df = pd.DataFrame({
        'Sentiment': ['Negative', 'Neutral', 'Positive'],
        'Probability': [probs[0][0], probs[0][1], probs[0][2]]
    })
    
    fig_proba = px.bar(
        proba_df, 
        x='Sentiment', 
        y='Probability',
        color='Sentiment',
        color_discrete_map={'Negative': '#F44336', 'Neutral': '#9E9E9E', 'Positive': '#4CAF50'},
        title="Sentiment Probability Distribution"
    )
    fig_proba.update_layout(
        height=350,
        showlegend=False,
        template="plotly_white",
        font=dict(family="Segoe UI")
    )
    st.plotly_chart(fig_proba, width='stretch')
    
    # LIME Explanations
    st.markdown("---")
    st.markdown("### LIME Explanations")
    
    info_text = f"Processing {num_samples} perturbations with {n_runs} aggregation runs. Estimated time: 60-180 seconds."
    st.markdown(f'<div class="info-box">{info_text}</div>', unsafe_allow_html=True)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prediction function
    def predict_fn(texts):
        return pipeline.predict_proba(texts)
    
    tokenizer = pipeline.tokenizer.tokenize
    
    # Vanilla LIME
    status_text.text("Vanilla LIME: generating explanations...")
    progress_bar.progress(20)
    
    start_time = time.time()
    
    vanilla_explainer = LimeTextExplainer(
        predict_fn=predict_fn,
        num_samples=num_samples,
        num_features=num_features,
        kernel_width=25.0,
    )
    vanilla_exp = vanilla_explainer.explain_instance(user_text, class_idx=pred_class)
    
    progress_bar.progress(45)
    
    # Enhanced LIME
    status_text.text("Enhanced LIME: generating explanations...")
    
    enhanced_explainer = StabilityEnhancedLIME(
        predict_fn=predict_fn,
        num_samples=num_samples,
        num_features=num_features,
        kernel_width=25.0,
        phrase_max_len=3,
        adjacency_window=2,
        mask_rate=mask_rate,
        propagation_prob=propagation_prob,
        n_runs=n_runs,
    )
    enhanced_detail = enhanced_explainer.explain_instance_detailed(user_text, class_idx=pred_class)
    enhanced_exp = enhanced_detail["aggregated"]
    
    progress_bar.progress(70)
    
    # Metrics computation
    status_text.text("Computing evaluation metrics...")
    
    vanilla_runs = []
    for i in range(min(n_runs, 5)):
        vanilla_explainer.rng = np.random.default_rng(i * 42)
        vanilla_runs.append(vanilla_explainer.explain_instance(user_text, class_idx=pred_class))
    
    vanilla_metrics = {
        "stability": compute_stability_score(vanilla_runs) if len(vanilla_runs) > 1 else 1.0,
        "faithfulness": compute_faithfulness(user_text, tokenizer, predict_fn, vanilla_exp, top_k=5),
        "sparsity": compute_sparsity(vanilla_exp),
        "rank_correlation": compute_rank_correlation(vanilla_runs) if len(vanilla_runs) > 1 else 1.0,
    }
    
    enhanced_metrics = {
        "stability": compute_stability_score(enhanced_detail["per_run"]),
        "faithfulness": compute_faithfulness(user_text, tokenizer, predict_fn, enhanced_exp, top_k=5),
        "sparsity": compute_sparsity(enhanced_exp),
        "rank_correlation": compute_rank_correlation(enhanced_detail["per_run"]),
    }
    
    elapsed_time = time.time() - start_time
    progress_bar.progress(100)
    status_text.text(f"Analysis completed in {elapsed_time:.1f} seconds")
    
    # Metrics display
    st.markdown("---")
    st.markdown("### Quantitative Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Vanilla LIME**")
        m1 = vanilla_metrics
        st.metric("Stability", f"{m1['stability']:.4f}")
        st.metric("Faithfulness", f"{m1['faithfulness']:.4f}")
        st.metric("Sparsity", f"{m1['sparsity']:.4f}")
        st.metric("Rank Correlation", f"{m1['rank_correlation']:.4f}")
    
    with col2:
        st.markdown("**Enhanced LIME**")
        m2 = enhanced_metrics
        delta_stab = m2['stability'] - m1['stability']
        delta_faith = m2['faithfulness'] - m1['faithfulness']
        delta_sparse = m2['sparsity'] - m1['sparsity']
        delta_rank = m2['rank_correlation'] - m1['rank_correlation']
        
        st.metric("Stability", f"{m2['stability']:.4f}", delta=f"{delta_stab:+.4f}")
        st.metric("Faithfulness", f"{m2['faithfulness']:.4f}", delta=f"{delta_faith:+.4f}")
        st.metric("Sparsity", f"{m2['sparsity']:.4f}", delta=f"{delta_sparse:+.4f}")
        st.metric("Rank Correlation", f"{m2['rank_correlation']:.4f}", delta=f"{delta_rank:+.4f}")
    
    # Radar chart
    st.plotly_chart(plot_metrics_radar(vanilla_metrics, enhanced_metrics), width='stretch')
    
    # Word importance charts
    st.markdown("---")
    st.markdown("### Word Importance Comparison")
    st.plotly_chart(plot_comparison_bar_chart(vanilla_exp, enhanced_exp, num_features), width='stretch')
    
    # Highlighted text visualization
    st.markdown("---")
    st.markdown("### Word Highlighting")
    st.caption("Green = Positive contribution | Red = Negative contribution | Gray = Neutral")
    
    tab1, tab2 = st.tabs([" Vanilla LIME", " Enhanced LIME"])
    
    with tab1:
        highlighted_html = highlight_words(user_text, vanilla_exp, tokenizer)
        st.markdown(
            f'<div style="background: #fafafa; padding: 20px; border-radius: 10px; font-size: 1.15em; line-height: 1.8; border: 1px solid #e0e0e0;">'
            f'{highlighted_html}</div>',
            unsafe_allow_html=True
        )
        
        with st.expander("View full word importance table"):
            df_vanilla = pd.DataFrame(vanilla_exp, columns=["Word", "Importance Score"])
            st.dataframe(df_vanilla, width='stretch')
    
    with tab2:
        highlighted_html = highlight_words(user_text, enhanced_exp, tokenizer)
        st.markdown(
            f'<div style="background: #fafafa; padding: 20px; border-radius: 10px; font-size: 1.15em; line-height: 1.8; border: 1px solid #e0e0e0;">'
            f'{highlighted_html}</div>',
            unsafe_allow_html=True
        )
        
        with st.expander("View full word importance table"):
            df_enhanced = pd.DataFrame(enhanced_exp, columns=["Word", "Importance Score"])
            st.dataframe(df_enhanced, width='stretch')
        
        if enhanced_detail.get("stability"):
            with st.expander("View stability analysis (variance across runs)"):
                stability_df = pd.DataFrame([
                    {"Word": w, "Variance": f"{v:.6f}"}
                    for w, v in sorted(enhanced_detail["stability"].items(), key=lambda x: x[1])
                ])
                st.dataframe(stability_df, width='stretch')
                st.caption("Lower variance indicates more stable explanations across multiple runs")
    
    # Download report
    st.markdown("---")
    st.markdown("### Export Report")
    
    # Create HTML report
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>LIME Comparison Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #1a1a2e; border-bottom: 2px solid #1a1a2e; padding-bottom: 10px; }}
            h2 {{ color: #16213e; border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-top: 30px; }}
            h3 {{ color: #333; margin-top: 20px; }}
            .highlight {{ padding: 20px; background: #f5f5f5; border-radius: 8px; margin: 20px 0; font-size: 1.1em; line-height: 1.8; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f0f0f0; font-weight: 600; }}
            .metric-box {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 8px; min-width: 150px; }}
            .metric-value {{ font-size: 1.5em; font-weight: bold; color: #1a1a2e; }}
            .positive {{ background-color: rgba(76, 175, 80, 0.3); }}
            .negative {{ background-color: rgba(244, 67, 54, 0.3); }}
            .footer {{ margin-top: 50px; text-align: center; color: #888; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <h1>LIME Explanation Comparison Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Model:</strong> {model_path}</p>
        
        <h2>Input Text</h2>
        <p><strong>{user_text}</strong></p>
        
        <h2>Prediction Results</h2>
        <p><strong>Predicted Sentiment:</strong> {pred_label.upper()} (confidence: {pred_conf:.2%})</p>
        <p><strong>Probability Distribution:</strong> Negative: {probs[0][0]:.2%} | Neutral: {probs[0][1]:.2%} | Positive: {probs[0][2]:.2%}</p>
        
        <h2>Metrics Comparison</h2>
        <table>
            <tr><th>Metric</th><th>Vanilla LIME</th><th>Enhanced LIME</th><th>Difference</th></tr>
            <tr><td>Stability</dt><td>{vanilla_metrics['stability']:.4f}</td><td>{enhanced_metrics['stability']:.4f}</td><td>{enhanced_metrics['stability']-vanilla_metrics['stability']:+.4f}</td></tr>
            <tr><td>Faithfulness</dt><td>{vanilla_metrics['faithfulness']:.4f}</td><td>{enhanced_metrics['faithfulness']:.4f}</td><td>{enhanced_metrics['faithfulness']-vanilla_metrics['faithfulness']:+.4f}</td></tr>
            <tr><td>Sparsity</dt><td>{vanilla_metrics['sparsity']:.4f}</td><td>{enhanced_metrics['sparsity']:.4f}</td><td>{enhanced_metrics['sparsity']-vanilla_metrics['sparsity']:+.4f}</td></tr>
            <tr><td>Rank Correlation</dt><td>{vanilla_metrics['rank_correlation']:.4f}</td><td>{enhanced_metrics['rank_correlation']:.4f}</td><td>{enhanced_metrics['rank_correlation']-vanilla_metrics['rank_correlation']:+.4f}</td></tr>
        </table>
        
        <h2>Vanilla LIME Explanation</h2>
        <div class="highlight">{highlight_words(user_text, vanilla_exp, tokenizer)}</div>
        
        <h2>Enhanced LIME Explanation</h2>
        <div class="highlight">{highlight_words(user_text, enhanced_exp, tokenizer)}</div>
        
        <h2>Word Importance Tables</h2>
        <h3>Vanilla LIME - Top 15 Words</h3>
        <table><th>Word</th><th>Score</th></tr>
        {"".join(f"<tr><td>{w}</td><td>{s:+.4f}</td></tr>" for w, s in vanilla_exp[:15])}
        </table>
        
        <h3>Enhanced LIME - Top 15 Words</h3>
        <table><th>Word</th><th>Score</th></tr>
        {"".join(f"<tr><td>{w}</td><td>{s:+.4f}</td></tr>" for w, s in enhanced_exp[:15])}
        </table>
        
        <div class="footer">
            <p>Generated by LIME Comparison Tool | Stability-Enhanced LIME for Russian Sentiment Analysis</p>
        </div>
    </body>
    </html>
    """
    
    st.download_button(
        label="Download HTML Report",
        data=report_html,
        file_name=f"lime_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
        width='stretch'
    )
    
    # Completion message
    st.success("Analysis complete")

elif analyze_button and not user_text:
    st.warning("Please enter text for analysis")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; padding: 20px; font-size: 0.8rem;">
        Stability-Enhanced LIME for Russian Sentiment Analysis<br>
        Based on RuBERT transformer model and MINDFUL-LIME structure-aware perturbations
    </div>
    """,
    unsafe_allow_html=True
)