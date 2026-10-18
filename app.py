"""
TruthLens - AI-Generated Text Detection System
Streamlit Web Application

A professional web interface for detecting AI-generated text using
machine learning models trained on human and AI-written samples.
"""

import streamlit as st
import torch
import numpy as np
import os
from pathlib import Path

# Import our modules
from src.data_preprocessing import preprocess_text
from src.train_baseline import load_baseline_model, predict_baseline
from src.train_distilbert import load_distilbert_model, predict_distilbert, DEVICE
from src.explain import BaselineExplainer

# Page configuration
st.set_page_config(
    page_title="TruthLens - AI Text Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.1em;
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 20px 0;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load both models and explainer."""
    try:
        # Load baseline model
        tfidf_vec, lr_model = load_baseline_model()
        
        # Load DistilBERT model
        distilbert_model, distilbert_tokenizer = load_distilbert_model()
        
        # Load explainer
        explainer = BaselineExplainer()
        
        return {
            'tfidf': tfidf_vec,
            'lr_model': lr_model,
            'distilbert': distilbert_model,
            'tokenizer': distilbert_tokenizer,
            'explainer': explainer
        }
    except FileNotFoundError:
        return None


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
    """Split long text into overlapping chunks for processing."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if start <= len(text) - overlap:
            start += overlap
    
    return chunks if chunks else [text]


def predict_with_chunking(text: str, model, tokenizer, max_chunk_length: int = 512) -> dict:
    """
    Handle long texts by chunking and aggregating predictions.
    """
    cleaned_text = preprocess_text(text)
    if not cleaned_text:
        raise ValueError("Text is empty after preprocessing.")

    token_count = len(tokenizer.encode(cleaned_text))
    if token_count <= max_chunk_length:
        return predict_distilbert(cleaned_text, model, tokenizer, DEVICE)

    chunks = chunk_text(cleaned_text, chunk_size=max_chunk_length - 50)
    predictions = []

    for chunk in chunks:
        pred = predict_distilbert(chunk, model, tokenizer, DEVICE)
        predictions.append(pred['ai_prob'])

    mean_ai_prob = float(np.mean(predictions))
    human_prob = 1.0 - mean_ai_prob
    decision = 1 if mean_ai_prob > 0.5 else 0

    return {
        'prediction': decision,
        'human_prob': human_prob,
        'ai_prob': mean_ai_prob,
        'chunks_used': len(chunks)
    }


def classify_uncertainty(ai_prob: float, human_prob: float, margin: float = 0.15) -> str:
    """Return a label based on a provisional decision margin.

    This rule is not validation-calibrated on a held-out set. The threshold is a
    conservative heuristic only; a larger validation study is required before
    making any confidence claims.
    """
    if abs(ai_prob - human_prob) < margin:
        return 'Inconclusive'
    return 'AI-Generated' if ai_prob >= human_prob else 'Human-Written'


def main():
    # Header
    st.markdown('<div class="main-header">🔍 TruthLens</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Generated Text Detection System</div>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    
    if models is None:
        st.error("❌ Models not found!")
        st.info("""
        Please train the models first:
        1. Run: `python src/train_baseline.py`
        2. Run: `python src/train_distilbert.py`
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.radio(
            "Select Model:",
            ["DistilBERT (Experimental)", "Baseline (TF-IDF + LR)"],
            help="Experimental classifiers; outputs do not establish authorship"
        )
        
        show_explanation = st.checkbox(
            "Show Feature Importance",
            value=True,
            help="Display important features (baseline only)"
        )
        
        st.divider()
        st.subheader("About")
        st.markdown("""
        **TruthLens** uses machine learning to detect AI-generated text.
        
        - **Baseline**: TF-IDF + Logistic Regression
        - **Main Model**: DistilBERT (Fine-tuned)
        - **Purpose**: Experimental AI-text classification using TF-IDF + Logistic Regression and DistilBERT
        
        **Limitations:**
        - Cannot detect paraphrased AI text
        - May fail on very short texts
        - Vulnerable to out-of-distribution samples
        """)
    
    # Main content
    st.subheader("📝 Paste Your Text")
    
    text_input = st.text_area(
        "Enter the text you want to analyze:",
        height=200,
        placeholder="Paste your text here...",
        help="Minimum 10 characters recommended"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze Text", use_container_width=True):
            if not text_input or len(text_input.strip()) < 10:
                st.error("Please enter at least 10 characters")
            else:
                with st.spinner("Analyzing..."):
                    baseline_result = predict_baseline(text_input, models['tfidf'], models['lr_model'])
                    distilbert_result = predict_with_chunking(text_input, models['distilbert'], models['tokenizer'])

                    if "Baseline" in model_choice:
                        result = baseline_result
                        model_name = "Baseline (TF-IDF + Logistic Regression)"
                    else:
                        result = distilbert_result
                        model_name = "DistilBERT"

                    pred_label = classify_uncertainty(result['ai_prob'], result['human_prob'])
                    if pred_label == 'Inconclusive':
                        result['prediction'] = -1

                    model_disagreement = (
                        baseline_result['prediction'] != distilbert_result['prediction']
                        and abs(baseline_result['ai_prob'] - baseline_result['human_prob']) > 0.05
                        and abs(distilbert_result['ai_prob'] - distilbert_result['human_prob']) > 0.05
                    )

                    st.divider()
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

                    st.subheader("Prediction result")
                    if pred_label == 'Inconclusive':
                        st.warning(f"Result: {pred_label}", icon=":material/warning:")
                    else:
                        st.info(f"Model result: {pred_label}", icon=":material/analytics:")

                    st.caption(f"Selected model: {model_name}")
                    st.progress(result['ai_prob'], text=f"AI score: {result['ai_prob']:.1%}")
                    st.progress(result['human_prob'], text=f"Human score: {result['human_prob']:.1%}")
                    st.caption(
                        "Inconclusive means the AI-vs-human score gap is under 15 percentage points. "
                        "This fixed threshold is a heuristic, not calibrated confidence."
                    )
                    if 'chunks_used' in result:
                        st.markdown(f"**Note**: Text was analyzed using {result['chunks_used']} chunks")
                    if model_disagreement:
                        st.warning("The two models disagree on this text. This is a warning sign that the prediction may be uncertain or out-of-distribution.")

                    
                    # Explanation (only for baseline)
                    if show_explanation and "Baseline" in model_choice:
                        st.divider()
                        st.subheader("💡 Feature Importance Analysis")
                        
                        with st.spinner("Generating explanation..."):
                            try:
                                explanation = models['explainer'].explain_prediction(text_input)
                                
                                col1_exp, col2_exp = st.columns(2)
                                
                                with col1_exp:
                                    st.markdown("**Features Supporting the Prediction:**")
                                    for feature, weight in explanation['supporting_features'][:5]:
                                        st.write(f"• {feature}: +{weight:.4f}")
                                
                                with col2_exp:
                                    if explanation['opposing_features']:
                                        st.markdown("**Features Opposing the Prediction:**")
                                        for feature, weight in explanation['opposing_features'][:5]:
                                            st.write(f"• {feature}: {weight:.4f}")
                                
                            except Exception as e:
                                st.warning(f"Could not generate explanation: {str(e)}")
    
    with col2:
        st.write("")  # Spacing
    
    with col3:
        st.write("")  # Spacing
    
    # Important warning
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Important Disclaimer</strong><br>
    This detector provides a probabilistic prediction and <strong>cannot prove authorship</strong> or definitively show whether a person used AI. The result is not evidence of misconduct on its own. AI detection can fail because of:
    <ul>
    <li>Paraphrasing and editing</li>
    <li>Translation</li>
    <li>Unseen AI models</li>
    <li>Short texts</li>
    <li>Changes in writing style</li>
    </ul>
    <strong>Use this tool as a supportive signal only, not as the sole basis for academic, employment, or disciplinary decisions.</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Information tabs
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📚 About the Models", "🔬 Research Details", "❓ FAQ"])
    
    with tab1:
        st.subheader("Model Details")
        st.markdown("""
        ### Baseline Model
        - **Algorithm**: TF-IDF Vectorizer + Logistic Regression
        - **Features**: Unigrams and bigrams with max 50,000 features
        - **Profile**: Fast, interpretable, low memory
        
        ### DistilBERT Model
        - **Architecture**: DistilBERT (distilled version of BERT)
        - **Training**: Fine-tuned on binary classification task
        - **Profile**: Context-aware architecture; performance is experimental
        - **Trade-off**: Requires more computation
        """)
    
    with tab2:
        st.subheader("Research Methodology")
        st.markdown("""
        ### Dataset
        - Built-in demonstration texts are synthetic and template-generated
        - Labels identify the generator's template category, not verified authorship
        - Deterministic group-stratified 60/20/20 train/validation/test split
        
        ### Evaluation Metrics
        - Accuracy
        - Precision & Recall
        - F1-Score
        - ROC-AUC Score
        - Confusion Matrix
        
        ### Key Finding
        The project demonstrates both the capabilities and limitations of AI detection.
        No model can be 100% reliable without additional context.
        """)
    
    with tab3:
        st.subheader("Frequently Asked Questions")
        
        faq_items = [
            ("Can this prove someone used AI?", 
             "No. This is a probabilistic classifier, not proof. Use it as one data point among others."),
            
            ("What if the text is very long?",
             "Texts longer than 512 tokens are automatically split into chunks and predictions are averaged."),
            
            ("Why are results different between models?",
             "Different models use different features. Both have different strengths and weaknesses."),
            
            ("Can it detect GPT-4, Claude, etc?",
             "It was trained on specific models. Other models may produce different signatures."),
            
            ("How should I interpret its results?",
             "The scores are experimental. The built-in template-generated dataset does not establish real-world accuracy or authorship."),
        ]
        
        for question, answer in faq_items:
            with st.expander(question):
                st.write(answer)


if __name__ == "__main__":
    main()
