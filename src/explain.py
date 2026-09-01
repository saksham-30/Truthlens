"""
LIME-based explainability for the baseline model.

This module provides interpretable explanations for model predictions by identifying
which words/features most influence the decision.

Important: These are model features, NOT proof of authorship.
"""

import lime
import lime.lime_text
import numpy as np
import pickle
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineExplainer:
    """LIME explainer for baseline model predictions."""
    
    def __init__(self, vectorizer_path: str = "models/tfidf_vectorizer.pkl",
                 model_path: str = "models/logistic_regression.pkl"):
        """
        Initialize explainer with trained model and vectorizer.
        
        Args:
            vectorizer_path: Path to saved TF-IDF vectorizer
            model_path: Path to saved Logistic Regression model
        """
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Create LIME explainer
        self.explainer = lime.lime_text.LimeTextExplainer(
            class_names=['Human', 'AI'],
            verbose=True
        )
        
        logger.info("✓ LIME Explainer initialized")
    
    def predict_proba(self, texts: list) -> np.ndarray:
        """Predict probabilities for texts (required by LIME)."""
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_tfidf)
    
    def explain_prediction(self, text: str, num_features: int = 10) -> Dict:
        """
        Explain a single prediction using LIME.
        
        Args:
            text: Text to explain
            num_features: Number of top features to show
        
        Returns:
            Dictionary with explanation data
        """
        # Get prediction
        pred_proba = self.predict_proba([text])[0]
        pred_label = 'AI' if pred_proba[1] > pred_proba[0] else 'Human'
        pred_confidence = max(pred_proba)
        
        # Get LIME explanation
        exp = self.explainer.explain_instance(
            text,
            self.predict_proba,
            num_features=num_features,
            top_labels=1
        )
        
        # Extract feature weights
        # Get the predicted class index (1 for AI, 0 for Human)
        pred_class_idx = 1 if pred_label == 'AI' else 0
        feature_weights = exp.as_list(label=pred_class_idx)
        
        # Separate positive and negative features
        positive_features = [(word, weight) for word, weight in feature_weights if weight > 0]
        negative_features = [(word, weight) for word, weight in feature_weights if weight < 0]
        
        return {
            'text': text,
            'prediction': pred_label,
            'ai_probability': float(pred_proba[1]),
            'human_probability': float(pred_proba[0]),
            'confidence': float(pred_confidence),
            'supporting_features': positive_features,  # Features supporting the prediction
            'opposing_features': negative_features,     # Features opposing the prediction
            'explanation_object': exp
        }
    
    def format_explanation(self, explanation: Dict) -> str:
        """Format explanation for display."""
        output = []
        output.append("=" * 60)
        output.append(f"PREDICTION: {explanation['prediction']}-Generated Text")
        output.append(f"AI Probability: {explanation['ai_probability']:.1%}")
        output.append(f"Human Probability: {explanation['human_probability']:.1%}")
        output.append(f"Confidence: {explanation['confidence']:.1%}")
        output.append("=" * 60)
        
        output.append("\n⚠️  IMPORTANT DISCLAIMER")
        output.append("-" * 60)
        output.append("The features below show what the model considered important.")
        output.append("This is a statistical pattern, NOT proof of authorship.")
        output.append("Individual words do not prove that text was AI-generated.")
        output.append("Limitations: paraphrasing, editing, translation, writing style")
        output.append("            changes, unseen AI models, and fine-tuned variants")
        output.append("            can all affect model predictions.")
        output.append("-" * 60)
        
        output.append(f"\nSupporting Features (favor {explanation['prediction']}):")
        for feature, weight in explanation['supporting_features'][:5]:
            # Clean up LIME output (remove HTML entities if any)
            feature = feature.strip()
            output.append(f"  • {feature}: +{weight:.4f}")
        
        if explanation['opposing_features']:
            output.append(f"\nOpposing Features (favor alternative):")
            for feature, weight in explanation['opposing_features'][:5]:
                feature = feature.strip()
                output.append(f"  • {feature}: {weight:.4f}")
        
        output.append("\n" + "=" * 60)
        return "\n".join(output)


def explain_batch(
    explainer: BaselineExplainer,
    texts: list,
    output_file: str = None
) -> list:
    """
    Explain multiple predictions and optionally save to file.
    
    Args:
        explainer: BaselineExplainer instance
        texts: List of texts to explain
        output_file: Optional file to save explanations
    
    Returns:
        List of explanation dictionaries
    """
    explanations = []
    for i, text in enumerate(texts):
        logger.info(f"Explaining text {i+1}/{len(texts)}...")
        exp = explainer.explain_prediction(text)
        explanations.append(exp)
        logger.info(f"  -> {exp['prediction']} ({exp['ai_probability']:.1%})")
    
    if output_file:
        with open(output_file, 'w') as f:
            for i, exp in enumerate(explanations):
                f.write(f"\n\nTEXT {i+1}:\n")
                f.write(f"{exp['text'][:100]}...\n")
                f.write(explainer.format_explanation(exp))
        logger.info(f"✓ Explanations saved to {output_file}")
    
    return explanations


if __name__ == "__main__":
    # Example usage
    try:
        explainer = BaselineExplainer()
        
        # Test text
        test_text = "The rapid advancement of artificial intelligence has fundamentally " \
                   "transformed various industries, requiring comprehensive analysis and " \
                   "strategic implementation. Furthermore, these technological innovations " \
                   "necessitate careful consideration of ethical implications."
        
        exp = explainer.explain_prediction(test_text)
        print(explainer.format_explanation(exp))
        
    except FileNotFoundError:
        print("Models not found. Train baseline model first.")
