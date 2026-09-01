"""
Baseline model: TF-IDF + Logistic Regression

This module trains a simple but effective baseline model for AI-generated text detection.
The model serves as a performance comparison point for the transformer-based model.

Training pipeline:
1. Load preprocessed training data
2. Vectorize text using TF-IDF with bigrams
3. Train Logistic Regression classifier
4. Evaluate on validation and test sets
5. Save model and vectorizer for later use
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RANDOM_SEED = 42
MODELS_DIR = "models"


def train_baseline_model(
    train_csv: str = "data/processed/train.csv",
    val_csv: str = "data/processed/val.csv",
    test_csv: str = "data/processed/test.csv"
) -> Dict:
    """
    Train TF-IDF + Logistic Regression baseline model.
    
    Args:
        train_csv: Path to training data
        val_csv: Path to validation data
        test_csv: Path to test data
    
    Returns:
        Dictionary with model, vectorizer, and evaluation metrics
    """
    
    logger.info("=" * 60)
    logger.info("TRAINING BASELINE MODEL (TF-IDF + Logistic Regression)")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading training data from {train_csv}")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Extract texts and labels
    X_train = train_df['text_cleaned'].values
    y_train = train_df['label'].values
    
    X_val = val_df['text_cleaned'].values
    y_val = val_df['label'].values
    
    X_test = test_df['text_cleaned'].values
    y_test = test_df['label'].values
    
    # TF-IDF Vectorization
    logger.info("\nVectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_features=50000,
        sublinear_tf=True,
        max_df=0.95,  # Ignore very common words
        min_df=2      # Ignore very rare words
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    logger.info(f"TF-IDF vocabulary size: {len(vectorizer.get_feature_names_out())}")
    logger.info(f"Training matrix shape: {X_train_tfidf.shape}")
    
    # Train Logistic Regression
    logger.info("\nTraining Logistic Regression classifier...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    lr_model.fit(X_train_tfidf, y_train)
    logger.info("✓ Model training complete")
    
    # Evaluation
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON VALIDATION SET")
    logger.info("=" * 60)
    
    X_val_tfidf = vectorizer.transform(X_val)
    y_val_pred = lr_model.predict(X_val_tfidf)
    y_val_proba = lr_model.predict_proba(X_val_tfidf)[:, 1]
    
    val_metrics = _evaluate_model(y_val, y_val_pred, y_val_proba)
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)
    
    X_test_tfidf = vectorizer.transform(X_test)
    y_test_pred = lr_model.predict(X_test_tfidf)
    y_test_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]
    
    test_metrics = _evaluate_model(y_test, y_test_pred, y_test_proba)
    
    # Save model and vectorizer
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL AND VECTORIZER")
    logger.info("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    vectorizer_path = f"{MODELS_DIR}/tfidf_vectorizer.pkl"
    model_path = f"{MODELS_DIR}/logistic_regression.pkl"
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info(f"✓ Vectorizer saved to {vectorizer_path}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(lr_model, f)
    logger.info(f"✓ Model saved to {model_path}")
    
    return {
        'model': lr_model,
        'vectorizer': vectorizer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }


def _evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    _print_metrics(metrics)
    return metrics


def _print_metrics(metrics: Dict) -> None:
    """Print evaluation metrics in a readable format."""
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")


def load_baseline_model(vectorizer_path: str = None, model_path: str = None) -> Tuple:
    """
    Load a trained baseline model and vectorizer.
    
    Args:
        vectorizer_path: Path to saved vectorizer (default: models/tfidf_vectorizer.pkl)
        model_path: Path to saved model (default: models/logistic_regression.pkl)
    
    Returns:
        Tuple of (vectorizer, model)
    """
    vectorizer_path = vectorizer_path or f"{MODELS_DIR}/tfidf_vectorizer.pkl"
    model_path = model_path or f"{MODELS_DIR}/logistic_regression.pkl"
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return vectorizer, model


def predict_baseline(text: str, vectorizer, model) -> Dict:
    """
    Make prediction using baseline model.
    
    Args:
        text: Input text to classify
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained Logistic Regression model
    
    Returns:
        Dictionary with prediction and probability
    """
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    return {
        'prediction': pred,
        'human_prob': proba[0],
        'ai_prob': proba[1],
        'top_features': _get_top_features(X, model, vectorizer)
    }


def _get_top_features(X, model, vectorizer, n_features: int = 10) -> Dict:
    """Get top features influencing the prediction."""
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Get indices of top positive (AI) features
    top_ai_indices = np.argsort(coefficients)[-n_features:]
    top_human_indices = np.argsort(coefficients)[:n_features]
    
    top_ai_features = [feature_names[i] for i in top_ai_indices[::-1]]
    top_human_features = [feature_names[i] for i in top_human_indices[::-1]]
    
    return {
        'top_ai_features': top_ai_features,
        'top_human_features': top_human_features
    }


if __name__ == "__main__":
    result = train_baseline_model()
    print("\n✓ Baseline model training complete!")
