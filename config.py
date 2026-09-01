"""
Configuration file for TruthLens project

This file documents all key settings and hyperparameters used throughout
the project. Modify these values to experiment with different configurations.
"""

# ============================================================================
# GENERAL SETTINGS
# ============================================================================

PROJECT_NAME = "TruthLens"
PROJECT_VERSION = "1.0.0"
RANDOM_SEED = 42
DEVICE = "cuda"  # or "cpu" for CPU-only mode

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

# Minimum character length for text samples
MIN_TEXT_LENGTH = 10

# Stratified split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Sample data generator settings
SAMPLE_DATA_GENERATION = {
    'num_samples': 1000,  # Total samples to generate
    'human_ratio': 0.5,   # Fraction of human-written samples
    'seed': RANDOM_SEED
}

# ============================================================================
# BASELINE MODEL (TF-IDF + Logistic Regression)
# ============================================================================

BASELINE_CONFIG = {
    # TF-IDF Vectorizer settings
    'tfidf': {
        'ngram_range': (1, 2),      # Unigrams and bigrams
        'max_features': 50000,      # Limit vocabulary size
        'sublinear_tf': True,       # Apply sublinear TF scaling
        'max_df': 0.95,             # Ignore very common words
        'min_df': 2,                # Ignore very rare words
    },
    
    # Logistic Regression settings
    'logistic_regression': {
        'max_iter': 1000,           # Maximum iterations
        'random_state': RANDOM_SEED,
        'n_jobs': -1,               # Use all CPU cores
        'class_weight': 'balanced', # Handle class imbalance
    }
}

# ============================================================================
# DISTILBERT MODEL
# ============================================================================

DISTILBERT_CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'num_labels': 2,  # Binary classification
    
    # Training arguments
    'num_epochs': 3,
    'batch_size': 32,  # Reduce to 16 if GPU memory < 4GB
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    
    # Tokenization
    'max_length': 512,  # DistilBERT max sequence length
    
    # Evaluation
    'eval_strategy': 'epoch',
    'save_strategy': 'epoch',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
    
    # Early stopping
    'early_stopping_patience': 2,
}

# ============================================================================
# LIME EXPLAINABILITY
# ============================================================================

LIME_CONFIG = {
    'num_features': 10,     # Number of features to explain
    'class_names': ['Human', 'AI'],
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'calibration_bins': 10,
}

# ============================================================================
# STREAMLIT APP SETTINGS
# ============================================================================

STREAMLIT_CONFIG = {
    'page_title': 'TruthLens - AI Text Detector',
    'layout': 'wide',
    'max_input_length': 10000,  # Maximum characters allowed
    'min_input_length': 10,     # Minimum characters required
}

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

DIRECTORIES = {
    'data_raw': 'data/raw',
    'data_processed': 'data/processed',
    'models': 'models',
    'models_baseline': 'models',  # Models stored here
    'models_distilbert': 'models/distilbert',
    'results': 'results',
    'notebooks': 'notebooks',
    'logs': 'models/logs',
}

# ============================================================================
# FILE PATHS
# ============================================================================

FILE_PATHS = {
    # Input data
    'dataset_raw': 'data/raw/dataset.csv',
    'dataset_train': 'data/processed/train.csv',
    'dataset_val': 'data/processed/val.csv',
    'dataset_test': 'data/processed/test.csv',
    
    # Models - Baseline
    'model_tfidf': 'models/tfidf_vectorizer.pkl',
    'model_baseline': 'models/logistic_regression.pkl',
    
    # Models - DistilBERT
    'model_distilbert': 'models/distilbert',
    
    # Results
    'results_comparison': 'results/model_comparison.csv',
    'failure_analysis_baseline': 'results/failure_analysis_baseline.csv',
    'failure_analysis_distilbert': 'results/failure_analysis_distilbert.csv',
}

# ============================================================================
# LOGGING
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# ============================================================================
# PRODUCTION RECOMMENDATIONS
# ============================================================================

PRODUCTION_NOTES = """
For deployment, consider:

1. Model Optimization:
   - Quantize DistilBERT for faster inference
   - Use ONNX Runtime instead of PyTorch
   - Set batch_size=1 for API calls

2. Scaling:
   - Use model caching to load once
   - Implement request queuing
   - Monitor GPU memory usage

3. Safety:
   - Add input length validation
   - Implement timeout on inference
   - Log predictions for auditing

4. Performance:
   - Baseline: ~1ms per prediction (CPU/GPU)
   - DistilBERT GPU: ~50-100ms
   - DistilBERT CPU: ~500-1000ms

5. Accuracy:
   - Expected: 85-91% depending on dataset
   - Validate on domain-specific text
   - Monitor for distribution shift
"""

if __name__ == "__main__":
    # Print configuration summary
    print("TruthLens Configuration Summary")
    print("=" * 60)
    print(f"Project: {PROJECT_NAME} v{PROJECT_VERSION}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Device: {DEVICE}")
    print()
    print("Data Split: Train/Val/Test = {}/{}/{}".format(
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO))
    print()
    print(f"Baseline Model: TF-IDF + Logistic Regression")
    print(f"  Vocabulary: {BASELINE_CONFIG['tfidf']['max_features']:,} features")
    print()
    print(f"Main Model: {DISTILBERT_CONFIG['model_name']}")
    print(f"  Training epochs: {DISTILBERT_CONFIG['num_epochs']}")
    print(f"  Batch size: {DISTILBERT_CONFIG['batch_size']}")
    print(f"  Learning rate: {DISTILBERT_CONFIG['learning_rate']}")
