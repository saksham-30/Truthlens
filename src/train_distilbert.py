"""
DistilBERT fine-tuning for AI-generated text detection.

This module fine-tunes a pre-trained DistilBERT model for binary classification:
0 = Human-written text
1 = AI-generated text

Features:
- Automatic GPU/CPU detection
- Proper tokenization with truncation and padding
- Stratified train/val/test split
- Early stopping and best model checkpointing
- Comprehensive evaluation metrics
- Handles long texts through chunking (during inference)
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "distilbert-base-uncased"
MODELS_DIR = "models"
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_device() -> str:
    """Setup and log device information."""
    if torch.cuda.is_available():
        logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("⚠ GPU not available, using CPU")
    
    return DEVICE


def load_data_for_transformers(
    train_csv: str = "data/processed/train.csv",
    val_csv: str = "data/processed/val.csv",
    test_csv: str = "data/processed/test.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed data."""
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df


def tokenize_function(examples: Dict, tokenizer, max_length: int = 512) -> Dict:
    """Tokenize texts with truncation and padding."""
    return tokenizer(
        examples['text_cleaned'],
        padding='max_length',
        truncation=True,
        max_length=max_length
    )


def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer
) -> DatasetDict:
    """Convert DataFrames to HuggingFace Datasets and tokenize."""
    logger.info("Preparing datasets...")
    
    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_dict({
        'text_cleaned': train_df['text_cleaned'].values,
        'label': train_df['label'].values
    })
    
    val_dataset = Dataset.from_dict({
        'text_cleaned': val_df['text_cleaned'].values,
        'label': val_df['label'].values
    })
    
    test_dataset = Dataset.from_dict({
        'text_cleaned': test_df['text_cleaned'].values,
        'label': test_df['label'].values
    })
    
    # Tokenize
    logger.info("Tokenizing texts...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # Remove original text column (keep only tokenized)
    train_dataset = train_dataset.remove_columns(['text_cleaned'])
    val_dataset = val_dataset.remove_columns(['text_cleaned'])
    test_dataset = test_dataset.remove_columns(['text_cleaned'])
    
    # Set format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    logger.info("✓ Datasets prepared")
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })


def compute_metrics(eval_pred: Tuple) -> Dict:
    """Compute evaluation metrics for training."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0)
    }


def train_distilbert_model(
    train_csv: str = "data/processed/train.csv",
    val_csv: str = "data/processed/val.csv",
    test_csv: str = "data/processed/test.csv",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5
) -> Dict:
    """
    Fine-tune DistilBERT for AI-generated text detection.
    
    Args:
        train_csv: Path to training data
        val_csv: Path to validation data
        test_csv: Path to test data
        num_epochs: Number of training epochs (default: 3)
        batch_size: Batch size (default: 32, adjust based on GPU memory)
        learning_rate: Learning rate (default: 2e-5)
    
    Returns:
        Dictionary with model, tokenizer, and evaluation metrics
    """
    
    logger.info("=" * 60)
    logger.info("FINE-TUNING DISTILBERT MODEL")
    logger.info("=" * 60)
    
    device = setup_device()
    
    # Load data
    logger.info("\nLoading data...")
    train_df, val_df, test_df = load_data_for_transformers(train_csv, val_csv, test_csv)
    
    # Load tokenizer and model
    logger.info(f"\nLoading {MODEL_NAME} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # Binary classification: Human (0) vs AI (1)
    )
    model.to(device)
    logger.info(f"✓ Model loaded and moved to {device}")
    
    # Prepare datasets
    datasets = prepare_datasets(train_df, val_df, test_df, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{MODELS_DIR}/distilbert-checkpoint",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=RANDOM_SEED,
        logging_steps=50,
        disable_tqdm=False
    )
    
    # Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    logger.info("\nStarting training...")
    train_result = trainer.train()
    logger.info(f"✓ Training complete")
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)
    
    test_results = trainer.evaluate(eval_dataset=datasets['test'], metric_key_prefix="test")
    
    # Get predictions for full analysis
    predictions = trainer.predict(datasets['test'])
    y_test_pred = np.argmax(predictions.predictions, axis=1)
    y_test_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    y_test = datasets['test']['label']
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred)
    }
    
    _print_metrics(test_metrics)
    
    # Save model and tokenizer
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL AND TOKENIZER")
    logger.info("=" * 60)
    
    os.makedirs(f"{MODELS_DIR}/distilbert", exist_ok=True)
    model.save_pretrained(f"{MODELS_DIR}/distilbert")
    tokenizer.save_pretrained(f"{MODELS_DIR}/distilbert")
    logger.info(f"✓ Model saved to {MODELS_DIR}/distilbert")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'trainer': trainer,
        'test_metrics': test_metrics,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }


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


def load_distilbert_model(model_dir: str = f"{MODELS_DIR}/distilbert"):
    """Load a trained DistilBERT model and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def predict_distilbert(text: str, model, tokenizer, device: str = DEVICE) -> Dict:
    """
    Make prediction using DistilBERT model.
    
    Args:
        text: Input text to classify    
        model: Trained DistilBERT model
        tokenizer: DistilBERT tokenizer
        device: Device to run inference on
    
    Returns:
        Dictionary with prediction and probabilities
    """
    model.to(device)
    model.eval()
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = np.argmax(proba)
    
    return {
        'prediction': int(pred),
        'human_prob': float(proba[0]),
        'ai_prob': float(proba[1])
    }


if __name__ == "__main__":
    result = train_distilbert_model()
    print("\n✓ DistilBERT model training complete!")
