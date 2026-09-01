"""
Data preprocessing pipeline for the TruthLens AI-Generated Text Detector.

This module handles:
- Loading raw CSV data with 'text' and 'label' columns
- Cleaning and normalizing text
- Removing duplicates and empty texts
- Filtering by minimum text length
- Stratified train/val/test splitting (70/15/15)
- Preventing data leakage
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MIN_TEXT_LENGTH = 10  # Minimum character length for a text sample
RANDOM_SEED = 42


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving punctuation."""
    if not isinstance(text, str):
        return ""
    # Replace multiple spaces/newlines with single space
    text = " ".join(text.split())
    return text.strip()


def preprocess_text(text: str) -> str:
    """
    Preprocess text for model input.
    
    Steps:
    1. Normalize whitespace
    2. Remove leading/trailing spaces
    
    Note: We preserve punctuation and don't aggressively remove stopwords
    because transformer models need contextual information.
    """
    if not isinstance(text, str):
        return ""
    
    text = normalize_whitespace(text)
    return text


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data and perform initial cleaning.
    
    Expected CSV format:
    text,label
    "Sample text here",0
    "Another sample",1
    
    Args:
        csv_path: Path to CSV file with 'text' and 'label' columns
    
    Returns:
        Cleaned DataFrame with columns: ['text', 'label', 'text_cleaned']
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    
    logger.info(f"Initial dataset size: {len(df)} samples")
    
    # Remove null values
    df = df.dropna(subset=['text', 'label'])
    logger.info(f"After removing nulls: {len(df)} samples")
    
    # Remove duplicates (keeping first occurrence)
    df = df.drop_duplicates(subset=['text'], keep='first')
    logger.info(f"After removing duplicates: {len(df)} samples")
    
    # Preprocess text
    df['text_cleaned'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts after preprocessing
    df = df[df['text_cleaned'].str.len() > 0]
    logger.info(f"After removing empty texts: {len(df)} samples")
    
    # Remove extremely short texts
    df = df[df['text_cleaned'].str.len() >= MIN_TEXT_LENGTH]
    logger.info(f"After removing texts shorter than {MIN_TEXT_LENGTH} chars: {len(df)} samples")
    
    # Validate labels
    unique_labels = df['label'].unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels must be 0 (Human) or 1 (AI). Found: {unique_labels}")
    
    # Check class balance
    class_counts = df['label'].value_counts()
    logger.info(f"Class distribution:\n{class_counts}")
    logger.info(f"Class balance ratio: {class_counts[1] / class_counts[0]:.2%}")
    
    return df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test with stratification to prevent data leakage.
    
    Args:
        df: DataFrame to split
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df['label'],
        random_state=random_seed
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        stratify=temp_df['label'],
        random_state=random_seed
    )
    
    logger.info(f"Train set size: {len(train_df)} samples")
    logger.info(f"Validation set size: {len(val_df)} samples")
    logger.info(f"Test set size: {len(test_df)} samples")
    
    # Verify no leakage
    train_texts = set(train_df['text'].values)
    val_texts = set(val_df['text'].values)
    test_texts = set(test_df['text'].values)
    
    assert len(train_texts & val_texts) == 0, "Train/Val leakage detected!"
    assert len(train_texts & test_texts) == 0, "Train/Test leakage detected!"
    assert len(val_texts & test_texts) == 0, "Val/Test leakage detected!"
    
    logger.info("✓ No data leakage detected")
    
    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed"
) -> None:
    """Save processed datasets to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    logger.info(f"✓ Processed data saved to {output_dir}/")


def preprocess_pipeline(csv_path: str, output_dir: str = "data/processed") -> Dict:
    """
    Complete preprocessing pipeline.
    
    Args:
        csv_path: Path to raw CSV data
        output_dir: Directory to save processed data
    
    Returns:
        Dictionary with train/val/test DataFrames and statistics
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    # Load and clean
    df = load_and_clean_data(csv_path)
    
    # Split
    train_df, val_df, test_df = stratified_split(df)
    
    # Save
    save_processed_data(train_df, val_df, test_df, output_dir)
    
    result = {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'total_samples': len(df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df)
    }
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    # Example usage (requires a CSV file with 'text' and 'label' columns)
    csv_path = "data/raw/dataset.csv"
    
    if os.path.exists(csv_path):
        result = preprocess_pipeline(csv_path)
        print(f"\nProcessing complete! Total samples: {result['total_samples']}")
    else:
        print(f"Please place your dataset at {csv_path}")
        print("Expected CSV format: text,label")
