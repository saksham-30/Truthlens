"""
Data preprocessing pipeline for the TruthLens AI-Generated Text Detector.

This module handles:
- Loading raw CSV data with 'text' and 'label' columns
- Cleaning and normalizing text
- Removing duplicates and empty texts
- Filtering by minimum text length
- Group-stratified train/val/test splitting (60/20/20)
- Preventing data leakage
"""

import os
import math
import re
import hashlib
import json
import unicodedata
from fractions import Fraction
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MIN_TEXT_LENGTH = 10  # Minimum character length for a text sample
RANDOM_SEED = 42
SYNTHETIC_TOPICS = (
    'technology', 'climate change', 'education', 'artificial intelligence',
    'sustainable development', 'healthcare innovation', 'digital transformation',
    'cybersecurity', 'renewable energy', 'data privacy', 'social media',
    'machine learning', 'blockchain', 'quantum computing', 'biotechnology',
)
SYNTHETIC_TEMPLATE_SIGNATURES = frozenset({
    'i think topic is really important like it s something everyone should know about anyway i ve been reading about it and it s pretty fascinating',
    'so topic is actually more complicated than most people realize i didn t know this until recently it makes sense though',
    'honestly i m not sure about topic some people say it s good others disagree i guess it depends on your perspective',
    'the thing about topic is that it s always changing you can t really make permanent conclusions',
    'i learned something new about topic yesterday it blew my mind i definitely recommend looking into it',
    'the comprehensive analysis of topic demonstrates significant implications for contemporary society furthermore empirical evidence suggests that topic necessitates careful examination and strategic implementation',
    'in light of recent developments topic represents a pivotal consideration for stakeholders across multiple sectors the multifaceted dimensions of this phenomenon warrant rigorous investigation',
    'the integration of topic into existing frameworks has precipitated transformative changes across diverse domains consequently comprehensive evaluation is essential for optimal outcomes',
    'scholarly discourse surrounding topic has been substantially enriched by recent empirical findings these developments warrant careful deliberation and continued academic inquiry',
    'the implementation of topic protocols necessitates meticulous attention to procedural frameworks and regulatory compliance mechanisms such considerations are fundamental to ensuring sustainable progress',
})
SOURCE_GROUP_COLUMNS = (
    'source_id', 'source_group_id', 'source_group', 'document_id', 'document_group_id',
    'article_id', 'question_id', 'conversation_id', 'template_id', 'template_group_id',
    'template_key', 'group_id', 'split_group_id',
)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving punctuation."""
    if not isinstance(text, str):
        return ""
    # Replace multiple spaces/newlines with single space
    text = " ".join(text.split())
    return text.strip()


def normalize_text_for_grouping(text: str) -> str:
    """Canonical form used only to detect duplicate and related samples."""
    if not isinstance(text, str):
        return ''
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _template_group_key(text: str):
    normalized = re.sub(r'[^a-z0-9]+', ' ', normalize_text_for_grouping(text))
    for topic in sorted(SYNTHETIC_TOPICS, key=len, reverse=True):
        normalized = normalized.replace(topic, ' topic ')
    signature = ' '.join(normalized.split())
    if signature not in SYNTHETIC_TEMPLATE_SIGNATURES:
        return None
    return 'template:' + signature


def _has_recognized_group(row, columns) -> bool:
    if _template_group_key(str(row['text'])) is not None:
        return True
    return any(
        column in columns and pd.notna(row[column]) and str(row[column]).strip()
        for column in SOURCE_GROUP_COLUMNS
    )


def _build_group_ids(df: pd.DataFrame) -> pd.Series:
    """Join rows related by normalized text, recognized templates, or source IDs."""
    row_count = len(df)
    parents = list(range(row_count))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    first_row_by_key = {}
    group_columns = [column for column in SOURCE_GROUP_COLUMNS if column in df.columns]
    for position, (_, row) in enumerate(df.iterrows()):
        text = str(row['text'])
        keys = [('normalized_text', normalize_text_for_grouping(text))]
        template_key = _template_group_key(text)
        if template_key is not None:
            keys.append(('template', template_key))
        for column in group_columns:
            value = row[column]
            if pd.notna(value) and str(value).strip():
                keys.append((column, str(value).strip().casefold()))

        for key in keys:
            if key in first_row_by_key:
                union(position, first_row_by_key[key])
            else:
                first_row_by_key[key] = position

    roots = [find(position) for position in range(row_count)]
    return pd.Series([f'group-{root:06d}' for root in roots], index=df.index, dtype='object')


def validate_split_integrity(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Raise when labels, duplicate text, or recognized groups cross split boundaries."""
    named_splits = [('train', train_df), ('validation', val_df), ('test', test_df)]
    for name, frame in named_splits:
        if not {'text', 'label'}.issubset(frame.columns):
            raise ValueError(f'{name} split must contain text and label columns.')
        labels = set(pd.to_numeric(frame['label'], errors='coerce').dropna().astype(int))
        if labels != {0, 1}:
            raise ValueError(f'{name} split must contain both labels 0 and 1; found {sorted(labels)}.')

    combined = pd.concat([frame.reset_index(drop=True) for _, frame in named_splits], ignore_index=True)
    if not all(_has_recognized_group(row, combined.columns) for _, row in combined.iterrows()):
        raise ValueError('Cannot validate splits: source/template group metadata is unavailable for one or more rows.')
    normalized = combined['text'].astype(str).map(normalize_text_for_grouping)
    combined_groups = _build_group_ids(combined)
    conflicts = combined.assign(_normalized=normalized).groupby('_normalized')['label'].nunique()
    if (conflicts > 1).any():
        raise ValueError('Identical normalized text has conflicting labels across the splits.')

    starts = {}
    offset = 0
    for name, frame in named_splits:
        starts[name] = (offset, offset + len(frame))
        offset += len(frame)

    for left_index, (left_name, _) in enumerate(named_splits):
        left_start, left_end = starts[left_name]
        left = combined.iloc[left_start:left_end]
        left_exact = set(left['text'].astype(str))
        left_normalized = set(normalized.iloc[left_start:left_end])
        left_groups = set(combined_groups.iloc[left_start:left_end])
        for right_name, _ in named_splits[left_index + 1:]:
            right_start, right_end = starts[right_name]
            right = combined.iloc[right_start:right_end]
            right_exact = set(right['text'].astype(str))
            if left_exact & right_exact:
                raise ValueError(f'Exact duplicate text leaks between {left_name} and {right_name}.')
            right_normalized = set(normalized.iloc[right_start:right_end])
            if left_normalized & right_normalized:
                raise ValueError(f'Normalized duplicate text leaks between {left_name} and {right_name}.')
            right_groups = set(combined_groups.iloc[right_start:right_end])
            if left_groups & right_groups:
                raise ValueError(f'Recognized template/source groups leak between {left_name} and {right_name}.')


def _training_data_manifest(train_csv: str, model_name: str) -> dict:
    train_df = pd.read_csv(train_csv)
    if 'split_group_id' not in train_df.columns:
        raise ValueError('Training data has no validated split_group_id metadata.')
    digest = hashlib.sha256(Path(train_csv).read_bytes()).hexdigest()
    return {
        'model': model_name,
        'training_csv_sha256': digest,
        'training_rows': int(len(train_df)),
        'training_group_ids': sorted(train_df['split_group_id'].astype(str).unique().tolist()),
    }


def write_training_manifest(train_csv: str, manifest_path: str, model_name: str) -> None:
    manifest = _training_data_manifest(train_csv, model_name)
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, indent=2)


def validate_training_manifest(train_csv: str, manifest_path: str, model_name: str) -> None:
    if not os.path.exists(manifest_path):
        raise ValueError(
            f'Model training provenance is unavailable: missing {manifest_path}. '
            'Retrain on the validated grouped split before reporting evaluation metrics.'
        )
    with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
        saved_manifest = json.load(manifest_file)
    current_manifest = _training_data_manifest(train_csv, model_name)
    if saved_manifest != current_manifest:
        raise ValueError(
            f'Model training manifest does not match the current grouped training data: {manifest_path}.'
        )


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

    if 'dataset_type' not in df.columns and len(df) and all(
        _template_group_key(str(text)) is not None for text in df['text']
    ):
        df['dataset_type'] = 'synthetic_demo'

    logger.info(f"Initial dataset size: {len(df)} samples")

    # Remove null values
    df = df.dropna(subset=['text', 'label'])
    logger.info(f"After removing nulls: {len(df)} samples")

    if all(_has_recognized_group(row, df.columns) for _, row in df.iterrows()):
        df['split_group_id'] = _build_group_ids(df).values

    # Reject contradictory labels before removing exact or normalized duplicates.
    df['_normalized_text'] = df['text'].astype(str).map(normalize_text_for_grouping)
    conflicts = df.groupby('_normalized_text')['label'].nunique()
    if (conflicts > 1).any():
        raise ValueError('Identical normalized text has conflicting labels.')
    df = df.drop_duplicates(subset=['_normalized_text'], keep='first').drop(columns=['_normalized_text'])
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
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test with stratification to prevent data leakage.

    Args:
        df: DataFrame to split
        train_ratio: Fraction for training (default 0.60)
        val_ratio: Fraction for validation (default 0.20)
        test_ratio: Fraction for testing (default 0.20)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError('Train, validation, and test ratios must sum to 1.0.')
    ratios = [Fraction(str(ratio)).limit_denominator(1000) for ratio in (train_ratio, val_ratio, test_ratio)]
    n_splits = math.lcm(*(ratio.denominator for ratio in ratios))
    fold_counts = [int(ratio * n_splits) for ratio in ratios]

    if any(count < 1 for count in fold_counts):
        raise ValueError('Grouped split ratios must allocate at least one fold to each partition.')

    if not all(_has_recognized_group(row, df.columns) for _, row in df.iterrows()):
        raise ValueError('Cannot create a grouped split: source/template group metadata is unavailable for one or more rows.')

    group_ids = _build_group_ids(df)
    labels = set(pd.to_numeric(df['label'], errors='coerce').dropna().astype(int))
    if labels != {0, 1}:
        raise ValueError(f'Dataset must contain both labels 0 and 1; found {sorted(labels)}.')
    groups_per_label = {
        label: int(group_ids.loc[pd.to_numeric(df['label']) == label].nunique())
        for label in (0, 1)
    }
    insufficient = {label: count for label, count in groups_per_label.items() if count < n_splits}
    if insufficient:
        raise ValueError(
            f'Cannot create {n_splits} grouped folds with both classes in every partition; '
            f'groups per label: {groups_per_label}.'
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_indices = list(splitter.split(df, df['label'], groups=group_ids))
    if len(fold_indices) != n_splits:
        raise ValueError(f'Expected {n_splits} grouped folds, received {len(fold_indices)}.')

    assignments = np.full(len(df), '', dtype=object)
    test_fold_end = fold_counts[2]
    val_fold_end = test_fold_end + fold_counts[1]
    for fold_index, (_, held_out_indices) in enumerate(fold_indices):
        if fold_index < test_fold_end:
            assignments[held_out_indices] = 'test'
        elif fold_index < val_fold_end:
            assignments[held_out_indices] = 'validation'
        else:
            assignments[held_out_indices] = 'train'

    if (assignments == '').any():
        raise ValueError('Grouped splitting left one or more rows unassigned.')

    train_df = df.iloc[np.flatnonzero(assignments == 'train')].copy()
    val_df = df.iloc[np.flatnonzero(assignments == 'validation')].copy()
    test_df = df.iloc[np.flatnonzero(assignments == 'test')].copy()
    train_df['split_group_id'] = group_ids.iloc[np.flatnonzero(assignments == 'train')].values
    val_df['split_group_id'] = group_ids.iloc[np.flatnonzero(assignments == 'validation')].values
    test_df['split_group_id'] = group_ids.iloc[np.flatnonzero(assignments == 'test')].values

    validate_split_integrity(train_df, val_df, test_df)

    logger.info(f"Train set size: {len(train_df)} samples")
    logger.info(f"Validation set size: {len(val_df)} samples")
    logger.info(f"Test set size: {len(test_df)} samples")

    logger.info("✓ No exact, normalized, or recognized group leakage detected")

    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data/processed"
) -> None:
    """Save processed datasets to CSV files."""
    validate_split_integrity(train_df, val_df, test_df)
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
