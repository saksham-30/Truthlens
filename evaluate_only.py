#!/usr/bin/env python3
"""Evaluate saved TruthLens models on integrity-checked processed test data.

This script never generates or overwrites datasets. It refuses evaluation unless
the existing train, validation, and test partitions pass group and class checks.
"""

import os
import sys
from pathlib import Path
import logging

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import validate_split_integrity, validate_training_manifest
from src.train_baseline import load_baseline_model
from src.train_distilbert import load_distilbert_model, predict_distilbert, DEVICE

TEST_PATH = ROOT / 'data' / 'processed' / 'test.csv'
TRAIN_PATH = ROOT / 'data' / 'processed' / 'train.csv'
BASELINE_MANIFEST_PATH = ROOT / 'models' / 'baseline_training_manifest.json'
DISTILBERT_MANIFEST_PATH = ROOT / 'models' / 'distilbert' / 'training_manifest.json'


def _metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'confusion_matrix': cm,
        'per_class': {
            'human': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1': float(f1[0]),
                'support': int(support[0]),
            },
            'ai': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1': float(f1[1]),
                'support': int(support[1]),
            },
        },
    }


def main():
    print('=' * 70)
    print('TruthLens Saved-Artifact Test-Split Evaluation')
    print('=' * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(ROOT / 'data' / 'processed' / 'val.csv')
    test_df = pd.read_csv(TEST_PATH)
    validate_split_integrity(train_df, val_df, test_df)
    validate_training_manifest(str(TRAIN_PATH), str(BASELINE_MANIFEST_PATH), 'tfidf_logistic_regression')
    validate_training_manifest(str(TRAIN_PATH), str(DISTILBERT_MANIFEST_PATH), 'distilbert')
    logger.info(f'Test set size: {len(test_df)}')
    logger.info(f'Label distribution: {test_df["label"].value_counts().to_dict()}')
    print(
        '\nWARNING: Results are diagnostic for the supplied dataset only. '
        'The synthetic demonstration data and saved models do not establish '
        'real-world authorship accuracy or verified model-training provenance.'
    )

    baseline_vectorizer, baseline_model = load_baseline_model()
    X_test = baseline_vectorizer.transform(test_df['text_cleaned'].fillna('').astype(str).values)
    baseline_pred = baseline_model.predict(X_test)
    baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_metrics = _metrics(test_df['label'].to_numpy(), baseline_pred, baseline_proba)

    distilbert_model, distilbert_tokenizer = load_distilbert_model()
    distilbert_pred = []
    distilbert_proba = []
    for text in test_df['text_cleaned'].fillna('').astype(str).tolist():
        result = predict_distilbert(text, distilbert_model, distilbert_tokenizer, DEVICE)
        distilbert_pred.append(result['prediction'])
        distilbert_proba.append(result['ai_prob'])
    distilbert_metrics = _metrics(test_df['label'].to_numpy(), distilbert_pred, distilbert_proba)

    print('\nBaseline (TF-IDF + LR):')
    for key, value in baseline_metrics.items():
        if key in {'confusion_matrix', 'per_class'}:
            continue
        print(f'  {key}: {value:.4f}')
    print('  confusion_matrix: [[TN, FP], [FN, TP]] =', baseline_metrics['confusion_matrix'].tolist())
    print('  per_class:', baseline_metrics['per_class'])

    print('\nDistilBERT:')
    for key, value in distilbert_metrics.items():
        if key in {'confusion_matrix', 'per_class'}:
            continue
        print(f'  {key}: {value:.4f}')
    print('  confusion_matrix: [[TN, FP], [FN, TP]] =', distilbert_metrics['confusion_matrix'].tolist())
    print('  per_class:', distilbert_metrics['per_class'])

    print('\nSame test examples were used for both models; no model was retrained during this audit.')
    print('=' * 70)


if __name__ == '__main__':
    main()
