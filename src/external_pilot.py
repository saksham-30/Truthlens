"""Fail-closed, candidate-only pipeline for an approved external dataset."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data_preprocessing import (
    normalize_text_for_grouping,
    preprocess_text,
    stratified_split,
    validate_split_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = ROOT / 'models' / 'candidates'
REQUIRED_APPROVAL_FIELDS = (
    'dataset_id',
    'revision',
    'source_url',
    'license',
    'license_review_status',
    'training_use_approved',
    'approval_reference',
    'retrieval_date',
    'dataset_sha256',
    'label_mapping',
)


def validate_approval_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ValueError('Approval manifest must be a JSON object.')
    missing = [field for field in REQUIRED_APPROVAL_FIELDS if not manifest.get(field)]
    if missing:
        raise ValueError(f'Approval manifest is missing required fields: {missing}.')
    if manifest['license_review_status'] != 'approved' or manifest['training_use_approved'] is not True:
        raise ValueError('External training is blocked until license review and training use are explicitly approved.')
    mapping = manifest['label_mapping']
    if not isinstance(mapping, dict) or not mapping.get('0') or not mapping.get('1') or mapping['0'] == mapping['1']:
        raise ValueError("label_mapping must explicitly map distinct source classes to '0' and '1'.")
    digest = str(manifest['dataset_sha256']).lower()
    if len(digest) != 64 or any(character not in '0123456789abcdef' for character in digest):
        raise ValueError('dataset_sha256 must be a 64-character SHA-256 hex digest.')


def validate_external_frame(frame: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    validate_approval_manifest(manifest)
    required_columns = {'text', 'label', 'group_id', 'dataset_type'}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f'External dataset is missing canonical fields: {sorted(missing)}.')
    if frame['dataset_type'].isna().any() or set(frame['dataset_type'].astype(str)) != {'external'}:
        raise ValueError('Only dataset_type=external rows are permitted; synthetic demo data is blocked.')

    result = frame.copy()
    result['label'] = pd.to_numeric(result['label'], errors='coerce')
    if result['label'].isna().any() or set(result['label'].unique()) != {0, 1}:
        raise ValueError('External dataset must have valid labels 0 and 1 with no unknown labels.')
    if result['text'].isna().any() or result['group_id'].isna().any():
        raise ValueError('External text and group_id values must be present for every row.')
    result['text'] = result['text'].astype(str)
    result['group_id'] = result['group_id'].astype(str).str.strip()
    if (result['group_id'] == '').any():
        raise ValueError('External group_id values must not be empty.')

    result['text_cleaned'] = result['text'].map(preprocess_text)
    result = result[result['text_cleaned'].str.len() > 0].copy()
    normalized = result['text'].map(normalize_text_for_grouping)
    conflicting = result.assign(_normalized=normalized).groupby('_normalized')['label'].nunique()
    if (conflicting > 1).any():
        raise ValueError('Normalized duplicate text has conflicting labels.')
    return result


def choose_validation_threshold(y_true, y_probabilities) -> float:
    """Choose the maximum-F1 threshold using validation scores only."""
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_probabilities, dtype=float)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Threshold selection requires both labels in validation data.')
    if (
        len(labels) != len(probabilities)
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError('Validation labels and probabilities must have matching lengths and values in [0, 1].')

    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = [(f1_score(labels, probabilities >= threshold, zero_division=0), threshold) for threshold in candidates]
    best_f1 = max(score for score, _ in scores)
    return float(max(threshold for score, threshold in scores if score == best_f1))


def _test_metrics(y_true, y_probabilities, threshold: float) -> dict:
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    tn, fp, _, _ = matrix.ravel()
    return {
        'sample_count': int(len(labels)),
        'accuracy': float(accuracy_score(labels, predictions)),
        'roc_auc': float(roc_auc_score(labels, probabilities)),
        'pr_auc': float(average_precision_score(labels, probabilities)),
        'false_positive_rate': float(fp / (fp + tn)) if fp + tn else 0.0,
        'threshold_selected_on_validation': float(threshold),
        'confusion_matrix_labels_0_1': matrix.tolist(),
        'per_class': {
            'human_0': {'precision': float(precision[0]), 'recall': float(recall[0]), 'f1': float(f1[0]), 'support': int(support[0])},
            'ai_1': {'precision': float(precision[1]), 'recall': float(recall[1]), 'f1': float(f1[1]), 'support': int(support[1])},
        },
    }


def _candidate_directory(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    allowed_root = CANDIDATE_ROOT.resolve()
    if candidate == allowed_root or allowed_root not in candidate.parents:
        raise ValueError(f'Candidate artifacts must be written under {allowed_root}; production paths are forbidden.')
    if candidate.exists():
        raise FileExistsError(f'Candidate path already exists; refusing to overwrite it: {candidate}')
    return candidate


def run_external_pilot(
    dataset_csv: str | Path,
    approval_manifest_path: str | Path,
    candidate_dir: str | Path,
) -> dict:
    """Train/evaluate a baseline only after approval; save artifacts to a new candidate path."""
    with open(approval_manifest_path, 'r', encoding='utf-8') as manifest_file:
        manifest = json.load(manifest_file)
    validate_approval_manifest(manifest)

    dataset_path = Path(dataset_csv).resolve()
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if dataset_hash != str(manifest['dataset_sha256']).lower():
        raise ValueError('External dataset bytes do not match the approved manifest hash.')
    frame = validate_external_frame(pd.read_csv(dataset_path), manifest)
    split_seed = 42
    train_df, val_df, test_df = stratified_split(
        frame, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_seed=split_seed
    )
    validate_split_integrity(train_df, val_df, test_df)
    candidate_path = _candidate_directory(candidate_dir)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True, max_df=0.95, min_df=2)
    X_train = vectorizer.fit_transform(train_df['text_cleaned'])
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, train_df['label'].astype(int))

    validation_probabilities = model.predict_proba(vectorizer.transform(val_df['text_cleaned']))[:, 1]
    threshold = choose_validation_threshold(val_df['label'], validation_probabilities)
    test_probabilities = model.predict_proba(vectorizer.transform(test_df['text_cleaned']))[:, 1]
    metrics = _test_metrics(test_df['label'], test_probabilities, threshold)

    candidate_path.mkdir(parents=True, exist_ok=False)
    vectorizer_path = candidate_path / 'tfidf_vectorizer.pkl'
    model_path = candidate_path / 'logistic_regression.pkl'
    with open(vectorizer_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    candidate_manifest = {
        **manifest,
        'candidate_model': 'tfidf_logistic_regression',
        'split_method': 'StratifiedGroupKFold',
        'split_seed': split_seed,
        'split_ratios': {'train': 0.60, 'validation': 0.20, 'test': 0.20},
        'training_rows': int(len(train_df)),
        'training_groups': int(train_df['split_group_id'].nunique()),
        'validation_rows': int(len(val_df)),
        'validation_groups': int(val_df['split_group_id'].nunique()),
        'test_rows': int(len(test_df)),
        'test_groups': int(test_df['split_group_id'].nunique()),
        'threshold_selection': 'maximum validation F1; test scores not used for selection',
        'candidate_artifact_sha256': {
            'tfidf_vectorizer.pkl': hashlib.sha256(vectorizer_path.read_bytes()).hexdigest(),
            'logistic_regression.pkl': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        },
        'metrics': metrics,
    }
    with open(candidate_path / 'candidate_manifest.json', 'w', encoding='utf-8') as manifest_file:
        json.dump(candidate_manifest, manifest_file, indent=2)
    return candidate_manifest