from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_baseline import load_baseline_model
from src.train_distilbert import DEVICE, load_distilbert_model, predict_distilbert

RAW_PATH = ROOT / 'data' / 'raw' / 'dataset.csv'
OUT_DIR = ROOT / 'benchmark'

TOPICS = [
    'technology',
    'climate change',
    'education',
    'artificial intelligence',
    'sustainable development',
    'healthcare innovation',
    'digital transformation',
    'cybersecurity',
    'renewable energy',
    'data privacy',
    'social media',
    'machine learning',
    'blockchain',
    'quantum computing',
    'biotechnology',
]


def template_key(text: str) -> str:
    """Create a group key that keeps template structure together while masking topics."""
    t = text.lower()
    for topic in sorted(TOPICS, key=len, reverse=True):
        t = t.replace(topic.lower(), ' TOPIC ')
    t = re.sub(r'[^a-z0-9]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _per_class_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
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
    }


def _get_valid_test_partition(benchmark: pd.DataFrame) -> pd.DataFrame:
    test_df = benchmark.loc[benchmark['split'] == 'test'].copy()
    test_labels = {int(label) for label in test_df['label'].astype(int).unique()}
    issues = []

    if test_labels != {0, 1}:
        issues.append(
            f"test split must contain both labels 0 and 1; found {sorted(test_labels)}"
        )

    training_path = ROOT / 'data' / 'processed' / 'train.csv'
    if training_path.exists():
        training_df = pd.read_csv(training_path)
        training_groups = set(training_df['text'].astype(str).map(template_key))
        overlapping_groups = sorted(set(test_df['template_key']) & training_groups)
        if overlapping_groups:
            issues.append(
                f"{len(overlapping_groups)} test template groups occur in the configured model training data"
            )
    else:
        issues.append(f"cannot verify model training groups; missing {training_path}")

    if issues:
        raise ValueError('Held-out evaluation refused: ' + '; '.join(issues))
    return test_df


def _evaluate_baseline(df: pd.DataFrame):
    vec, model = load_baseline_model()
    X = vec.transform(df['text'].astype(str).tolist())
    y_true = df['label'].to_numpy()
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'per_class': _per_class_metrics(y_true, y_pred),
    }


def _evaluate_distilbert(df: pd.DataFrame):
    model, tokenizer = load_distilbert_model()
    y_true = df['label'].to_numpy()
    y_pred = []
    y_proba = []
    for text in df['text'].astype(str).tolist():
        result = predict_distilbert(text, model, tokenizer, DEVICE)
        y_pred.append(result['prediction'])
        y_proba.append(result['ai_prob'])
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'per_class': _per_class_metrics(y_true, y_pred),
    }


def build_benchmark() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH)
    raw = raw[['text', 'label']].copy()
    raw['text'] = raw['text'].astype(str)
    raw['template_key'] = raw['text'].map(template_key)

    train_idx, temp_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42).split(raw, groups=raw['template_key'])
    )
    train_df = raw.iloc[train_idx].copy()
    temp_df = raw.iloc[temp_idx].copy()

    val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42).split(temp_df, groups=temp_df['template_key'])
    )
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    benchmark = pd.concat([train_df, val_df, test_df], ignore_index=True)
    benchmark = benchmark[['text', 'label', 'split', 'template_key']]
    return benchmark


def write_summary(benchmark: pd.DataFrame):
    test_df = _get_valid_test_partition(benchmark)
    baseline = _evaluate_baseline(test_df)
    distilbert = _evaluate_distilbert(test_df)

    summary = {
        'random_seed': 42,
        'grouping_method': 'template_key derived from topic-masked text; GroupShuffleSplit with template groups kept together',
        'evaluation': {
            'split': 'test',
            'sample_count': int(len(test_df)),
            'label_counts': test_df['label'].value_counts().sort_index().to_dict(),
            'training_group_overlap': False,
        },
        'notes': (
            'Metrics are computed only on rows assigned to the test split. Evaluation is refused if either class is absent or if test template groups occur in the configured training CSV. '
            'The dataset is synthetic and does not support claims of real-world generalization.'
        ),
        'dataset': {
            'raw_total': int(len(pd.read_csv(RAW_PATH))),
            'unique_texts': int(pd.read_csv(RAW_PATH)['text'].nunique()),
            'duplicate_rows': int(pd.read_csv(RAW_PATH)['text'].duplicated().sum()),
            'train': int((benchmark['split'] == 'train').sum()),
            'val': int((benchmark['split'] == 'val').sum()),
            'test': int((benchmark['split'] == 'test').sum()),
            'label_counts': {
                'train': benchmark[benchmark['split'] == 'train']['label'].value_counts().to_dict(),
                'val': benchmark[benchmark['split'] == 'val']['label'].value_counts().to_dict(),
                'test': benchmark[benchmark['split'] == 'test']['label'].value_counts().to_dict(),
            },
        },
        'baseline': baseline,
        'distilbert': distilbert,
    }
    with open(OUT_DIR / 'benchmark_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    readme = OUT_DIR / 'README.md'
    readme.write_text(
        '# TruthLens benchmark audit\n\n'
        'This directory is separate from the raw dataset and existing processed splits. It stores a reproducible, grouped benchmark built to limit leakage from the synthetic generator.\n\n'
        '## Provenance\n'
        'The raw CSV contains 1000 rows but only 150 unique texts. The same templates are repeated across a narrow topic list, so the benchmark is synthetic and template-driven rather than true source-based.\n\n'
        '## Split method\n'
        'Because no source IDs or template IDs are available, the benchmark groups by a deterministic topic-masked template signature and then applies GroupShuffleSplit with seed 42. The resulting split is group-disjoint, but it is not class-balanced: validation contains only AI-labeled examples and test contains only human-labeled examples. The saved model metrics were calculated across all rows, not on the test split, so they are not held-out test performance.\n\n'
        '## Recommendation\n'
        'This benchmark is not currently suitable for a two-class held-out evaluation or a jury demonstration. The generator creates repeated templates and topic patterns, and the current test partition contains only one class. Use it for dataset and split diagnostics only; do not interpret the saved metrics as test performance.\n'
    )


def main():
    OUT_DIR.mkdir(exist_ok=True)
    benchmark = build_benchmark()
    write_summary(benchmark)
    benchmark.to_csv(OUT_DIR / 'benchmark_split.csv', index=False)
    print(f'benchmark_dir={OUT_DIR}')
    print(f'train={int((benchmark["split"] == "train").sum())}, val={int((benchmark["split"] == "val").sum())}, test={int((benchmark["split"] == "test").sum())}')
    print(f'benchmark_split={OUT_DIR / "benchmark_split.csv"}')
    print(f'benchmark_summary={OUT_DIR / "benchmark_summary.json"}')


if __name__ == '__main__':
    main()
