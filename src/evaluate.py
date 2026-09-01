"""
Comprehensive evaluation pipeline for AI-generated text detector.

This module generates:
- Classification reports
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Model comparison charts
- Calibration analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    roc_auc_score,
    brier_score_loss
)
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> str:
    """Generate detailed classification report."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=['Human', 'AI'],
        digits=4
    )
    
    logger.info(f"\n{model_name} Classification Report:")
    logger.info(report)
    return report


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model", save_path: str = None) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Human', 'AI'], 
                yticklabels=['Human', 'AI'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved to {save_path}")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "Model", save_path: str = None) -> float:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved to {save_path}")
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "Model", save_path: str = None) -> float:
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved to {save_path}")
    plt.close()
    
    return pr_auc


def plot_model_comparison(
    results: Dict[str, Dict],
    metrics_to_compare: list = None,
    save_path: str = None
) -> None:
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dict with model names as keys and metric dicts as values
        metrics_to_compare: List of metrics to compare (default: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        save_path: Optional path to save figure
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Prepare data
    data = []
    for model_name, metrics in results.items():
        for metric in metrics_to_compare:
            if metric in metrics:
                data.append({
                    'Model': model_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': metrics[metric]
                })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Metric', y='Score', hue='Model', palette='Set2')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    plt.legend(title='Model', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved to {save_path}")
    plt.close()


def plot_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "Model", n_bins: int = 10, save_path: str = None) -> Dict:
    """
    Plot calibration (reliability) curve.
    
    A well-calibrated model will show predictions close to the diagonal line.
    """
    # Compute calibration curve
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    predicted_means = []
    actual_means = []
    counts = []
    
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i+1])
        if mask.sum() > 0:
            predicted_means.append(y_proba[mask].mean())
            actual_means.append(y_true[mask].mean())
            counts.append(mask.sum())
        else:
            predicted_means.append(np.nan)
            actual_means.append(np.nan)
            counts.append(0)
    
    # Calculate metrics
    brier_score = brier_score_loss(y_true, y_proba)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', lw=2)
    plt.plot(predicted_means, actual_means, 'o-', label=model_name, lw=2, markersize=8)
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Calibration Curve - {model_name}\n(Brier Score: {brier_score:.4f})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved to {save_path}")
    plt.close()
    
    return {'brier_score': brier_score, 'calibration_data': (predicted_means, actual_means)}


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table for multiple models."""
    rows = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in metrics:
                row[metric.title()] = f"{metrics[metric]:.4f}"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("=" * 80)
    logger.info(df.to_string(index=False))
    return df


def full_evaluation_pipeline(
    baseline_results: Dict,
    distilbert_results: Dict,
    output_dir: str = "results"
) -> None:
    """
    Run complete evaluation pipeline comparing both models.
    
    Args:
        baseline_results: Results from baseline model training
        distilbert_results: Results from DistilBERT training
        output_dir: Directory to save evaluation outputs
    """
    
    logger.info("=" * 60)
    logger.info("STARTING COMPREHENSIVE EVALUATION PIPELINE")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test data
    y_test = baseline_results['y_test']
    y_test_baseline_pred = baseline_results['y_test_pred']
    y_test_baseline_proba = baseline_results['y_test_proba']
    
    y_test_distilbert_pred = distilbert_results['y_test_pred']
    y_test_distilbert_proba = distilbert_results['y_test_proba']
    
    # Classification Reports
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE MODEL - CLASSIFICATION REPORT")
    logger.info("=" * 60)
    generate_classification_report(y_test, y_test_baseline_pred, "Baseline (TF-IDF + LR)")
    
    logger.info("\n" + "=" * 60)
    logger.info("DISTILBERT MODEL - CLASSIFICATION REPORT")
    logger.info("=" * 60)
    generate_classification_report(y_test, y_test_distilbert_pred, "DistilBERT")
    
    # Confusion Matrices
    plot_confusion_matrix(y_test, y_test_baseline_pred, "Baseline (TF-IDF + LR)", 
                         f"{output_dir}/cm_baseline.png")
    plot_confusion_matrix(y_test, y_test_distilbert_pred, "DistilBERT", 
                         f"{output_dir}/cm_distilbert.png")
    
    # ROC Curves
    plot_roc_curve(y_test, y_test_baseline_proba, "Baseline (TF-IDF + LR)", 
                  f"{output_dir}/roc_baseline.png")
    plot_roc_curve(y_test, y_test_distilbert_proba, "DistilBERT", 
                  f"{output_dir}/roc_distilbert.png")
    
    # Precision-Recall Curves
    plot_precision_recall_curve(y_test, y_test_baseline_proba, "Baseline (TF-IDF + LR)", 
                               f"{output_dir}/pr_baseline.png")
    plot_precision_recall_curve(y_test, y_test_distilbert_proba, "DistilBERT", 
                               f"{output_dir}/pr_distilbert.png")
    
    # Calibration Curves
    plot_calibration_curve(y_test, y_test_baseline_proba, "Baseline (TF-IDF + LR)", 
                          save_path=f"{output_dir}/calibration_baseline.png")
    plot_calibration_curve(y_test, y_test_distilbert_proba, "DistilBERT", 
                          save_path=f"{output_dir}/calibration_distilbert.png")
    
    # Model Comparison
    results = {
        'Baseline (TF-IDF + LR)': baseline_results['test_metrics'],
        'DistilBERT': distilbert_results['test_metrics']
    }
    
    plot_model_comparison(results, save_path=f"{output_dir}/model_comparison.png")
    comparison_table = create_comparison_table(results)
    comparison_table.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"All results saved to {output_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    print("Evaluation module loaded. Use this with trained model results.")
