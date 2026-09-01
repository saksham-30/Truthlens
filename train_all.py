"""
Master Training Script

Orchestrates the entire TruthLens training pipeline:
1. Generate sample data (if needed)
2. Preprocess data
3. Train baseline model
4. Train DistilBERT model
5. Run full evaluation
6. Generate failure analysis

Run this script to train the entire system from scratch.
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run complete training pipeline."""
    
    logger.info("=" * 70)
    logger.info("TruthLens Complete Training Pipeline")
    logger.info("=" * 70)
    
    # Step 1: Check/Create data
    logger.info("\n[STEP 1/6] Data Preparation")
    logger.info("-" * 70)
    
    dataset_path = "data/raw/dataset.csv"
    if not os.path.exists(dataset_path):
        logger.info("Dataset not found. Generating sample data...")
        from src.sample_data_generator import create_sample_dataset
        create_sample_dataset(num_samples=1000, human_ratio=0.5)
    else:
        logger.info(f"✓ Using existing dataset: {dataset_path}")
    
    # Step 2: Preprocess
    logger.info("\n[STEP 2/6] Data Preprocessing")
    logger.info("-" * 70)
    
    from src.data_preprocessing import preprocess_pipeline
    preprocess_result = preprocess_pipeline(dataset_path, "data/processed")
    
    # Step 3: Train baseline
    logger.info("\n[STEP 3/6] Training Baseline Model (TF-IDF + Logistic Regression)")
    logger.info("-" * 70)
    
    from src.train_baseline import train_baseline_model
    baseline_results = train_baseline_model()
    
    # Step 4: Train DistilBERT
    logger.info("\n[STEP 4/6] Training DistilBERT Model")
    logger.info("-" * 70)
    logger.info("This may take 5-20 minutes depending on your GPU...")
    
    from src.train_distilbert import train_distilbert_model
    distilbert_results = train_distilbert_model()
    
    # Step 5: Full evaluation
    logger.info("\n[STEP 5/6] Comprehensive Evaluation")
    logger.info("-" * 70)
    
    from src.evaluate import full_evaluation_pipeline
    full_evaluation_pipeline(baseline_results, distilbert_results)
    
    # Step 6: Failure analysis
    logger.info("\n[STEP 6/6] Failure Case Analysis")
    logger.info("-" * 70)
    
    from src.failure_analysis import run_failure_analysis, analyze_failures
    df_failures = run_failure_analysis('distilbert')
    analyze_failures(df_failures)
    
    # Save failure analysis
    os.makedirs("results", exist_ok=True)
    df_failures.to_csv("results/failure_analysis_distilbert.csv", index=False)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("✓ TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    
    logger.info("\n📊 RESULTS SUMMARY:")
    logger.info("-" * 70)
    
    print(f"\nBaseline Model Performance:")
    print(f"  Accuracy:  {baseline_results['test_metrics']['accuracy']:.4f}")
    print(f"  Precision: {baseline_results['test_metrics']['precision']:.4f}")
    print(f"  Recall:    {baseline_results['test_metrics']['recall']:.4f}")
    print(f"  F1-Score:  {baseline_results['test_metrics']['f1']:.4f}")
    print(f"  ROC-AUC:   {baseline_results['test_metrics']['roc_auc']:.4f}")
    
    print(f"\nDistilBERT Model Performance:")
    print(f"  Accuracy:  {distilbert_results['test_metrics']['accuracy']:.4f}")
    print(f"  Precision: {distilbert_results['test_metrics']['precision']:.4f}")
    print(f"  Recall:    {distilbert_results['test_metrics']['recall']:.4f}")
    print(f"  F1-Score:  {distilbert_results['test_metrics']['f1']:.4f}")
    print(f"  ROC-AUC:   {distilbert_results['test_metrics']['roc_auc']:.4f}")
    
    logger.info("\n📁 Output Files:")
    logger.info("-" * 70)
    logger.info("  Models:")
    logger.info("    • models/tfidf_vectorizer.pkl")
    logger.info("    • models/logistic_regression.pkl")
    logger.info("    • models/distilbert/")
    logger.info("\n  Results:")
    logger.info("    • results/model_comparison.csv")
    logger.info("    • results/failure_analysis_distilbert.csv")
    logger.info("    • results/*.png (visualizations)")
    
    logger.info("\n🚀 NEXT STEPS:")
    logger.info("-" * 70)
    logger.info("  1. Run Streamlit app:")
    logger.info("     streamlit run app.py")
    logger.info("\n  2. View evaluation results:")
    logger.info("     results/model_comparison.csv")
    logger.info("\n  3. Test on your own text:")
    logger.info("     Open http://localhost:8501 in your browser")
    
    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error during training: {str(e)}")
        logger.error("Check the error message above and try again.")
        sys.exit(1)
