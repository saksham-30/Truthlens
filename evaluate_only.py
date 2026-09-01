#!/usr/bin/env python3
"""
Evaluate existing trained models and generate comparison visualizations.
This script runs only Steps 5-6 of the pipeline.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import RANDOM_SEED
from evaluate import full_evaluation_pipeline
from failure_analysis import run_failure_analysis

# Define paths
MODELS_DIR = "models"
RESULTS_DIR = "results"

def main():
    """Run evaluation and failure analysis on trained models."""
    print("=" * 70)
    print("TruthLens Evaluation Pipeline (Steps 5-6)")
    print("=" * 70)
    print()
    
    try:
        # Load preprocessing results
        print("[Loading] Preparing datasets...")
        from src.data_preprocessing import preprocess_pipeline
        preprocess_result = preprocess_pipeline("data/raw/dataset.csv", "data/processed")
        
        # Get baseline results  
        print("[Loading] Training/loading baseline model...")
        from src.train_baseline import train_baseline_model
        baseline_results = train_baseline_model()
        
        # Get DistilBERT results
        print("[Loading] Training/loading DistilBERT model...")
        from src.train_distilbert import train_distilbert_model
        distilbert_results = train_distilbert_model()
        
        # Step 5: Full Evaluation
        print()
        print("[STEP 5/6] Comprehensive Evaluation")
        print("-" * 70)
        print("Generating evaluation metrics and visualizations...")
        print()
        
        full_evaluation_pipeline(baseline_results, distilbert_results)
        print("✓ Evaluation complete")
        print()
        
        # Step 6: Failure Analysis
        print("[STEP 6/6] Failure Analysis")
        print("-" * 70)
        print("Testing models on edge cases...")
        print()
        
        run_failure_analysis()
        print("✓ Failure analysis complete")
        print()
        
        print("=" * 70)
        print("Pipeline Complete! All evaluation results saved to results/")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        logger.exception("Detailed error traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()
