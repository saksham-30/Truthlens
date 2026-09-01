"""
Failure Case Analysis Script

Tests the detector on various edge cases and challenging scenarios:
1. Original AI-generated text
2. Human-written text
3. Paraphrased AI text
4. Human-edited AI text
5. AI text with spelling mistakes
6. Very short text
7. Very long text
8. Mixed human + AI text
9. Text from different AI models

This analysis is crucial for understanding model limitations.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import our modules
from src.train_baseline import load_baseline_model, predict_baseline
from src.train_distilbert import load_distilbert_model, predict_distilbert, DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test cases with expected labels
TEST_CASES = [
    {
        'category': 'Human - Professional',
        'expected': 0,
        'text': 'The findings of this study suggest that early intervention strategies can significantly '
                'improve long-term outcomes. Our research team examined data from over 500 participants '
                'over a 5-year period. The results were consistent across all demographic groups studied.'
    },
    {
        'category': 'AI - Formal Style',
        'expected': 1,
        'text': 'The comprehensive analysis of contemporary technological advancements demonstrates '
                'significant implications for various sectors. Furthermore, the integration of artificial '
                'intelligence systems has necessitated fundamental reconsideration of existing paradigms. '
                'In conclusion, these developments warrant careful examination and strategic implementation.'
    },
    {
        'category': 'AI - Academic',
        'expected': 1,
        'text': 'This research endeavor seeks to elucidate the multifaceted relationship between '
                'environmental factors and behavioral manifestations. The methodology employed herein '
                'comprises a rigorous quantitative framework designed to facilitate comprehensive analysis. '
                'Notably, the empirical findings corroborate existing theoretical frameworks.'
    },
    {
        'category': 'Human - Casual',
        'expected': 0,
        'text': 'So I was thinking about starting a new project and I have no idea where to begin. '
                'My friend told me to just start small and see what happens. That actually sounds pretty good. '
                'Anyway, I\'ve been researching tools online and there\'s like a ton of options. It\'s overwhelming.'
    },
    {
        'category': 'Human - Short',
        'expected': 0,
        'text': 'I like cats and dogs. Both are great pets.'
    },
    {
        'category': 'AI - Short',
        'expected': 1,
        'text': 'The implementation of advanced methodologies necessitates comprehensive evaluation.'
    },
    {
        'category': 'Human - Very Long',
        'expected': 0,
        'text': '''
        So I went to the grocery store yesterday and it was really busy. I had to wait in line for like 
        twenty minutes just to check out. They only had three registers open, which seemed like not enough 
        for the number of people shopping. Anyway, I managed to find most of what I needed. They were out 
        of the specific brand of milk I usually buy, so I had to get a different one. I hope it tastes okay.
        When I got home, I realized I forgot to buy coffee. I was really annoyed because I specifically 
        wrote it on my list. I guess I'll have to go back tomorrow or just order it online. Online shopping 
        is pretty convenient these days. You can get most things delivered within a day or two. I've been 
        using it more and more instead of going to physical stores. It saves time and it's easier to compare 
        prices between different retailers.
        '''
    },
    {
        'category': 'AI - Very Long',
        'expected': 1,
        'text': '''
        The multifaceted dimensions of contemporary technological advancement necessitate a comprehensive 
        and nuanced examination of underlying paradigmatic shifts. The exponential proliferation of digital 
        infrastructure has precipitated transformative changes across economic, social, and institutional 
        spheres. Furthermore, the integration of artificial intelligence technologies represents a seminal 
        development that warrants meticulous analysis and strategic contemplation.
        
        In light of these developments, it becomes imperative to evaluate the efficacy of existing frameworks 
        in addressing emergent challenges and opportunities. The implementation of robust methodologies serves 
        to facilitate empirical investigation and substantive discourse. Consequently, stakeholders across 
        diverse sectors must engage in deliberate and constructive dialogue to navigate the complex landscape 
        of technological transformation.
        
        Moreover, the implications of these advancements extend beyond mere technical considerations. The 
        societal ramifications necessitate careful deliberation regarding ethical frameworks and policy 
        mechanisms. In conclusion, a holistic approach incorporating multidisciplinary perspectives constitutes 
        a prerequisite for informed decision-making and sustainable progress.
        '''
    },
    {
        'category': 'Mixed - 50/50 Human + AI',
        'expected': 1,  # Harder to predict
        'text': 'So I went to the store and bought some groceries. In today\'s interconnected global economy, '
                'consumer behavior patterns reflect complex interplay of cultural, psychological, and economic factors. '
                'I got milk, eggs, and bread. Furthermore, empirical evidence suggests that purchasing decisions are '
                'increasingly influenced by sustainability considerations and ethical sourcing practices.'
    },
    {
        'category': 'AI - With Mistakes',
        'expected': 1,
        'text': 'The comprehensiv analysys of contemporry technologicl advancments demostrates signifiant '
                'implications for various sectors. Furthermor, the intergation of artifical inteligence systems '
                'has necesitated fundamental reconsideration of existing paradigms.'
    },
]


def run_failure_analysis(model_type: str = 'distilbert') -> pd.DataFrame:
    """
    Run failure case analysis on all test cases.
    
    Args:
        model_type: 'baseline' or 'distilbert'
    
    Returns:
        DataFrame with analysis results
    """
    
    logger.info("=" * 70)
    logger.info("FAILURE CASE ANALYSIS")
    logger.info("=" * 70)
    
    # Load models
    if model_type == 'baseline':
        logger.info("Loading baseline model...")
        tfidf_vec, lr_model = load_baseline_model()
        model = (tfidf_vec, lr_model)
    else:
        logger.info("Loading DistilBERT model...")
        distilbert_model, tokenizer = load_distilbert_model()
        model = (distilbert_model, tokenizer)
    
    # Run predictions
    results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        logger.info(f"\n[{i}/{len(TEST_CASES)}] Testing: {test_case['category']}")
        
        try:
            if model_type == 'baseline':
                tfidf_vec, lr_model = model
                pred_result = predict_baseline(test_case['text'], tfidf_vec, lr_model)
                prediction = pred_result['prediction']
                ai_prob = pred_result['ai_prob']
                human_prob = pred_result['human_prob']
            else:
                distilbert_model, tokenizer = model
                pred_result = predict_distilbert(test_case['text'], distilbert_model, tokenizer, DEVICE)
                prediction = pred_result['prediction']
                ai_prob = pred_result['ai_prob']
                human_prob = pred_result['human_prob']
            
            is_correct = (prediction == test_case['expected'])
            
            result = {
                'Category': test_case['category'],
                'Expected': 'AI' if test_case['expected'] == 1 else 'Human',
                'Predicted': 'AI' if prediction == 1 else 'Human',
                'AI Probability': ai_prob,
                'Human Probability': human_prob,
                'Correct': '✓' if is_correct else '✗',
                'Text Length': len(test_case['text'].split())
            }
            
            results.append(result)
            
            status = '✓' if is_correct else '✗'
            logger.info(f"  {status} Expected: {result['Expected']}, Got: {result['Predicted']} "
                       f"(AI prob: {ai_prob:.1%})")
            
        except Exception as e:
            logger.error(f"  ✗ Error: {str(e)}")
            results.append({
                'Category': test_case['category'],
                'Expected': 'AI' if test_case['expected'] == 1 else 'Human',
                'Predicted': 'ERROR',
                'AI Probability': None,
                'Human Probability': None,
                'Correct': '✗',
                'Text Length': len(test_case['text'].split())
            })
    
    df = pd.DataFrame(results)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FAILURE ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    print(df.to_string(index=False))
    
    accuracy = (df['Correct'] == '✓').sum() / len(df)
    ai_correct = len(df[(df['Expected'] == 'AI') & (df['Correct'] == '✓')])
    human_correct = len(df[(df['Expected'] == 'Human') & (df['Correct'] == '✓')])
    
    logger.info(f"\nOverall Accuracy: {accuracy:.1%}")
    logger.info(f"AI Detection Accuracy: {ai_correct}/{len(df[df['Expected'] == 'AI'])}")
    logger.info(f"Human Detection Accuracy: {human_correct}/{len(df[df['Expected'] == 'Human'])}")
    
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    
    return df


def analyze_failures(df: pd.DataFrame) -> None:
    """Analyze and report on failure cases."""
    
    logger.info("\n" + "=" * 70)
    logger.info("FAILURE CASE INSIGHTS")
    logger.info("=" * 70)
    
    failures = df[df['Correct'] == '✗']
    
    if len(failures) == 0:
        logger.info("✓ All test cases passed!")
        return
    
    logger.info(f"\nModel failed on {len(failures)}/{len(df)} test cases:\n")
    
    for idx, row in failures.iterrows():
        logger.info(f"❌ {row['Category']}")
        logger.info(f"   Expected: {row['Expected']}, Got: {row['Predicted']}")
        logger.info(f"   Confidence: AI={row['AI Probability']:.1%}")
        logger.info()
    
    logger.info("\n" + "-" * 70)
    logger.info("KEY OBSERVATIONS:")
    logger.info("-" * 70)
    
    # Check if short texts are problematic
    short_texts = df[df['Text Length'] < 20]
    if len(short_texts) > 0:
        short_errors = len(short_texts[short_texts['Correct'] == '✗'])
        logger.info(f"Short text performance: {len(short_texts) - short_errors}/{len(short_texts)} correct")
    
    # Check category-specific failures
    for category in df['Category'].unique():
        category_df = df[df['Category'] == category]
        accuracy = (category_df['Correct'] == '✓').sum() / len(category_df)
        logger.info(f"'{category}': {accuracy:.0%} accuracy")


if __name__ == "__main__":
    import sys
    
    model_choice = sys.argv[1] if len(sys.argv) > 1 else 'distilbert'
    
    try:
        df = run_failure_analysis(model_choice)
        analyze_failures(df)
        
        # Save results
        output_path = f"results/failure_analysis_{model_choice}.csv"
        os.makedirs("results", exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Results saved to {output_path}")
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {str(e)}")
        logger.error("Please train models first: python src/train_baseline.py && python src/train_distilbert.py")
