"""
Sample Data Generator

Creates a synthetic dataset for demonstration and testing purposes.
This allows the project to run without requiring a pre-existing dataset.

The generator creates realistic but synthetic human and AI-generated text samples.
"""

import os
import pandas as pd
import random
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class SampleDataGenerator:
    """Generate realistic sample datasets for AI detection."""
    
    def __init__(self):
        """Initialize with text templates."""
        self.human_templates = [
            "I think {} is really important. Like, it's something everyone should know about. "
            "Anyway, I've been reading about it and it's pretty fascinating.",
            
            "So {} is actually more complicated than most people realize. "
            "I didn't know this until recently. It makes sense though.",
            
            "Honestly, I'm not sure about {}. Some people say it's good, others disagree. "
            "I guess it depends on your perspective.",
            
            "The thing about {} is that it's always changing. "
            "You can't really make permanent conclusions.",
            
            "I learned something new about {} yesterday. It blew my mind. "
            "I definitely recommend looking into it.",
        ]
        
        self.ai_templates = [
            "The comprehensive analysis of {} demonstrates significant implications "
            "for contemporary society. Furthermore, empirical evidence suggests that {} "
            "necessitates careful examination and strategic implementation.",
            
            "In light of recent developments, {} represents a pivotal consideration "
            "for stakeholders across multiple sectors. The multifaceted dimensions of this "
            "phenomenon warrant rigorous investigation.",
            
            "The integration of {} into existing frameworks has precipitated transformative "
            "changes across diverse domains. Consequently, comprehensive evaluation is essential "
            "for optimal outcomes.",
            
            "Scholarly discourse surrounding {} has been substantially enriched by recent "
            "empirical findings. These developments warrant careful deliberation and continued "
            "academic inquiry.",
            
            "The implementation of {} protocols necessitates meticulous attention to procedural "
            "frameworks and regulatory compliance mechanisms. Such considerations are fundamental "
            "to ensuring sustainable progress.",
        ]
        
        self.topics = [
            "technology",
            "climate change",
            "education",
            "artificial intelligence",
            "sustainable development",
            "healthcare innovation",
            "digital transformation",
            "cybersecurity",
            "renewable energy",
            "data privacy",
            "social media",
            "machine learning",
            "blockchain",
            "quantum computing",
            "biotechnology"
        ]
    
    def generate_human_text(self) -> str:
        """Generate synthetic human-written text."""
        template = random.choice(self.human_templates)
        topic = random.choice(self.topics)
        return template.format(topic)
    
    def generate_ai_text(self) -> str:
        """Generate synthetic AI-style text."""
        template = random.choice(self.ai_templates)
        topic = random.choice(self.topics)
        return template.format(topic, topic)
    
    def generate_dataset(self, num_samples: int = 1000, human_ratio: float = 0.5) -> pd.DataFrame:
        """
        Generate a balanced dataset.
        
        Args:
            num_samples: Total number of samples to generate
            human_ratio: Fraction of samples that should be human-written (0.5 = 50/50 split)
        
        Returns:
            DataFrame with 'text' and 'label' columns
        """
        logger.info("=" * 60)
        logger.info("GENERATING SAMPLE DATASET")
        logger.info("=" * 60)
        
        num_human = int(num_samples * human_ratio)
        num_ai = num_samples - num_human
        
        logger.info(f"Generating {num_human} human-written samples...")
        human_texts = [self.generate_human_text() for _ in range(num_human)]
        
        logger.info(f"Generating {num_ai} AI-generated samples...")
        ai_texts = [self.generate_ai_text() for _ in range(num_ai)]
        
        # Combine and shuffle
        texts = human_texts + ai_texts
        labels = [0] * num_human + [1] * num_ai
        
        # Shuffle
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        logger.info(f"✓ Generated {len(df)} samples")
        logger.info(f"  Human samples: {(df['label'] == 0).sum()}")
        logger.info(f"  AI samples: {(df['label'] == 1).sum()}")
        
        return df


def create_sample_dataset(
    output_path: str = "data/raw/dataset.csv",
    num_samples: int = 1000,
    human_ratio: float = 0.5
) -> None:
    """
    Create and save sample dataset.
    
    Args:
        output_path: Where to save the CSV file
        num_samples: Number of samples to generate
        human_ratio: Fraction that should be human-written
    """
    
    generator = SampleDataGenerator()
    df = generator.generate_dataset(num_samples, human_ratio)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Dataset saved to {output_path}")
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    # Create sample dataset with 1000 samples (50/50 split)
    create_sample_dataset(num_samples=1000, human_ratio=0.5)
    
    logger.info("\n✓ Sample dataset created successfully!")
    logger.info("You can now run: python src/train_baseline.py")
