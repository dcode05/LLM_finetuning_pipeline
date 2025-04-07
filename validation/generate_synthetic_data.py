#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate synthetic text classification data for testing the LLM finetuning pipeline.
"""

import json
import os
import random
import sys
from typing import List, Dict, Any

def generate_synthetic_reviews(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate synthetic movie reviews with positive/negative sentiment.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        List of dictionaries containing text and label
    """
    # Templates for positive and negative reviews
    positive_templates = [
        "This movie was {adj}! The {noun} was {adj2} and the {noun2} was {adj3}. {reason}",
        "I really enjoyed this {noun}. The {noun2} was {adj} and {reason}",
        "A {adj} film with {adj2} {noun} and {adj3} {noun2}. {reason}",
        "The {noun} was {adj} and the {noun2} was {adj2}. {reason}",
        "This {noun} is {adj}! {reason}"
    ]
    
    negative_templates = [
        "This movie was {adj}. The {noun} was {adj2} and the {noun2} was {adj3}. {reason}",
        "I didn't enjoy this {noun}. The {noun2} was {adj} and {reason}",
        "A {adj} film with {adj2} {noun} and {adj3} {noun2}. {reason}",
        "The {noun} was {adj} and the {noun2} was {adj2}. {reason}",
        "This {noun} is {adj}. {reason}"
    ]
    
    # Word pools for template filling
    positive_adj = ["excellent", "amazing", "brilliant", "fantastic", "wonderful", "great", "outstanding"]
    negative_adj = ["terrible", "awful", "poor", "bad", "disappointing", "mediocre", "boring"]
    nouns = ["acting", "plot", "story", "direction", "cinematography", "character", "performance", "script"]
    reasons = [
        "I would definitely recommend it!",
        "It kept me engaged throughout.",
        "The pacing was perfect.",
        "The character development was superb.",
        "The special effects were impressive.",
        "I couldn't stop watching!",
        "It exceeded my expectations.",
        "I was disappointed throughout.",
        "I wouldn't recommend it.",
        "The pacing was too slow.",
        "The characters were flat.",
        "The special effects were poor.",
        "I struggled to finish it.",
        "It fell short of expectations."
    ]
    
    reviews = []
    
    for _ in range(num_samples):
        # Randomly choose positive or negative
        is_positive = random.random() > 0.5
        template = random.choice(positive_templates if is_positive else negative_templates)
        adj_pool = positive_adj if is_positive else negative_adj
        
        # Fill template with random words
        review = template.format(
            adj=random.choice(adj_pool),
            adj2=random.choice(adj_pool),
            adj3=random.choice(adj_pool),
            noun=random.choice(nouns),
            noun2=random.choice(nouns),
            reason=random.choice(reasons)
        )
        
        reviews.append({
            "text": review,
            "label": 1 if is_positive else 0
        })
    
    return reviews

def save_dataset(reviews: List[Dict[str, Any]], output_dir: str):
    """
    Save the generated dataset to JSON files.
    
    Args:
        reviews: List of review dictionaries
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into train/validation/test
    random.shuffle(reviews)
    train_size = int(0.7 * len(reviews))
    val_size = int(0.15 * len(reviews))
    
    train_data = reviews[:train_size]
    val_data = reviews[train_size:train_size + val_size]
    test_data = reviews[train_size + val_size:]
    
    # Save splits
    splits = {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(split_data)} samples to {output_file}")

def main():
    """Generate synthetic dataset and save it."""
    # Generate 1000 samples
    reviews = generate_synthetic_reviews(1000)
    
    # Add the parent directory to the Python path if needed
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Save to data/synthetic directory 
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synthetic")
    save_dataset(reviews, output_dir)
    
    print("\nDataset generation complete!")
    print(f"Total samples: {len(reviews)}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main() 