#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for model trainers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

from datasets import DatasetDict


class BaseTrainer(ABC):
    """
    Base class for model trainers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary for the trainer
        """
        self.config = config
    
    @abstractmethod
    def train(self, model, tokenizer, dataset: Optional[DatasetDict] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            model: The model to train
            tokenizer: The tokenizer to use
            dataset: The dataset to train on
            config: Optional configuration override
            
        Returns:
            Dict[str, Any]: Training results
        """
        pass
    
    @abstractmethod
    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model.
        
        Args:
            output_dir: Output directory
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, eval_preds):
        """
        Compute evaluation metrics.
        
        Args:
            eval_preds: Evaluation predictions from the trainer
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        pass 