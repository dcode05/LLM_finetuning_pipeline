#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for evaluators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

from datasets import DatasetDict


class BaseEvaluator(ABC):
    """
    Base class for model evaluators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary for the evaluator
        """
        self.config = config
    
    @abstractmethod
    def evaluate(self, model, tokenizer, dataset: Optional[DatasetDict] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            dataset: The dataset to evaluate on
            config: Optional configuration override
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions, references) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        pass 