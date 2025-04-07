#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics utilities for model evaluation.
"""

import logging
from typing import Dict, Any, Optional, Union, Callable

import evaluate
import numpy as np


def load_metric(metric_name: str) -> Any:
    """
    Load a metric from the Hugging Face evaluate library.
    
    Args:
        metric_name: Name of the metric to load
        
    Returns:
        The loaded metric
    """
    logger = logging.getLogger(__name__)
    
    try:
        if metric_name == "accuracy":
            return evaluate.load("accuracy")
        elif metric_name == "f1":
            return evaluate.load("f1")
        elif metric_name == "precision":
            return evaluate.load("precision")
        elif metric_name == "recall":
            return evaluate.load("recall")
        elif metric_name == "rouge":
            return evaluate.load("rouge")
        elif metric_name == "bleu":
            return evaluate.load("bleu")
        elif metric_name == "meteor":
            return evaluate.load("meteor")
        elif metric_name == "bertscore":
            return evaluate.load("bertscore")
        else:
            # Try to load metric directly
            return evaluate.load(metric_name)
    except Exception as e:
        logger.error(f"Failed to load metric {metric_name}: {e}")
        # Return dummy metric that returns 0
        return DummyMetric(metric_name)


class DummyMetric:
    """
    Dummy metric that returns 0 for any input.
    Used as a fallback when a metric fails to load.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def compute(self, **kwargs) -> Dict[str, float]:
        """Return 0 for the metric."""
        return {self.name: 0.0}


class MetricAggregator:
    """
    Aggregator for multiple metrics.
    """
    
    def __init__(self, metrics: Dict[str, Any]):
        """
        Initialize the metric aggregator.
        
        Args:
            metrics: Dictionary of metric names and metric objects
        """
        self.metrics = metrics
    
    def compute(self, predictions, references, **kwargs) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        results = {}
        
        for name, metric in self.metrics.items():
            try:
                metric_result = metric.compute(
                    predictions=predictions,
                    references=references,
                    **kwargs
                )
                
                # Handle different result formats
                if isinstance(metric_result, dict):
                    # Update results with all returned values
                    for k, v in metric_result.items():
                        results[f"{name}_{k}" if k != name else name] = v
                else:
                    # Single value
                    results[name] = metric_result
            except Exception as e:
                logging.warning(f"Failed to compute metric {name}: {e}")
                results[name] = float('nan')
        
        return results 