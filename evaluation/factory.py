#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factory for creating evaluators.
"""

import logging
from typing import Dict, Any, Optional

from evaluation.base import BaseEvaluator
from evaluation.hf_evaluator import HuggingFaceEvaluator


def create_evaluator(config: Dict[str, Any]) -> Optional[BaseEvaluator]:
    """
    Create an evaluator based on configuration.
    
    Args:
        config: Configuration dictionary for evaluation
        
    Returns:
        BaseEvaluator: An evaluator instance
    """
    logger = logging.getLogger(__name__)
    
    # Default to HuggingFace evaluator
    evaluator_type = config.get("type", "huggingface")
    
    logger.info(f"Creating evaluator of type: {evaluator_type}")
    
    if evaluator_type == "huggingface":
        return HuggingFaceEvaluator(config)
    else:
        logger.warning(f"Unknown evaluator type: {evaluator_type}")
        return None 