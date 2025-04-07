#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factory for creating trainers.
"""

import logging
from typing import Dict, Any, Optional

from training.base import BaseTrainer
from training.hf_trainer import HuggingFaceTrainer


def create_trainer(config: Dict[str, Any]) -> Optional[BaseTrainer]:
    """
    Create a trainer based on configuration.
    
    Args:
        config: Configuration dictionary for training
        
    Returns:
        BaseTrainer: A trainer instance
    """
    logger = logging.getLogger(__name__)
    
    # Default to HuggingFace trainer
    trainer_type = config.get("type", "huggingface")
    
    logger.info(f"Creating trainer of type: {trainer_type}")
    
    if trainer_type == "huggingface":
        return HuggingFaceTrainer(config)
    else:
        logger.warning(f"Unknown trainer type: {trainer_type}")
        return None 