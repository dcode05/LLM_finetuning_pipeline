#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factory for creating data preprocessors.
"""

import logging
from typing import Dict, Any, Optional

from data.preprocessing.base import BasePreprocessor
from data.preprocessing.text_preprocessor import TextPreprocessor


def create_preprocessor(config: Dict[str, Any]) -> Optional[BasePreprocessor]:
    """
    Create a preprocessor based on configuration.
    
    Args:
        config: Configuration dictionary for preprocessing
        
    Returns:
        BasePreprocessor: A preprocessor instance
    """
    logger = logging.getLogger(__name__)
    
    # Skip preprocessing if not enabled
    if not config.get("enabled", True):
        logger.info("Preprocessing is disabled in configuration")
        return None
    
    # Default to text preprocessor
    preprocessor_type = config.get("type", "text")
    
    logger.info(f"Creating preprocessor of type: {preprocessor_type}")
    
    if preprocessor_type == "text":
        return TextPreprocessor(config)
    else:
        logger.warning(f"Unknown preprocessor type: {preprocessor_type}")
        return None 