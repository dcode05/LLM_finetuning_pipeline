#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factory for creating tokenizers.
"""

import logging
from typing import Dict, Any, Optional

from data.tokenization.base import BaseTokenizer
from data.tokenization.hf_tokenizer import HuggingFaceTokenizer


def create_tokenizer(config: Dict[str, Any]) -> Optional[BaseTokenizer]:
    """
    Create a tokenizer based on configuration.
    
    Args:
        config: Configuration dictionary for tokenization
        
    Returns:
        BaseTokenizer: A tokenizer instance
    """
    logger = logging.getLogger(__name__)
    
    # Default to HuggingFace tokenizer
    tokenizer_type = config.get("type", "huggingface")
    
    logger.info(f"Creating tokenizer of type: {tokenizer_type}")
    
    if tokenizer_type == "huggingface":
        return HuggingFaceTokenizer(config)
    else:
        logger.warning(f"Unknown tokenizer type: {tokenizer_type}")
        return None 