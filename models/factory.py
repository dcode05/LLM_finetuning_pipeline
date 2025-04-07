#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factory for creating models and adapters.
"""

import logging
from typing import Dict, Any, Optional

from models.base import BaseModel
from models.hf_model import HuggingFaceModel
from models.adapters.base import BaseAdapter
from models.adapters.lora_adapter import LoRAAdapter
from models.adapters.prompt_tuning_adapter import PromptTuningAdapter
from models.adapters.prefix_tuning_adapter import PrefixTuningAdapter
from models.adapters.qlora_adapter import QLoRAAdapter


def create_model(config: Dict[str, Any]) -> Optional[BaseModel]:
    """
    Create a model based on configuration.
    
    Args:
        config: Configuration dictionary for the model
        
    Returns:
        BaseModel: A model instance
    """
    logger = logging.getLogger(__name__)
    
    # Debug: print config details
    logger.info(f"Creating model with config: {config}")
    logger.info(f"Config keys: {list(config.keys())}")
    
    # Default to HuggingFace model
    model_type = config.get("type", "huggingface")
    
    logger.info(f"Creating model of type: {model_type}")
    
    # Debug: check if name_or_path exists
    model_path = config.get("name_or_path")
    if model_path:
        logger.info(f"Found model_name_or_path: {model_path}")
    else:
        logger.warning("No model name_or_path found in config!")
    
    if model_type == "huggingface":
        model = HuggingFaceModel(config)
        # Debug: check if model was created successfully
        if model:
            logger.info("HuggingFaceModel instance created successfully")
            if model.model is not None:
                logger.info("Model loaded successfully")
            else:
                logger.warning("Model instance created but no model loaded")
        return model
    else:
        logger.warning(f"Unknown model type: {model_type}")
        return None


def create_adapter(config: Dict[str, Any]) -> Optional[BaseAdapter]:
    """
    Create a model adapter based on configuration.
    
    Args:
        config: Configuration dictionary for the adapter
        
    Returns:
        BaseAdapter: An adapter instance
    """
    logger = logging.getLogger(__name__)
    
    adapter_type = config.get("type")
    if not adapter_type:
        logger.warning("No adapter type specified in configuration")
        return None
    
    logger.info(f"Creating adapter of type: {adapter_type}")
    
    if adapter_type == "lora":
        return LoRAAdapter(config)
    elif adapter_type == "prompt_tuning":
        return PromptTuningAdapter(config)
    elif adapter_type == "prefix_tuning":
        return PrefixTuningAdapter(config)
    elif adapter_type == "qlora":
        return QLoRAAdapter(config)
    else:
        logger.warning(f"Unknown adapter type: {adapter_type}")
        return None 