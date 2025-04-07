#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factory for creating dataset loaders.
"""

import logging
from typing import Dict, Any, Optional

from data.dataset.base import BaseDatasetLoader
from data.dataset.hf_dataset_loader import HuggingFaceDatasetLoader
from data.dataset.local_dataset_loader import LocalDatasetLoader


def create_dataset_loader(config: Dict[str, Any]) -> Optional[BaseDatasetLoader]:
    """
    Create a dataset loader based on configuration.
    
    Args:
        config: Configuration dictionary for dataset loading
        
    Returns:
        BaseDatasetLoader: A dataset loader instance
    """
    logger = logging.getLogger(__name__)
    
    # Determine which loader to use based on configuration
    if config.get("load_from_hub", False):
        logger.info(f"Creating HuggingFace dataset loader for {config.get('hf_dataset_name', 'unknown')}")
        return HuggingFaceDatasetLoader(config)
    elif config.get("load_from_disk", False):
        logger.info(f"Creating local dataset loader")
        return LocalDatasetLoader(config)
    else:
        logger.warning("No valid dataset loading method specified in configuration")
        return None 