#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Factory for creating hyperparameter optimization instances.
"""

import logging
from typing import Dict, Any, Optional

from hpo.base import BaseHPO
from hpo.grid_search_hpo import GridSearchHPO


def create_hpo_optimizer(config: Dict[str, Any]) -> Optional[BaseHPO]:
    """
    Create an HPO optimizer based on configuration.
    
    Args:
        config: Configuration dictionary for HPO
        
    Returns:
        BaseHPO: An HPO optimizer instance
    """
    logger = logging.getLogger(__name__)
    
    # Skip HPO if not enabled
    if not config.get("enabled", False):
        logger.info("HPO is disabled in configuration")
        return None
    
    # Get HPO type
    hpo_type = config.get("type", "grid").lower()
    logger.info(f"Creating HPO optimizer of type: {hpo_type}")
    
    if hpo_type == "grid":
        return GridSearchHPO(config)
    elif hpo_type == "ray_tune":
        try:
            from hpo.ray_tune_hpo import RayTuneHPO
            return RayTuneHPO(config)
        except ImportError as e:
            logger.warning(f"Could not import RayTuneHPO: {e}. Falling back to GridSearchHPO.")
            return GridSearchHPO(config)
    else:
        logger.warning(f"Unknown HPO type: {hpo_type}. Falling back to GridSearchHPO.")
        return GridSearchHPO(config) 