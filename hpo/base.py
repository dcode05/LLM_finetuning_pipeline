#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base hyperparameter optimization module.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable


class BaseHPO(ABC):
    """
    Base class for hyperparameter optimization.
    
    This class defines the interface for hyperparameter optimization strategies
    used in the LLM finetuning pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HPO with a configuration.
        
        Args:
            config: Dictionary containing configuration for the HPO
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing {self.__class__.__name__} with config: {config}")
        
        # Track best parameters and score
        self.best_params = None
        self.best_score = None
    
    @abstractmethod
    def optimize(self, objective_fn: Callable[[Dict[str, Any]], float], 
                search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_fn: Function that takes hyperparameters and returns a score to maximize
            search_space: Dictionary defining the search space for hyperparameters
            
        Returns:
            Dictionary containing the best hyperparameters found
        """
        pass
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best hyperparameters found during optimization.
        
        Returns:
            Dictionary containing the best hyperparameters
        """
        pass
    
    @abstractmethod
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization results.
        
        Returns:
            Dictionary containing summary information about the optimization run
        """
        pass 