#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for model adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAdapter(ABC):
    """
    Base class for model adapters.
    
    Adapters provide parameter-efficient finetuning methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration dictionary for the adapter
        """
        self.config = config
    
    @abstractmethod
    def adapt_model(self, model):
        """
        Apply the adapter to a model.
        
        Args:
            model: The model to adapt
            
        Returns:
            The adapted model
        """
        pass
    
    @abstractmethod
    def get_trainable_parameters(self, model):
        """
        Get the trainable parameters of the adapted model.
        
        Args:
            model: The adapted model
            
        Returns:
            List of trainable parameters
        """
        pass 