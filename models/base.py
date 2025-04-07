#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union


class BaseModel(ABC):
    """
    Base class for models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.
        
        Args:
            config: Configuration dictionary for the model
        """
        self.config = config
        self.model = None
    
    @abstractmethod
    def load_model(self, model_name_or_path: Optional[str] = None) -> None:
        """
        Load a model.
        
        Args:
            model_name_or_path: Path or name of the model to load
        """
        pass
    
    @abstractmethod
    def save_model(self, output_dir: str) -> None:
        """
        Save the model.
        
        Args:
            output_dir: Output directory
        """
        pass
    
    def get_model(self):
        """
        Get the underlying model.
        
        Returns:
            The model instance
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Dictionary with model information
        """
        pass 