#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prefix Tuning adapter implementation.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from peft import PrefixTuningConfig, get_peft_model, TaskType

from models.adapters.base import BaseAdapter


class PrefixTuningAdapter(BaseAdapter):
    """
    Adapter for Prefix Tuning finetuning method.
    
    Prefix Tuning prepends trainable continuous vectors to the keys and values of each attention layer.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Prefix Tuning adapter.
        
        Args:
            config: Configuration dictionary for the adapter
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Extract Prefix Tuning specific parameters
        self.prefix_tuning_config = self.config.get("prefix_tuning", {})
        
        # Set default values if not provided
        self.encoder_prefix_length = self.prefix_tuning_config.get("encoder_prefix_length", 10)
        self.decoder_prefix_length = self.prefix_tuning_config.get("decoder_prefix_length", 10)
        self.num_virtual_tokens = self.prefix_tuning_config.get("num_virtual_tokens", 
                                                               self.encoder_prefix_length)
        
        # Get task type
        self.task_type_str = self.prefix_tuning_config.get("task_type", "CAUSAL_LM")
        try:
            self.task_type = getattr(TaskType, self.task_type_str)
        except AttributeError:
            self.logger.warning(f"Unknown task type: {self.task_type_str}. Using CAUSAL_LM as default.")
            self.task_type = TaskType.CAUSAL_LM
    
    def adapt_model(self, model):
        """
        Apply Prefix Tuning adapter to the model.
        
        Args:
            model: The model to adapt
            
        Returns:
            The adapted model with Prefix Tuning
        """
        self.logger.info("Applying Prefix Tuning adapter to model")
        self.logger.info(f"Prefix Tuning config: num_virtual_tokens={self.num_virtual_tokens}, "
                        f"encoder_prefix_length={self.encoder_prefix_length}, "
                        f"decoder_prefix_length={self.decoder_prefix_length}")
        
        try:
            # Create Prefix Tuning config
            peft_config = PrefixTuningConfig(
                task_type=self.task_type,
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_hidden_size=model.config.hidden_size,
                prefix_projection=True,
            )
            
            # Apply Prefix Tuning adapter
            model = get_peft_model(model, peft_config)
            self.logger.info("Prefix Tuning adapter applied successfully")
            
            # Log trainable parameters
            model.print_trainable_parameters()
            
            return model
        except Exception as e:
            self.logger.error(f"Error applying Prefix Tuning adapter: {str(e)}")
            raise
    
    def get_trainable_parameters(self, model) -> List:
        """
        Get the trainable parameters of the adapted model.
        
        Args:
            model: The adapted model
            
        Returns:
            List of trainable parameters
        """
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        
        return trainable_params 