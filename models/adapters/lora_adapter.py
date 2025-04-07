#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA (Low-Rank Adaptation) adapter implementation.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from peft import LoraConfig, get_peft_model, TaskType

from models.adapters.base import BaseAdapter


class LoRAAdapter(BaseAdapter):
    """
    Adapter for LoRA (Low-Rank Adaptation) finetuning method.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LoRA adapter.
        
        Args:
            config: Configuration dictionary for the adapter
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Extract LoRA-specific parameters
        self.lora_config = self.config.get("lora", {})
        self.target_modules = self.config.get("target_modules", None)
        
        # Set default values if not provided
        self.r = self.lora_config.get("r", 8)
        self.alpha = self.lora_config.get("alpha", 16)
        self.dropout = self.lora_config.get("dropout", 0.05)
        self.bias = self.lora_config.get("bias", "none")
        
        # Get task type
        self.task_type_str = self.lora_config.get("task_type", "CAUSAL_LM")
        try:
            self.task_type = getattr(TaskType, self.task_type_str)
        except AttributeError:
            self.logger.warning(f"Unknown task type: {self.task_type_str}. Using CAUSAL_LM as default.")
            self.task_type = TaskType.CAUSAL_LM
    
    def adapt_model(self, model):
        """
        Apply LoRA adapter to the model.
        
        Args:
            model: The model to adapt
            
        Returns:
            The adapted model with LoRA layers
        """
        self.logger.info("Applying LoRA adapter to model")
        self.logger.info(f"LoRA config: r={self.r}, alpha={self.alpha}, target_modules={self.target_modules}")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type=self.task_type,
        )
        
        try:
            # Apply LoRA adapter
            model = get_peft_model(model, lora_config)
            self.logger.info("LoRA adapter applied successfully")
            
            # Log trainable parameters
            model.print_trainable_parameters()
            
            return model
        except Exception as e:
            self.logger.error(f"Error applying LoRA adapter: {str(e)}")
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