#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prompt Tuning adapter implementation.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType

from models.adapters.base import BaseAdapter


class PromptTuningAdapter(BaseAdapter):
    """
    Adapter for Prompt Tuning finetuning method.
    
    Prompt Tuning prepends trainable virtual tokens to the input.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Prompt Tuning adapter.
        
        Args:
            config: Configuration dictionary for the adapter
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Extract Prompt Tuning specific parameters
        self.prompt_tuning_config = self.config.get("prompt_tuning", {})
        
        # Set default values if not provided
        self.num_virtual_tokens = self.prompt_tuning_config.get("num_virtual_tokens", 20)
        self.prompt_tuning_init = self.prompt_tuning_config.get("init_from_vocab", True)
        
        # Convert string init to enum
        if isinstance(self.prompt_tuning_init, str):
            if self.prompt_tuning_init.lower() == "text":
                self.prompt_tuning_init = PromptTuningInit.TEXT
            elif self.prompt_tuning_init.lower() == "random":
                self.prompt_tuning_init = PromptTuningInit.RANDOM
            else:
                self.prompt_tuning_init = PromptTuningInit.RANDOM
                self.logger.warning(f"Unknown prompt tuning init: {self.prompt_tuning_init}. Using RANDOM as default.")
        elif isinstance(self.prompt_tuning_init, bool) and self.prompt_tuning_init:
            self.prompt_tuning_init = PromptTuningInit.TEXT
        else:
            self.prompt_tuning_init = PromptTuningInit.RANDOM
        
        # Get task type
        self.task_type_str = self.prompt_tuning_config.get("task_type", "CAUSAL_LM")
        try:
            self.task_type = getattr(TaskType, self.task_type_str)
        except AttributeError:
            self.logger.warning(f"Unknown task type: {self.task_type_str}. Using CAUSAL_LM as default.")
            self.task_type = TaskType.CAUSAL_LM
    
    def adapt_model(self, model):
        """
        Apply Prompt Tuning adapter to the model.
        
        Args:
            model: The model to adapt
            
        Returns:
            The adapted model with Prompt Tuning
        """
        self.logger.info("Applying Prompt Tuning adapter to model")
        self.logger.info(f"Prompt Tuning config: num_virtual_tokens={self.num_virtual_tokens}, "
                        f"init={self.prompt_tuning_init}")
        
        try:
            # Create Prompt Tuning config
            peft_config = PromptTuningConfig(
                task_type=self.task_type,
                num_virtual_tokens=self.num_virtual_tokens,
                prompt_tuning_init=self.prompt_tuning_init,
                tokenizer_name_or_path=model.config._name_or_path,
            )
            
            # Apply Prompt Tuning adapter
            model = get_peft_model(model, peft_config)
            self.logger.info("Prompt Tuning adapter applied successfully")
            
            # Log trainable parameters
            model.print_trainable_parameters()
            
            return model
        except Exception as e:
            self.logger.error(f"Error applying Prompt Tuning adapter: {str(e)}")
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