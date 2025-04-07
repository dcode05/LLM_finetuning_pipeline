#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HuggingFace model implementation for LLM finetuning.
"""

import logging
import os
from typing import Dict, Any, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)

from models.base import BaseModel


class HuggingFaceModel(BaseModel):
    """
    Model implementation using HuggingFace's transformers library.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace model.
        
        Args:
            config: Configuration dictionary for the model
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # The type of model to load
        self.model_type = config.get("model_type", "causal_lm")  # Default to causal language model
        
        # Model loading parameters
        self.dtype_str = config.get("dtype", "auto")
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.load_in_4bit = config.get("load_in_4bit", False)
        self.device_map = config.get("device_map", "auto")
        self.trust_remote_code = config.get("trust_remote_code", False)
        self.revision = config.get("revision", "main")
        
        # Load model if name or path is provided
        model_name_or_path = config.get("name_or_path")
        if model_name_or_path:
            self.load_model(model_name_or_path)
    
    def load_model(self, model_name_or_path: Optional[str] = None) -> None:
        """
        Load a model from HuggingFace Hub or local path.
        
        Args:
            model_name_or_path: Path or name of the model to load
        """
        if not model_name_or_path:
            model_name_or_path = self.config.get("name_or_path")
            if not model_name_or_path:
                raise ValueError("No model name or path provided")
        
        self.logger.info(f"Loading model from {model_name_or_path}")
        
        # Determine torch dtype
        torch_dtype = self._get_torch_dtype()
        self.logger.info(f"Using dtype: {torch_dtype}")
        
        # Set up quantization config if needed
        quantization_config = None
        if self.load_in_4bit or self.load_in_8bit:
            bits = 4 if self.load_in_4bit else 8
            self.logger.info(f"Using {bits}-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        try:
            # Load the model based on type
            if self.model_type.lower() in ["causal_lm", "causal"]:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map=self.device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                    revision=self.revision,
                    quantization_config=quantization_config,
                )
            elif self.model_type.lower() in ["seq2seq_lm", "seq2seq", "encoder-decoder"]:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name_or_path,
                    device_map=self.device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                    revision=self.revision,
                    quantization_config=quantization_config,
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.logger.info(f"Model loaded successfully. Type: {type(self.model).__name__}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the model to a directory.
        
        Args:
            output_dir: Output directory
        """
        if self.model is None:
            raise ValueError("No model to save. Load a model first.")
        
        self.logger.info(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save the model
            self.model.save_pretrained(output_dir)
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Dictionary with model information
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Get model size (number of parameters)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": self.model_type,
            "model_name": self.model.config._name_or_path,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": next(self.model.parameters()).device,
            "dtype": next(self.model.parameters()).dtype,
            "is_quantized": self.load_in_8bit or self.load_in_4bit,
        }
    
    def _get_torch_dtype(self) -> torch.dtype:
        """
        Get the torch dtype based on configuration.
        
        Returns:
            torch.dtype: The torch dtype to use
        """
        if self.dtype_str == "auto":
            # Automatically determine dtype
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            else:
                return torch.float32
        elif self.dtype_str == "float32":
            return torch.float32
        elif self.dtype_str == "float16":
            return torch.float16
        elif self.dtype_str in ["bfloat16", "bf16"]:
            return torch.bfloat16
        else:
            self.logger.warning(f"Unknown dtype: {self.dtype_str}. Using float32.")
            return torch.float32 