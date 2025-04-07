#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HuggingFace tokenizer implementation.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Callable

from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from data.tokenization.base import BaseTokenizer


class HuggingFaceTokenizer(BaseTokenizer):
    """
    Tokenizer implementation using HuggingFace's transformers library.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace tokenizer.
        
        Args:
            config: Configuration dictionary for the tokenizer
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Extract column names from config
        self.prompt_column = config.get("prompt_column", "input")
        self.response_column = config.get("response_column", "output")
        self.text_column = config.get("text_column", None)
        
        # Extract tokenization parameters
        self.max_length = config.get("max_length", 1024)
        self.padding = config.get("padding", "max_length")
        self.truncation = config.get("truncation", True)
        self.add_special_tokens = config.get("add_special_tokens", True)
        self.return_tensors = config.get("return_tensors", "pt")
        
        # Check if we want template-based formatting for instruction tuning
        self.use_template = config.get("use_template", False)
        self.template = config.get("template", "{prompt}\n{response}")
        
        # Load tokenizer if model path is provided
        model_name_or_path = config.get("model_name_or_path")
        if model_name_or_path:
            self.load_tokenizer(model_name_or_path)
    
    def load_tokenizer(self, model_name_or_path: Optional[str] = None) -> None:
        """
        Load a HuggingFace tokenizer.
        
        Args:
            model_name_or_path: Path or name of the model/tokenizer to load
        """
        if not model_name_or_path:
            model_name_or_path = self.config.get("model_name_or_path")
            if not model_name_or_path:
                raise ValueError("No model name or path provided")
        
        self.logger.info(f"Loading tokenizer from {model_name_or_path}")
        
        # Additional tokenizer loading arguments
        use_fast = self.config.get("use_fast", True)
        trust_remote_code = self.config.get("trust_remote_code", False)
        revision = self.config.get("revision", "main")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            
            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.logger.info("Pad token is None, setting to EOS token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set padding side (left/right)
            padding_side = self.config.get("padding_side")
            if padding_side:
                self.tokenizer.padding_side = padding_side
            
            self.logger.info(f"Tokenizer loaded successfully. Vocab size: {self.tokenizer.vocab_size}")
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {str(e)}")
            raise
    
    def tokenize(self, datasets: DatasetDict) -> DatasetDict:
        """
        Tokenize the datasets.
        
        Args:
            datasets: Datasets to tokenize
            
        Returns:
            DatasetDict: Tokenized datasets
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        self.logger.info("Tokenizing datasets")
        
        # Determine the tokenization function based on configuration
        if self.text_column:
            # Single text field tokenization
            tokenize_function = self._create_single_text_tokenizer()
            column_to_check = self.text_column
        elif self.use_template:
            # Template-based tokenization
            tokenize_function = self._create_template_tokenizer()
            column_to_check = self.prompt_column
        else:
            # Separate prompt/response tokenization
            tokenize_function = self._create_prompt_response_tokenizer()
            column_to_check = self.prompt_column
        
        # Apply tokenization to each split
        tokenized_datasets = DatasetDict()
        
        for split_name, split_dataset in datasets.items():
            # Verify the required column exists
            if column_to_check not in split_dataset.column_names:
                self.logger.warning(f"Column '{column_to_check}' not found in {split_name} split. Skipping tokenization.")
                tokenized_datasets[split_name] = split_dataset
                continue
            
            # Apply tokenization
            self.logger.info(f"Tokenizing {split_name} split")
            tokenized_datasets[split_name] = split_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=split_dataset.column_names if self.config.get("remove_original_columns", True) else None,
                desc=f"Tokenizing {split_name} split",
            )
            
            # Log example counts
            self.logger.info(f"Split '{split_name}' has {len(tokenized_datasets[split_name])} examples after tokenization")
        
        return tokenized_datasets
    
    def decode(self, token_ids: Union[list, int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            
        Returns:
            str: Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        # Handle different input types
        if isinstance(token_ids, int):
            return self.tokenizer.decode([token_ids])
        else:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def _create_single_text_tokenizer(self) -> Callable:
        """
        Create a tokenization function for a single text column.
        
        Returns:
            Callable: Tokenization function
        """
        def tokenize_function(examples):
            text_column = self.text_column
            texts = examples[text_column]
            
            # Tokenize the text
            tokenized_inputs = self.tokenizer(
                texts,
                padding=self.padding,
                max_length=self.max_length,
                truncation=self.truncation,
                return_tensors=self.return_tensors,
            )
            
            return tokenized_inputs
        
        return tokenize_function
    
    def _create_prompt_response_tokenizer(self) -> Callable:
        """
        Create a tokenization function for separate prompt and response columns.
        
        Returns:
            Callable: Tokenization function
        """
        def tokenize_function(examples):
            prompt_column = self.prompt_column
            response_column = self.response_column
            
            prompts = examples[prompt_column]
            responses = examples[response_column]
            
            # Check for missing responses
            for i, response in enumerate(responses):
                if not response:
                    self.logger.warning(f"Empty response found at index {i}. Using empty string.")
                    responses[i] = ""
            
            # Tokenize prompts and responses
            model_inputs = self.tokenizer(
                prompts,
                padding=self.padding,
                max_length=self.max_length,
                truncation=self.truncation,
                return_tensors=self.return_tensors,
            )
            
            # Tokenize responses
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    responses,
                    padding=self.padding,
                    max_length=self.max_length,
                    truncation=self.truncation,
                    return_tensors=self.return_tensors,
                )
            
            # Set up the labels for training
            model_inputs["labels"] = labels["input_ids"]
            
            return model_inputs
        
        return tokenize_function
    
    def _create_template_tokenizer(self) -> Callable:
        """
        Create a tokenization function that uses a template to format prompts and responses.
        
        Returns:
            Callable: Tokenization function
        """
        def tokenize_function(examples):
            prompt_column = self.prompt_column
            response_column = self.response_column
            
            prompts = examples[prompt_column]
            responses = examples[response_column]
            
            # Format using template
            formatted_texts = [
                self.template.format(prompt=prompt, response=response) 
                for prompt, response in zip(prompts, responses)
            ]
            
            # Tokenize the formatted text
            tokenized_inputs = self.tokenizer(
                formatted_texts,
                padding=self.padding,
                max_length=self.max_length,
                truncation=self.truncation,
                return_tensors=self.return_tensors,
            )
            
            return tokenized_inputs
        
        return tokenize_function 