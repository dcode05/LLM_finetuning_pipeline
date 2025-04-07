#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for tokenizers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

from datasets import DatasetDict, Dataset


class BaseTokenizer(ABC):
    """
    Base class for tokenizers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tokenizer.
        
        Args:
            config: Configuration dictionary for the tokenizer
        """
        self.config = config
        self.tokenizer = None
    
    @abstractmethod
    def load_tokenizer(self, model_name_or_path: Optional[str] = None) -> None:
        """
        Load a tokenizer.
        
        Args:
            model_name_or_path: Path or name of the model/tokenizer to load
        """
        pass
    
    @abstractmethod
    def tokenize(self, datasets: DatasetDict) -> DatasetDict:
        """
        Tokenize the datasets.
        
        Args:
            datasets: Datasets to tokenize
            
        Returns:
            DatasetDict: Tokenized datasets
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: Union[list, int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            
        Returns:
            str: Decoded text
        """
        pass
    
    def get_tokenizer(self):
        """
        Get the underlying tokenizer.
        
        Returns:
            The tokenizer instance
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        return self.tokenizer
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size of the tokenizer.
        
        Returns:
            int: Vocabulary size
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        return self.tokenizer.vocab_size 