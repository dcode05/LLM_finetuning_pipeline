#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text preprocessor for cleaning and preparing text data.
"""

import logging
import re
from typing import Dict, Any, List, Callable, Optional, Union

from datasets import DatasetDict, Dataset

from data.preprocessing.base import BasePreprocessor


class TextPreprocessor(BasePreprocessor):
    """
    Preprocessor for text data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text preprocessor.
        
        Args:
            config: Configuration dictionary for the preprocessor
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration parameters
        self.cleaning_config = self.config.get("cleaning", {})
        self.filtering_config = self.config.get("filtering", {})
        self.augmentation_config = self.config.get("data_augmentation", {})
        
        # Set up input and output column names
        self.input_column = config.get("input_column", None)
        self.output_column = config.get("output_column", None)
    
    def preprocess(self, datasets: DatasetDict) -> DatasetDict:
        """
        Preprocess the datasets.
        
        Args:
            datasets: Datasets to preprocess
            
        Returns:
            DatasetDict: Preprocessed datasets
        """
        self.logger.info("Starting text preprocessing")
        preprocessed_datasets = datasets
        
        # Log dataset statistics before preprocessing
        before_stats = self.get_stats(preprocessed_datasets)
        self.logger.info(f"Dataset statistics before preprocessing: {before_stats}")
        
        # Apply text cleaning if configured
        if self.cleaning_config:
            self.logger.info("Applying text cleaning")
            preprocessed_datasets = self.clean_text(preprocessed_datasets)
        
        # Apply filtering if configured
        if self.filtering_config:
            self.logger.info("Applying text filtering")
            preprocessed_datasets = self.filter_text(preprocessed_datasets)
        
        # Apply data augmentation if configured and enabled
        if self.augmentation_config.get("enabled", False):
            self.logger.info("Applying data augmentation")
            preprocessed_datasets = self.augment_data(preprocessed_datasets)
        
        # Log dataset statistics after preprocessing
        after_stats = self.get_stats(preprocessed_datasets)
        self.logger.info(f"Dataset statistics after preprocessing: {after_stats}")
        
        return preprocessed_datasets
    
    def clean_text(self, datasets: DatasetDict) -> DatasetDict:
        """
        Apply text cleaning operations to datasets.
        
        Args:
            datasets: Datasets to clean
            
        Returns:
            DatasetDict: Cleaned datasets
        """
        # Determine which columns to process
        columns_to_process = []
        if self.input_column:
            columns_to_process.append(self.input_column)
        if self.output_column:
            columns_to_process.append(self.output_column)
        
        # If no specific columns are configured, process all text columns
        if not columns_to_process:
            # Get column names from the first split
            first_split = next(iter(datasets.values()))
            columns_to_process = first_split.column_names
        
        cleaned_datasets = datasets
        
        for column in columns_to_process:
            # Define cleaning function for this column
            def clean_function(examples):
                if column not in examples:
                    return examples
                
                texts = examples[column]
                
                # Apply each enabled cleaning operation
                if self.cleaning_config.get("remove_html", False):
                    texts = [self._remove_html(text) if text else text for text in texts]
                
                if self.cleaning_config.get("fix_unicode", False):
                    texts = [self._fix_unicode(text) if text else text for text in texts]
                
                if self.cleaning_config.get("normalize_whitespace", False):
                    texts = [self._normalize_whitespace(text) if text else text for text in texts]
                
                # Add more cleaning operations as needed
                
                examples[column] = texts
                return examples
            
            # Apply cleaning to the column
            self.logger.info(f"Cleaning column: {column}")
            cleaned_datasets = self.apply_to_column(
                cleaned_datasets, 
                column, 
                clean_function, 
                batched=True
            )
        
        return cleaned_datasets
    
    def filter_text(self, datasets: DatasetDict) -> DatasetDict:
        """
        Filter examples based on text criteria.
        
        Args:
            datasets: Datasets to filter
            
        Returns:
            DatasetDict: Filtered datasets
        """
        # Extract filtering criteria
        min_length = self.filtering_config.get("min_length", None)
        max_length = self.filtering_config.get("max_length", None)
        
        if not (min_length or max_length):
            self.logger.info("No length filtering criteria specified")
            return datasets
        
        # Determine which column to use for filtering
        filter_column = self.input_column or (
            datasets[next(iter(datasets.keys()))].column_names[0]
        )
        
        self.logger.info(f"Filtering based on column: {filter_column}")
        self.logger.info(f"Filtering criteria - min_length: {min_length}, max_length: {max_length}")
        
        # Define filtering function
        def filter_function(example):
            text = example.get(filter_column, "")
            if not text:
                return False
            
            text_length = len(text)
            
            if min_length and text_length < min_length:
                return False
            
            if max_length and text_length > max_length:
                return False
            
            return True
        
        # Apply filtering
        filtered_datasets = self.filter_dataset(datasets, filter_function)
        
        # Log filtering results
        for split_name in datasets.keys():
            before_count = len(datasets[split_name])
            after_count = len(filtered_datasets[split_name])
            self.logger.info(f"Split '{split_name}': Filtered {before_count - after_count} examples ({before_count} -> {after_count})")
        
        return filtered_datasets
    
    def augment_data(self, datasets: DatasetDict) -> DatasetDict:
        """
        Apply data augmentation techniques.
        
        Args:
            datasets: Datasets to augment
            
        Returns:
            DatasetDict: Augmented datasets
        """
        # Only augment training data
        if "train" not in datasets:
            self.logger.info("No training split found, skipping data augmentation")
            return datasets
        
        techniques = self.augmentation_config.get("techniques", [])
        if not techniques:
            self.logger.info("No augmentation techniques specified")
            return datasets
        
        augmented_datasets = DatasetDict()
        
        # Copy non-training splits unchanged
        for split_name, split_dataset in datasets.items():
            if split_name != "train":
                augmented_datasets[split_name] = split_dataset
        
        # Augment training split
        train_dataset = datasets["train"]
        augmented_train = train_dataset
        
        # Apply each augmentation technique
        for technique in techniques:
            if technique == "synonym_replacement":
                self.logger.info("Applying synonym replacement augmentation")
                # Implementation would go here
            elif technique == "random_swap":
                self.logger.info("Applying random swap augmentation")
                # Implementation would go here
            elif technique == "back_translation":
                self.logger.info("Applying back translation augmentation")
                # Implementation would go here
        
        augmented_datasets["train"] = augmented_train
        
        return augmented_datasets
    
    # Helper methods for text cleaning
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return text
        
        # Simple regex to remove HTML tags
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text
    
    def _fix_unicode(self, text: str) -> str:
        """Fix common unicode issues."""
        if not text:
            return text
        
        # Replace common unicode characters
        replacements = {
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En-dash
            '\u2014': '--', # Em-dash
            '\u00a0': ' ',  # Non-breaking space
        }
        
        for orig, replacement in replacements.items():
            text = text.replace(orig, replacement)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        if not text:
            return text
        
        # Replace multiple spaces with a single space
        clean_text = re.sub(r'\s+', ' ', text)
        
        # Trim leading and trailing whitespace
        clean_text = clean_text.strip()
        
        return clean_text 