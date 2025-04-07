#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for data preprocessors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List, Callable

from datasets import DatasetDict, Dataset


class BasePreprocessor(ABC):
    """
    Base class for data preprocessors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary for the preprocessor
        """
        self.config = config
    
    @abstractmethod
    def preprocess(self, datasets: DatasetDict) -> DatasetDict:
        """
        Preprocess the datasets.
        
        Args:
            datasets: Datasets to preprocess
            
        Returns:
            DatasetDict: Preprocessed datasets
        """
        pass
    
    def apply_to_column(
        self, 
        datasets: DatasetDict, 
        column: str, 
        function: Callable, 
        batched: bool = False,
        batch_size: int = 1000,
        remove_columns = None
    ) -> DatasetDict:
        """
        Apply a function to a column in all datasets.
        
        Args:
            datasets: Datasets to process
            column: Column to apply function to
            function: Function to apply
            batched: Whether to apply function to batches
            batch_size: Batch size if batched is True
            remove_columns: Columns to remove after processing
            
        Returns:
            DatasetDict: Processed datasets
        """
        processed_datasets = DatasetDict()
        
        for split_name, split_dataset in datasets.items():
            if column in split_dataset.column_names:
                if batched:
                    processed_datasets[split_name] = split_dataset.map(
                        function,
                        batched=True,
                        batch_size=batch_size,
                        remove_columns=remove_columns,
                    )
                else:
                    processed_datasets[split_name] = split_dataset.map(
                        function,
                        remove_columns=remove_columns,
                    )
            else:
                # Keep dataset unchanged if column doesn't exist
                processed_datasets[split_name] = split_dataset
        
        return processed_datasets
    
    def filter_dataset(self, datasets: DatasetDict, function: Callable) -> DatasetDict:
        """
        Filter datasets based on a function.
        
        Args:
            datasets: Datasets to filter
            function: Function to apply for filtering
            
        Returns:
            DatasetDict: Filtered datasets
        """
        filtered_datasets = DatasetDict()
        
        for split_name, split_dataset in datasets.items():
            filtered_datasets[split_name] = split_dataset.filter(function)
        
        return filtered_datasets
    
    def get_stats(self, datasets: DatasetDict) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about the datasets.
        
        Args:
            datasets: Datasets to get statistics for
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of dataset statistics
        """
        stats = {}
        
        for split_name, split_dataset in datasets.items():
            stats[split_name] = {
                "num_examples": len(split_dataset),
                "columns": split_dataset.column_names,
            }
        
        return stats 