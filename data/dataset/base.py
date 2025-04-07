#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for dataset loaders.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List

from datasets import DatasetDict, Dataset


class BaseDatasetLoader(ABC):
    """
    Base class for dataset loaders.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset loader.
        
        Args:
            config: Configuration dictionary for the dataset loader
        """
        self.config = config
        self.datasets = None
        self.processed_datasets = None
    
    @abstractmethod
    def load_datasets(self, config: Optional[Dict[str, Any]] = None) -> DatasetDict:
        """
        Load datasets based on configuration.
        
        Args:
            config: Optional configuration override
            
        Returns:
            DatasetDict: Dictionary of datasets
        """
        pass
    
    def get_column_names(self) -> List[str]:
        """
        Get column names from the dataset.
        
        Returns:
            List[str]: List of column names
        """
        if self.datasets is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")
        
        # Get column names from the first split
        first_split = next(iter(self.datasets.values()))
        return first_split.column_names
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dict[str, Any]: Dictionary of dataset statistics
        """
        if self.datasets is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")
        
        stats = {}
        for split_name, split_dataset in self.datasets.items():
            stats[split_name] = {
                "num_examples": len(split_dataset),
                "columns": split_dataset.column_names,
            }
        
        return stats
    
    def filter_columns(self, columns: List[str]) -> DatasetDict:
        """
        Filter datasets to include only specified columns.
        
        Args:
            columns: List of column names to keep
            
        Returns:
            DatasetDict: Filtered datasets
        """
        if self.datasets is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")
        
        filtered_datasets = DatasetDict()
        for split_name, split_dataset in self.datasets.items():
            # Get the intersection of available columns and requested columns
            available_columns = set(split_dataset.column_names)
            requested_columns = set(columns)
            columns_to_keep = list(available_columns.intersection(requested_columns))
            
            # Remove columns not in the list
            filtered_datasets[split_name] = split_dataset.select_columns(columns_to_keep)
        
        return filtered_datasets
    
    def rename_columns(self, column_mapping: Dict[str, str]) -> DatasetDict:
        """
        Rename columns in the datasets.
        
        Args:
            column_mapping: Dictionary mapping old column names to new column names
            
        Returns:
            DatasetDict: Datasets with renamed columns
        """
        if self.datasets is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")
        
        renamed_datasets = DatasetDict()
        for split_name, split_dataset in self.datasets.items():
            renamed_datasets[split_name] = split_dataset.rename_columns(column_mapping)
        
        return renamed_datasets
    
    def set_processed_dataset(self, processed_datasets: DatasetDict) -> None:
        """
        Set the processed datasets.
        
        Args:
            processed_datasets: Processed datasets to set
        """
        self.processed_datasets = processed_datasets
    
    def get_processed_dataset(self) -> Optional[DatasetDict]:
        """
        Get the processed datasets.
        
        Returns:
            Optional[DatasetDict]: Processed datasets if available
        """
        return self.processed_datasets or self.datasets 