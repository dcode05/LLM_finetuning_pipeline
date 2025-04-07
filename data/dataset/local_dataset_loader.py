#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Local dataset loader implementation for loading datasets from disk.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset

from data.dataset.base import BaseDatasetLoader


class LocalDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for loading datasets from local files.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the local dataset loader.
        
        Args:
            config: Configuration dictionary for the dataset loader
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def load_datasets(self, config: Optional[Dict[str, Any]] = None) -> DatasetDict:
        """
        Load datasets from local files.
        
        Args:
            config: Optional configuration override
            
        Returns:
            DatasetDict: Dictionary of datasets
        """
        config = config or self.config
        data_dir = config.get("dir", "data")
        
        # Check if this is a Hugging Face dataset saved to disk
        if "saved_dataset_path" in config:
            self.logger.info(f"Loading dataset from disk: {config['saved_dataset_path']}")
            try:
                self.datasets = load_from_disk(config["saved_dataset_path"])
                if not isinstance(self.datasets, DatasetDict):
                    # Convert Dataset to DatasetDict
                    self.datasets = DatasetDict({"data": self.datasets})
                return self.datasets
            except Exception as e:
                self.logger.error(f"Error loading dataset from disk: {str(e)}")
                raise
        
        # Otherwise, load from individual files
        self.logger.info("Loading dataset from local files")
        
        dataset_format = config.get("dataset_format", "json")
        self.logger.info(f"Dataset format: {dataset_format}")
        
        # Get file paths
        train_file = os.path.join(data_dir, config.get("train_file")) if config.get("train_file") else None
        validation_file = os.path.join(data_dir, config.get("validation_file")) if config.get("validation_file") else None
        test_file = os.path.join(data_dir, config.get("test_file")) if config.get("test_file") else None
        
        if not any([train_file, validation_file, test_file]):
            raise ValueError("No data files specified. At least one of train_file, validation_file, or test_file must be provided.")
        
        # Log the file paths
        for name, path in [("Train", train_file), ("Validation", validation_file), ("Test", test_file)]:
            if path:
                self.logger.info(f"{name} file: {path}")
                if not os.path.exists(path):
                    self.logger.warning(f"{name} file {path} does not exist!")
        
        try:
            # Load each split
            self.datasets = DatasetDict()
            
            # Common loading arguments
            load_kwargs = {
                "keep_in_memory": config.get("keep_in_memory", False),
                "cache_dir": config.get("cache_dir", None),
            }
            
            # Load train dataset
            if train_file and os.path.exists(train_file):
                self.logger.info(f"Loading training data from {train_file}")
                self.datasets["train"] = load_dataset(
                    dataset_format, 
                    data_files=train_file, 
                    split="train",
                    **load_kwargs
                )
            
            # Load validation dataset
            if validation_file and os.path.exists(validation_file):
                self.logger.info(f"Loading validation data from {validation_file}")
                self.datasets["validation"] = load_dataset(
                    dataset_format, 
                    data_files=validation_file, 
                    split="train",  # Using 'train' split because we're loading a single file
                    **load_kwargs
                )
            
            # Load test dataset
            if test_file and os.path.exists(test_file):
                self.logger.info(f"Loading test data from {test_file}")
                self.datasets["test"] = load_dataset(
                    dataset_format, 
                    data_files=test_file, 
                    split="train",  # Using 'train' split because we're loading a single file
                    **load_kwargs
                )
            
            # Apply column filtering if specified
            if "filter_columns" in config:
                self.logger.info(f"Filtering columns to: {config['filter_columns']}")
                self.datasets = self.filter_columns(config["filter_columns"])
            
            # Apply column renaming if specified
            if "column_mapping" in config:
                self.logger.info(f"Renaming columns according to mapping: {config['column_mapping']}")
                self.datasets = self.rename_columns(config["column_mapping"])
            
            # Log dataset information
            self.logger.info(f"Successfully loaded dataset with splits: {list(self.datasets.keys())}")
            for split_name, split_dataset in self.datasets.items():
                self.logger.info(f"Split '{split_name}' has {len(split_dataset)} examples")
            
            return self.datasets
            
        except Exception as e:
            self.logger.error(f"Error loading dataset from local files: {str(e)}")
            raise
    
    def save_to_disk(self, output_dir: str) -> None:
        """
        Save the datasets to disk.
        
        Args:
            output_dir: Output directory
        """
        if self.datasets is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")
        
        self.logger.info(f"Saving datasets to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        self.datasets.save_to_disk(output_dir)
        self.logger.info(f"Datasets saved successfully") 