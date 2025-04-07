#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HuggingFace dataset loader implementation.
"""

import logging
from typing import Dict, Any, Optional, List, Union

from datasets import load_dataset, DatasetDict, Dataset

from data.dataset.base import BaseDatasetLoader


class HuggingFaceDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for loading datasets from the HuggingFace Hub.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace dataset loader.
        
        Args:
            config: Configuration dictionary for the dataset loader
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def load_datasets(self, config: Optional[Dict[str, Any]] = None) -> DatasetDict:
        """
        Load datasets from the HuggingFace Hub.
        
        Args:
            config: Optional configuration override
            
        Returns:
            DatasetDict: Dictionary of datasets
        """
        config = config or self.config
        
        dataset_name = config.get("hf_dataset_name")
        if not dataset_name:
            raise ValueError("Dataset name (hf_dataset_name) must be provided for HuggingFace Hub datasets")
        
        dataset_config = config.get("hf_dataset_config")
        dataset_splits = config.get("hf_dataset_split", ["train", "validation"])
        
        # Convert splits to list if it's a string
        if isinstance(dataset_splits, str):
            dataset_splits = [dataset_splits]
        
        self.logger.info(f"Loading dataset {dataset_name} from HuggingFace Hub")
        if dataset_config:
            self.logger.info(f"Using configuration: {dataset_config}")
        
        try:
            # Load datasets from HuggingFace Hub
            kwargs = {}
            if dataset_config:
                kwargs["name"] = dataset_config
                
            # Load each split
            self.datasets = DatasetDict()
            for split in dataset_splits:
                try:
                    self.logger.info(f"Loading split: {split}")
                    self.datasets[split] = load_dataset(dataset_name, split=split, **kwargs)
                except ValueError as e:
                    self.logger.warning(f"Could not load split {split}: {str(e)}")
                    
            if not self.datasets:
                self.logger.error(f"Failed to load any splits for {dataset_name}")
                raise ValueError(f"No valid splits found for dataset {dataset_name}")
            
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
            self.logger.error(f"Error loading dataset from HuggingFace Hub: {str(e)}")
            raise
    
    def get_features_info(self) -> Dict[str, Any]:
        """
        Get information about the features in the dataset.
        
        Returns:
            Dict[str, Any]: Dictionary with feature information
        """
        if self.datasets is None:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")
        
        # Get features from the first split
        first_split = next(iter(self.datasets.values()))
        features = first_split.features
        
        return {
            "feature_names": list(features.keys()),
            "feature_types": {name: str(feat) for name, feat in features.items()}
        } 