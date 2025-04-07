#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline builder module - builds the pipeline components based on configuration.
"""

import logging
from typing import Dict, Any, List

from data.preprocessing.factory import create_preprocessor
from data.tokenization.factory import create_tokenizer
from data.dataset.factory import create_dataset_loader
from models.factory import create_model, create_adapter
from training.factory import create_trainer
from evaluation.factory import create_evaluator
from hpo.factory import create_hpo_optimizer


class Pipeline:
    """
    Pipeline class that holds all components for the finetuning pipeline.
    """
    
    def __init__(self):
        self.components = {}
    
    def add_component(self, name: str, component: Any) -> None:
        """Add a component to the pipeline."""
        self.components[name] = component
    
    def get_component(self, name: str) -> Any:
        """Get a component from the pipeline."""
        return self.components.get(name)


def create_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    Create a pipeline from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Pipeline: Assembled pipeline with all components
    """
    logging.info("Creating pipeline components")
    pipeline = Pipeline()
    
    # Create data components
    if "data" in config:
        data_config = config["data"]
        
        # Create preprocessor if configured
        if "preprocessing" in data_config:
            logging.info("Creating data preprocessor")
            preprocessor = create_preprocessor(data_config["preprocessing"])
            pipeline.add_component("preprocessor", preprocessor)
        
        # Create tokenizer if configured
        if "tokenization" in data_config:
            logging.info("Creating tokenizer")
            tokenizer = create_tokenizer(data_config["tokenization"])
            pipeline.add_component("tokenizer", tokenizer)
        
        # Create dataset loader if configured
        if "dataset" in data_config:
            logging.info("Creating dataset loader")
            dataset_loader = create_dataset_loader(data_config["dataset"])
            pipeline.add_component("dataset_loader", dataset_loader)
    
    # Create model components
    if "model" in config:
        model_config = config["model"]
        
        # Create base model
        logging.info(f"Creating model: {model_config.get('name_or_path', 'custom')}")
        model = create_model(model_config)
        pipeline.add_component("model", model)
        
        # Create model adapter if configured
        if "adapter" in model_config:
            logging.info(f"Creating adapter: {model_config['adapter'].get('type', 'custom')}")
            adapter = create_adapter(model_config["adapter"])
            pipeline.add_component("adapter", adapter)
    
    # Create training components
    if "training" in config:
        logging.info("Creating trainer")
        trainer = create_trainer(config["training"])
        pipeline.add_component("trainer", trainer)
    
    # Create evaluation components
    if "evaluation" in config:
        logging.info("Creating evaluator")
        evaluator = create_evaluator(config["evaluation"])
        pipeline.add_component("evaluator", evaluator)
    
    # Create HPO components if configured
    if "hpo" in config:
        logging.info("Creating HPO optimizer")
        hpo_optimizer = create_hpo_optimizer(config["hpo"])
        pipeline.add_component("hpo_optimizer", hpo_optimizer)
    
    logging.info(f"Pipeline created with components: {list(pipeline.components.keys())}")
    
    return pipeline 