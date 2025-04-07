#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline executor module - handles running the pipeline components.
"""

import logging
import os
import time
from typing import Dict, Any

from pipeline.builder import Pipeline


def execute_pipeline(pipeline: Pipeline, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the pipeline with all components.
    
    Args:
        pipeline: The pipeline to execute
        config: Configuration dictionary
    
    Returns:
        Dict containing results of pipeline execution
    """
    start_time = time.time()
    results = {}
    
    try:
        # Validate configuration
        if not isinstance(config, dict):
            logging.error(f"Configuration must be a dictionary, got {type(config)}")
            results["error"] = f"Invalid configuration type: {type(config)}"
            return results
        
        # Debug: Log all top-level keys in the config
        logging.info(f"Configuration contains keys: {list(config.keys())}")
        
        output_dir = config.get("output_dir", "outputs")
        logging.info(f"Output directory: {output_dir}")
        
        logging.info(f"Starting pipeline execution")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process data
        logging.info("Processing data")
        data_results = process_data(pipeline, config)
        results["data"] = data_results
        
        # Check for errors in data processing
        if "error" in data_results:
            logging.error(f"Data processing failed: {data_results['error']}")
            results["error"] = f"Pipeline failed during data processing: {data_results['error']}"
            results["elapsed_time"] = time.time() - start_time
            return results
        
        # Prepare model
        logging.info("Preparing model")
        model_results = prepare_model(pipeline, config)
        results["model"] = model_results
        
        # Check for errors in model preparation
        if "error" in model_results:
            logging.error(f"Model preparation failed: {model_results['error']}")
            results["error"] = f"Pipeline failed during model preparation: {model_results['error']}"
            results["elapsed_time"] = time.time() - start_time
            return results
        
        # Execute hyperparameter optimization if configured
        hpo_enabled = False
        if "hpo" in config and isinstance(config["hpo"], dict):
            hpo_enabled = config["hpo"].get("enabled", False)
        
        if "hpo_optimizer" in pipeline.components and hpo_enabled:
            logging.info("Running hyperparameter optimization")
            try:
                hpo_results = run_hyperparameter_optimization(pipeline, config)
                results["hpo"] = hpo_results
                
                # Update training config with best hyperparameters if available
                if "best_params" in hpo_results:
                    logging.info(f"Updating training config with best parameters: {hpo_results['best_params']}")
                    # Make a deep copy to avoid modifying the original config
                    if "training" in config and isinstance(config["training"], dict):
                        config["training"] = {**config["training"], **hpo_results["best_params"]}
                    else:
                        logging.warning("Cannot update training config: training section is missing or invalid")
            except Exception as e:
                logging.error(f"Hyperparameter optimization failed: {str(e)}")
                results["hpo"] = {"error": str(e)}
        else:
            logging.info("Skipping hyperparameter optimization")
        
        # Execute training
        if "trainer" in pipeline.components:
            logging.info("Running training")
            try:
                if "training" not in config or not isinstance(config["training"], dict):
                    logging.error("Training configuration missing or invalid")
                    results["training"] = {"error": "Invalid training configuration"}
                else:
                    training_results = run_training(pipeline, config)
                    results["training"] = training_results
            except Exception as e:
                logging.error(f"Training failed: {str(e)}")
                results["training"] = {"error": str(e)}
        
        # Execute evaluation
        if "evaluator" in pipeline.components:
            logging.info("Running evaluation")
            try:
                if "evaluation" not in config or not isinstance(config["evaluation"], dict):
                    logging.error("Evaluation configuration missing or invalid")
                    results["evaluation"] = {"error": "Invalid evaluation configuration"}
                else:
                    evaluation_results = run_evaluation(pipeline, config)
                    results["evaluation"] = evaluation_results
            except Exception as e:
                logging.error(f"Evaluation failed: {str(e)}")
                results["evaluation"] = {"error": str(e)}
        
        # Save results
        elapsed_time = time.time() - start_time
        results["elapsed_time"] = elapsed_time
        
        logging.info(f"Pipeline execution completed in {elapsed_time:.2f} seconds")
        
        return results
    except Exception as e:
        logging.error(f"Unexpected error in execute_pipeline: {str(e)}")
        results["error"] = f"Pipeline execution failed: {str(e)}"
        results["elapsed_time"] = time.time() - start_time
        return results


def process_data(pipeline: Pipeline, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process data for training."""
    results = {}
    
    try:
        # Validate data configuration
        if "data" not in config:
            logging.error("No 'data' section found in configuration")
            results["error"] = "Missing data configuration"
            return results
        
        if not isinstance(config["data"], dict):
            logging.error(f"'data' section must be a dictionary, got {type(config['data'])}")
            results["error"] = f"Invalid data configuration type: {type(config['data'])}"
            return results
        
        data_config = config["data"]
        
        # Validate dataset configuration
        if "dataset" not in data_config:
            logging.error("No 'dataset' section found in data configuration")
            results["error"] = "Missing dataset configuration"
            return results
        
        if not isinstance(data_config["dataset"], dict):
            logging.error(f"'dataset' section must be a dictionary, got {type(data_config['dataset'])}")
            results["error"] = f"Invalid dataset configuration type: {type(data_config['dataset'])}"
            return results
        
        # Load dataset
        dataset_loader = pipeline.get_component("dataset_loader")
        if dataset_loader:
            logging.info("Loading dataset")
            try:
                raw_datasets = dataset_loader.load_datasets(data_config["dataset"])
                results["raw_datasets"] = raw_datasets
            except Exception as e:
                logging.error(f"Error loading dataset: {str(e)}")
                results["error"] = f"Dataset loading error: {str(e)}"
                return results
        else:
            logging.warning("No dataset loader component found")
            results["warning"] = "No dataset loader available"
            return results
        
        # Preprocess data
        preprocessor = pipeline.get_component("preprocessor")
        if preprocessor and "raw_datasets" in results:
            logging.info("Preprocessing dataset")
            try:
                preprocessed_datasets = preprocessor.preprocess(results["raw_datasets"])
                results["preprocessed_datasets"] = preprocessed_datasets
            except Exception as e:
                logging.error(f"Error preprocessing dataset: {str(e)}")
                results["error"] = f"Preprocessing error: {str(e)}"
                return results
        
        # Tokenize data
        tokenizer = pipeline.get_component("tokenizer")
        if tokenizer and "preprocessed_datasets" in results:
            logging.info("Tokenizing dataset")
            try:
                tokenized_datasets = tokenizer.tokenize(results["preprocessed_datasets"])
                results["tokenized_datasets"] = tokenized_datasets
            except Exception as e:
                logging.error(f"Error tokenizing dataset: {str(e)}")
                results["error"] = f"Tokenization error: {str(e)}"
                return results
        
        return results
    except Exception as e:
        logging.error(f"Unexpected error in process_data: {str(e)}")
        results["error"] = f"Unexpected error: {str(e)}"
        return results


def prepare_model(pipeline: Pipeline, config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare model for training."""
    results = {}
    
    # Get model
    model = pipeline.get_component("model")
    if model:
        logging.info(f"Model component found: {type(model).__name__}")
        
        # Debug information about model
        logging.info(f"Model config keys: {list(model.config.keys())}")
        
        # Make sure the model is loaded
        if hasattr(model, 'model') and model.model is None:
            logging.warning("Model has not been loaded yet")
            # Try to load the model if not loaded yet
            model_path = model.config.get('name_or_path')
            if model_path:
                logging.info(f"Loading model from {model_path}")
                try:
                    model.load_model(model_path)
                    logging.info("Model loaded successfully")
                    results["loaded"] = True
                except Exception as e:
                    logging.error(f"Error loading model: {str(e)}")
                    results["error"] = str(e)
            else:
                logging.error("No name_or_path found in config, cannot load model")
                results["error"] = "No name_or_path in config"
        else:
            logging.info("Model already loaded or not a standard model class")
            results["loaded"] = True
            
        # Skip get_model_info to avoid potential errors
        results["model_type"] = model.config.get("model_type", "unknown")
    else:
        logging.warning("No model component found in pipeline")
        results["error"] = "No model component found"
    
    # Apply adapter if configured
    adapter = pipeline.get_component("adapter")
    if adapter and model and hasattr(model, 'model') and model.model is not None:
        logging.info("Applying adapter to model")
        try:
            adapted_model = adapter.adapt_model(model.model)
            results["is_adapted"] = True
            logging.info("Adapter applied successfully")
        except Exception as e:
            logging.error(f"Error applying adapter: {str(e)}")
            results["adapter_error"] = str(e)
    else:
        if not adapter:
            logging.info("No adapter component found")
        elif not model:
            logging.warning("No model component found")
        else:
            logging.warning("Model not loaded, cannot apply adapter")
    
    return results


def run_hyperparameter_optimization(pipeline: Pipeline, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run hyperparameter optimization."""
    results = {}
    
    hpo_optimizer = pipeline.get_component("hpo_optimizer")
    if hpo_optimizer:
        logging.info(f"Running HPO with optimizer: {type(hpo_optimizer).__name__}")
        # Define an objective function that will be optimized
        def objective_fn(trial_config):
            # Copy the current config and update with trial parameters
            logging.info(f"Running trial with config: {trial_config}")
            trial_training_config = config["training"].copy()
            trial_training_config.update(trial_config)
            
            # Create a temporary trainer for this trial
            from training.factory import create_trainer
            temp_trainer = create_trainer(trial_training_config)
            
            # Train and evaluate with this configuration
            temp_results = temp_trainer.train(
                model=pipeline.get_component("model").model,
                tokenizer=pipeline.get_component("tokenizer").tokenizer if pipeline.get_component("tokenizer") else None,
                dataset=pipeline.get_component("dataset_loader").get_processed_dataset() if pipeline.get_component("dataset_loader") else None,
                config=trial_training_config,
            )
            
            # Return the metric to optimize
            metric_name = config["hpo"].get("metric", "eval_loss")
            metric_value = temp_results.get(metric_name, float('inf'))
            
            # Handle minimization vs maximization
            if config["hpo"].get("direction", "minimize") == "minimize":
                return metric_value
            else:
                return -metric_value  # Negate for maximization
        
        # Get search space from config - new format uses 'parameters' instead of 'search_space'
        search_space = config["hpo"].get("parameters", {})
        if not search_space:
            # Fall back to old format for backward compatibility
            search_space = config["hpo"].get("search_space", {})
            
        logging.info(f"HPO search space: {search_space}")
        
        if not search_space:
            logging.warning("Empty search space for HPO, skipping optimization")
            results["warning"] = "Empty search space"
            return results
            
        # Run optimization
        try:
            best_params = hpo_optimizer.optimize(objective_fn, search_space)
            results["best_params"] = best_params
            results["summary"] = hpo_optimizer.get_results_summary()
            logging.info(f"HPO completed successfully. Best params: {best_params}")
        except Exception as e:
            logging.error(f"Error during HPO: {str(e)}")
            results["error"] = str(e)
    else:
        logging.warning("No HPO optimizer found in pipeline")
        results["error"] = "No HPO optimizer found"
    
    return results


def run_training(pipeline: Pipeline, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run model training."""
    results = {}
    
    trainer = pipeline.get_component("trainer")
    if trainer:
        training_results = trainer.train(
            model=pipeline.get_component("model").model,
            tokenizer=pipeline.get_component("tokenizer").tokenizer if pipeline.get_component("tokenizer") else None,
            dataset=pipeline.get_component("dataset_loader").get_processed_dataset() if pipeline.get_component("dataset_loader") else None,
            config=config["training"],
        )
        results.update(training_results)
    
    return results


def run_evaluation(pipeline: Pipeline, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run model evaluation."""
    results = {}
    
    evaluator = pipeline.get_component("evaluator")
    if evaluator:
        evaluation_results = evaluator.evaluate(
            model=pipeline.get_component("model").model,
            tokenizer=pipeline.get_component("tokenizer").tokenizer if pipeline.get_component("tokenizer") else None,
            dataset=pipeline.get_component("dataset_loader").get_processed_dataset() if pipeline.get_component("dataset_loader") else None,
            config=config["evaluation"],
        )
        results.update(evaluation_results)
    
    return results 