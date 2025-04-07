#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for running the LLM finetuning pipeline with synthetic data.
"""

import logging
import json
import os
import sys
import argparse
import importlib
from typing import Dict, Any, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define required packages
REQUIRED_PACKAGES = [
    "numpy", "torch", "transformers", "datasets", "peft", "evaluate", "sklearn"
]

def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required packages are installed.
    
    Returns:
        Tuple containing (all_installed, missing_packages)
    """
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    """Run the pipeline with synthetic data."""
    # Check if dependencies are installed
    all_installed, missing_packages = check_dependencies()
    if not all_installed:
        logger.error("Missing required packages: " + ", ".join(missing_packages))
        logger.error("Please run 'python validation/setup_validation.py' to install them.")
        return
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the LLM finetuning pipeline with synthetic data')
    parser.add_argument('--config', type=str, default='synthetic_test_config.json',
                        choices=['synthetic_test_config.json', 'quick_test_config.json'],
                        help='Configuration file to use (default: synthetic_test_config.json)')
    args = parser.parse_args()
    
    # Add the parent directory to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load configuration
    config_path = os.path.join('validation', args.config)
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        return
    
    # Load the configuration
    config = load_config(config_path)
    logger.info(f"Configuration loaded successfully from {args.config}")
    
    try:
        # Import the framework components
        from pipeline.builder import create_pipeline
        from pipeline.executor import execute_pipeline
        
        # Create the pipeline
        pipeline = create_pipeline(config)
        logger.info(f"Pipeline created with components: {list(pipeline.components.keys())}")
        
        # Execute the pipeline
        results = execute_pipeline(pipeline, config)
        
        # Print results summary
        logger.info("\nPipeline execution completed!")
        logger.info(f"Total execution time: {results['elapsed_time']:.2f} seconds")
        
        if "evaluation" in results:
            logger.info("\nEvaluation results:")
            for metric, value in results["evaluation"].items():
                logger.info(f"  {metric}: {value}")
        
        if "hpo" in results and "best_params" in results["hpo"]:
            logger.info("\nBest hyperparameters:")
            for param, value in results["hpo"]["best_params"].items():
                logger.info(f"  {param}: {value}")
        
        logger.info(f"\nAll outputs saved to: {config['output_dir']}")
    except ImportError as e:
        logger.error(f"Error importing framework components: {e}")
        logger.error("Make sure all dependencies are installed correctly.")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")

if __name__ == "__main__":
    main() 