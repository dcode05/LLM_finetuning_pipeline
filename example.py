#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for using the LLM Finetuning Framework.
This demonstrates how to load a configuration, create a pipeline,
execute it, and analyze the results.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("example")

# Import framework components
try:
    from pipeline.builder import create_pipeline
    from pipeline.executor import execute_pipeline
except ImportError:
    logger.error("Could not import pipeline modules. Make sure you're in the correct directory.")
    sys.exit(1)


def validate_config(config_path):
    """
    Validate the configuration file before using it.
    """
    logger.info(f"Validating configuration file: {config_path}")
    
    try:
        # Check if validate_config.py exists and use it if available
        if os.path.exists("validation/validate_config.py"):
            import subprocess
            result = subprocess.run(
                [sys.executable, "validation/validate_config.py", config_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Configuration validation failed: {result.stderr}")
                logger.error(result.stdout)
                if input("Continue anyway? (y/n): ").lower() != 'y':
                    return False
            else:
                logger.info("Configuration validation passed!")
                
    except Exception as e:
        logger.warning(f"Could not run configuration validator: {str(e)}")
        logger.warning("Continuing without validation...")
    
    # Basic validation - check if file exists and is valid JSON
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None


def setup_output_directory(config):
    """
    Create the output directory if it doesn't exist.
    """
    output_dir = config.get("output_dir", "outputs/default")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Set up a log file in the output directory
    log_file = os.path.join(output_dir, "finetuning.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    return output_dir


def run_finetuning(config):
    """
    Run the finetuning pipeline with the provided configuration.
    """
    try:
        # Create the pipeline
        logger.info("Creating pipeline...")
        pipeline = create_pipeline(config)
        
        # Execute the pipeline
        logger.info("Executing pipeline...")
        start_time = time.time()
        results = execute_pipeline(pipeline, config)
        elapsed_time = time.time() - start_time
        
        return results, elapsed_time
    except Exception as e:
        logger.error(f"Error during pipeline execution: {str(e)}", exc_info=True)
        return None, 0


def display_results(results, elapsed_time, output_dir):
    """
    Display and save the results of the finetuning process.
    """
    if not results:
        logger.error("No results to display.")
        return
    
    logger.info(f"Pipeline execution completed in {elapsed_time:.2f} seconds")
    
    # Display evaluation results
    if "evaluation" in results:
        logger.info("Evaluation Results:")
        for metric, value in results["evaluation"].items():
            logger.info(f"  {metric}: {value}")
    
    # Display hyperparameter optimization results if available
    if "hpo" in results and "best_params" in results["hpo"]:
        logger.info("Best Hyperparameters:")
        for param, value in results["hpo"]["best_params"].items():
            logger.info(f"  {param}: {value}")
    
    # Save results to a JSON file
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "elapsed_time": elapsed_time,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


def main():
    """
    Main function to parse arguments and run the finetuning process.
    """
    parser = argparse.ArgumentParser(description="LLM Finetuning Example")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    logger.info("Starting LLM Finetuning Example")
    logger.info(f"Configuration file: {args.config}")
    
    # Validate configuration
    config = validate_config(args.config)
    if not config:
        sys.exit(1)
    
    # Setup output directory
    output_dir = setup_output_directory(config)
    
    # Run finetuning
    results, elapsed_time = run_finetuning(config)
    
    # Display results
    display_results(results, elapsed_time, output_dir)
    
    logger.info("LLM Finetuning Example completed")


if __name__ == "__main__":
    main() 