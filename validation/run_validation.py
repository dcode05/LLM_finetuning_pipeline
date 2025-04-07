#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main validation script that runs the entire pipeline validation process.
"""

import os
import sys
import logging
import subprocess
import time
import argparse
import importlib
from typing import List, Tuple

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

def main():
    """
    Run the full validation process:
    1. Generate synthetic data
    2. Run the pipeline with the data
    """
    # Check if dependencies are installed
    all_installed, missing_packages = check_dependencies()
    if not all_installed:
        logger.error("Missing required packages: " + ", ".join(missing_packages))
        logger.error("Please run 'python validation/setup_validation.py' to install them.")
        return
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the full validation process')
    parser.add_argument('--config', type=str, default='synthetic_test_config.json',
                       choices=['synthetic_test_config.json', 'quick_test_config.json'],
                       help='Configuration file to use (default: synthetic_test_config.json)')
    parser.add_argument('--skip-data-generation', action='store_true',
                       help='Skip the data generation step if data already exists')
    args = parser.parse_args()
    
    start_time = time.time()
    validation_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(validation_dir)
    
    logger.info(f"Starting validation process with config: {args.config}...")
    
    # Step 1: Generate synthetic data (unless skipped)
    if args.skip_data_generation:
        logger.info("Skipping synthetic data generation (as requested)")
    else:
        logger.info("Step 1: Generating synthetic data")
        data_generator_path = os.path.join(validation_dir, "generate_synthetic_data.py")
        
        try:
            subprocess.run([sys.executable, data_generator_path], check=True)
            logger.info("Synthetic data generated successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating synthetic data: {e}")
            return
    
    # Step 2: Run the pipeline with the data
    logger.info("Step 2: Running the pipeline")
    pipeline_test_path = os.path.join(validation_dir, "test_pipeline.py")
    
    try:
        subprocess.run([sys.executable, pipeline_test_path, "--config", args.config], check=True)
        logger.info("Pipeline test completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running pipeline: {e}")
        return
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Validation completed in {total_time:.2f} seconds")
    
    # Determine the output directory from the config filename
    if args.config == "quick_test_config.json":
        output_dir = "quick_test"
    else:
        output_dir = "synthetic_test"
    
    logger.info(f"Validation outputs saved to: {os.path.join(validation_dir, 'outputs', output_dir)}")

if __name__ == "__main__":
    main() 