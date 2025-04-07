#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the LLM finetuning pipeline.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

import yaml

from utils.logging import setup_logging
from pipeline.builder import create_pipeline
from pipeline.executor import execute_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Finetuning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to the configuration file"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=None,
        help="Directory containing the training data"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--model-name-or-path", 
        type=str, 
        default=None,
        help="Pretrained model name or path"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", 
        help="Logging level"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update configuration with command line arguments."""
    # Override config with command line arguments if provided
    if args.data_dir is not None:
        config["data"]["dir"] = args.data_dir
    
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    
    if args.model_name_or_path is not None:
        config["model"]["name_or_path"] = args.model_name_or_path
    
    # Set debug mode if requested
    if args.debug:
        config["debug"] = True
        config["log_level"] = "DEBUG"
    elif args.log_level:
        config["log_level"] = args.log_level
    
    return config


def main() -> None:
    """Main entry point for the pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Setup logging
    setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_file=os.path.join(config.get("output_dir", "logs"), "pipeline.log"),
    )
    
    logging.info("Starting LLM finetuning pipeline")
    logging.debug(f"Configuration: {config}")
    
    try:
        # Create pipeline
        pipeline = create_pipeline(config)
        
        # Execute pipeline
        results = execute_pipeline(pipeline, config)
        
        logging.info("Pipeline execution completed successfully")
        
        # Print summary of results
        print("\nPipeline Execution Summary:")
        print(f"Model: {config['model']['name_or_path']}")
        print(f"Output directory: {config['output_dir']}")
        if "evaluation" in results:
            print("\nEvaluation Results:")
            for metric, value in results["evaluation"].items():
                print(f"  {metric}: {value}")
    
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 