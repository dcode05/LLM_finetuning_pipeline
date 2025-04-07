#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation script for checking configuration files.
This script validates the structure and types of configuration files
to ensure they match the expected format for the LLM finetuning pipeline.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, List, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expected top-level keys in configuration
TOP_LEVEL_KEYS = [
    "output_dir",
    "data",
    "model",
    "training",
    "evaluation",
    "hpo"
]

# Required keys for each section
REQUIRED_KEYS = {
    "data": ["preprocessing", "tokenization", "dataset"],
    "data.preprocessing": ["type"],
    "data.tokenization": ["type", "model_name_or_path"],
    "data.dataset": ["dir"],
    "model": ["name_or_path", "type"],
    "training": ["epochs", "batch_size", "learning_rate"],
    "evaluation": ["metrics"],
    "hpo": ["enabled"]
}

# Expected types for specific keys
EXPECTED_TYPES = {
    "output_dir": str,
    "data": dict,
    "data.preprocessing": dict,
    "data.tokenization": dict,
    "data.dataset": dict,
    "model": dict,
    "model.adapter": dict,
    "training": dict,
    "evaluation": dict,
    "hpo": dict,
    "hpo.parameters": dict,
    "hpo.enabled": bool,
    "data.preprocessing.type": str,
    "data.tokenization.type": str,
    "data.dataset.dir": str,
    "model.name_or_path": str,
    "model.type": str,
    "training.epochs": int,
    "training.batch_size": int,
    "training.learning_rate": (float, str),  # Can be float or string (for HPO)
    "evaluation.metrics": list
}

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {config_path}: {str(e)}")
        raise
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration file: {str(e)}")
        raise

def validate_keys(config: Dict[str, Any]) -> List[str]:
    """
    Validate that all required keys are present in the configuration.
    Returns a list of missing keys.
    """
    missing_keys = []

    # Check top-level keys
    for key in TOP_LEVEL_KEYS:
        if key not in config:
            missing_keys.append(key)
    
    # Check nested keys
    for section, required_keys in REQUIRED_KEYS.items():
        if "." in section:
            # This is a nested section
            parent, child = section.split(".", 1)
            if parent in config and isinstance(config[parent], dict) and child in config[parent]:
                for key in required_keys:
                    if key not in config[parent][child]:
                        missing_keys.append(f"{section}.{key}")
        else:
            # This is a top-level section
            if section in config and isinstance(config[section], dict):
                for key in required_keys:
                    if key not in config[section]:
                        missing_keys.append(f"{section}.{key}")
    
    return missing_keys

def validate_types(config: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Validate that all values have the expected types.
    Returns a list of (key, expected_type, actual_type) tuples for invalid types.
    """
    invalid_types = []

    # Check all expected types
    for key_path, expected_type in EXPECTED_TYPES.items():
        # Split the key path
        parts = key_path.split(".")
        
        # Navigate to the nested value
        current = config
        valid_path = True
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                valid_path = False
                break
        
        # Skip if the path doesn't exist
        if not valid_path:
            continue
        
        # Check the type
        if isinstance(expected_type, tuple):
            # Multiple allowed types
            if not any(isinstance(current, t) for t in expected_type):
                actual_type = type(current).__name__
                expected_type_str = " or ".join(t.__name__ for t in expected_type)
                invalid_types.append((key_path, expected_type_str, actual_type))
        else:
            # Single allowed type
            if not isinstance(current, expected_type):
                actual_type = type(current).__name__
                invalid_types.append((key_path, expected_type.__name__, actual_type))
    
    return invalid_types

def validate_config(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the configuration.
    Returns a tuple (is_valid, results) where results contains details about the validation.
    """
    results = {
        "missing_keys": [],
        "invalid_types": [],
        "warnings": []
    }
    
    # Check required keys
    results["missing_keys"] = validate_keys(config)
    
    # Check types
    results["invalid_types"] = validate_types(config)
    
    # Additional checks
    
    # Check if data preprocessing has text_column and label_column if needed
    if ("data" in config and isinstance(config["data"], dict) and 
        "preprocessing" in config["data"] and isinstance(config["data"]["preprocessing"], dict)):
        preprocessing = config["data"]["preprocessing"]
        if "type" in preprocessing and preprocessing["type"] == "text":
            if "text_column" not in preprocessing:
                results["warnings"].append("Missing 'text_column' in data.preprocessing for text type")
            if "label_column" not in preprocessing:
                results["warnings"].append("Missing 'label_column' in data.preprocessing for text type")
    
    # Check if model adapter has appropriate target_modules for the model type
    if ("model" in config and isinstance(config["model"], dict) and 
        "adapter" in config["model"] and isinstance(config["model"]["adapter"], dict)):
        adapter = config["model"]["adapter"]
        if ("type" in adapter and adapter["type"] == "lora" and 
            "name_or_path" in config["model"] and "distilgpt2" in config["model"]["name_or_path"]):
            if "target_modules" not in adapter:
                results["warnings"].append("Missing 'target_modules' for LoRA adapter")
            elif not isinstance(adapter["target_modules"], list):
                results["warnings"].append("'target_modules' should be a list")
            elif not any(module in ["c_attn", "c_proj"] for module in adapter["target_modules"]):
                results["warnings"].append("For GPT-2 models, target_modules should include 'c_attn' and/or 'c_proj'")
    
    # Check if dataset section has valid loading method
    if ("data" in config and isinstance(config["data"], dict) and 
        "dataset" in config["data"] and isinstance(config["data"]["dataset"], dict)):
        dataset = config["data"]["dataset"]
        if "load_from_hub" not in dataset and "load_from_disk" not in dataset:
            results["warnings"].append("Dataset configuration should specify either 'load_from_hub' or 'load_from_disk'")
    
    # Determine if the configuration is valid
    is_valid = (
        len(results["missing_keys"]) == 0 and 
        len(results["invalid_types"]) == 0
    )
    
    return is_valid, results

def print_validation_results(is_valid: bool, results: Dict[str, Any]) -> None:
    """Print the validation results in a user-friendly format."""
    if is_valid and len(results["warnings"]) == 0:
        logger.info("✅ Configuration is valid!")
        return
    
    if not is_valid:
        logger.error("❌ Configuration is invalid!")
    elif len(results["warnings"]) > 0:
        logger.warning("⚠️ Configuration is valid but has warnings!")
    
    # Print missing keys
    if len(results["missing_keys"]) > 0:
        logger.error("Missing required keys:")
        for key in results["missing_keys"]:
            logger.error(f"  - {key}")
    
    # Print invalid types
    if len(results["invalid_types"]) > 0:
        logger.error("Invalid types:")
        for key, expected, actual in results["invalid_types"]:
            logger.error(f"  - {key}: Expected {expected}, got {actual}")
    
    # Print warnings
    if len(results["warnings"]) > 0:
        logger.warning("Warnings:")
        for warning in results["warnings"]:
            logger.warning(f"  - {warning}")

def main():
    """Run the validation script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate a configuration file for the LLM finetuning pipeline')
    parser.add_argument('config_file', help='Path to the configuration file')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix common issues (not implemented yet)')
    args = parser.parse_args()
    
    logger.info(f"Validating configuration file: {args.config_file}")
    
    try:
        # Load the configuration
        config = load_config(args.config_file)
        
        # Validate the configuration
        is_valid, results = validate_config(config)
        
        # Print the results
        print_validation_results(is_valid, results)
        
        # Exit with appropriate status code
        if not is_valid:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 