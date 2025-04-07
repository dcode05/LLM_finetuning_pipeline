#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for installing the minimal required dependencies for validation tests.
"""

import subprocess
import sys
import os
import platform
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Minimal dependencies needed for validation
MINIMAL_DEPENDENCIES = [
    "numpy>=1.20.0",
    "torch>=1.10.0",
    "transformers>=4.26.0",
    "datasets>=2.10.0",
    "peft>=0.3.0",
    "evaluate>=0.4.0",
    "scikit-learn>=1.0.0",
]

# Optional performance enhancement packages
PERFORMANCE_DEPENDENCIES = [
    "huggingface_hub[hf_xet]>=0.15.0",
]

def check_pip():
    """Check if pip is installed and accessible."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, stdout=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("pip is not installed or not in PATH. Please install pip first.")
        return False

def install_dependencies(include_performance=False):
    """
    Install the required dependencies.
    
    Args:
        include_performance: Whether to include optional performance enhancement packages
    """
    if not check_pip():
        return False
    
    logger.info("Installing minimal required dependencies...")
    
    # Create a list of all dependencies to install
    all_dependencies = MINIMAL_DEPENDENCIES[:]
    
    if include_performance:
        logger.info("Including optional performance enhancement packages...")
        all_dependencies.extend(PERFORMANCE_DEPENDENCIES)
    
    # Create a temporary requirements file
    temp_req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_requirements.txt")
    with open(temp_req_file, "w") as f:
        f.write("\n".join(all_dependencies))
    
    try:
        # Install dependencies
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", temp_req_file],
            check=True
        )
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_req_file):
            os.remove(temp_req_file)

def main():
    """Run the setup process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Setup validation environment')
    parser.add_argument('--performance', action='store_true',
                        help='Install optional performance enhancement packages')
    args = parser.parse_args()
    
    logger.info("Starting validation setup...")
    logger.info(f"Python version: {platform.python_version()}")
    
    if install_dependencies(include_performance=args.performance):
        logger.info("\nSetup completed successfully!")
        if args.performance:
            logger.info("Performance enhancement packages installed for faster downloads.")
        else:
            logger.info("Note: For faster model downloads, you can re-run with --performance flag")
        logger.info("You can now run the validation using:")
        logger.info("python validation/run_validation.py --config quick_test_config.json")
    else:
        logger.error("\nSetup failed. Please try to install dependencies manually.")
        logger.info("Required packages:")
        for dep in MINIMAL_DEPENDENCIES:
            logger.info(f"  - {dep}")
        if args.performance:
            logger.info("Optional performance packages:")
            for dep in PERFORMANCE_DEPENDENCIES:
                logger.info(f"  - {dep}")

if __name__ == "__main__":
    main() 