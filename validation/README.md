# Validation Tests

This directory contains scripts and configurations for validating the LLM finetuning pipeline using synthetic data.

## Files

- `generate_synthetic_data.py`: Script to generate synthetic movie review data
- `synthetic_test_config.json`: Configuration file for running the pipeline with synthetic data
- `quick_test_config.json`: Simplified configuration for faster testing
- `test_pipeline.py`: Script to run the pipeline with synthetic data
- `run_validation.py`: Main script that runs the entire validation process
- `setup_validation.py`: Script to install required dependencies
- `diagnose_environment.py`: Script to diagnose issues with the environment
- `validate_config.py`: Script to validate configuration files and identify issues
- `TROUBLESHOOTING.md`: Comprehensive guide to fixing common issues
- `CHANGELOG.md`: History of changes and fixes to the framework

## Installation

Before running the validation tests, you need to install the required dependencies. You can do this by running:

```bash
python validation/setup_validation.py
```

This will install the minimal required packages:
- numpy
- torch
- transformers
- datasets
- peft
- evaluate
- scikit-learn

Alternatively, you can install all dependencies using the main requirements file:

```bash
pip install -r requirements.txt
```

### Optional Performance Enhancement

For faster downloads from Hugging Face Hub, you can install the Xet Storage package:

```bash
pip install huggingface_hub[hf_xet]
```

This is completely optional but can significantly improve download speeds for models and datasets.

## Running the Tests

### Option 1: Run the entire validation process with a single command

```bash
# Run with the default configuration (more comprehensive)
python validation/run_validation.py

# Run with the quick test configuration (faster)
python validation/run_validation.py --config quick_test_config.json

# Skip data generation if you've already generated the data
python validation/run_validation.py --skip-data-generation
```

This script will:
1. Generate the synthetic dataset (unless skipped)
2. Run the pipeline with the synthetic data 
3. Log the results

### Option 2: Run each step individually

1. First, generate the synthetic dataset:
```bash
python validation/generate_synthetic_data.py
```
This will create a `data/synthetic` directory with train, validation, and test splits.

2. Run the pipeline with the synthetic data:
```bash
# Run with the default configuration
python validation/test_pipeline.py

# Run with the quick test configuration
python validation/test_pipeline.py --config quick_test_config.json
```

## Configuration Validation

Before running the pipeline, you can validate your configuration files to ensure they have the correct structure and types:

```bash
python validation/validate_config.py validation/quick_test_config.json
```

This will check for:
- Missing required keys
- Incorrect data types
- Invalid or inconsistent values
- Common configuration issues

The validator will provide detailed feedback and warnings to help you fix any problems before running the pipeline.

## Configuration Options

- `synthetic_test_config.json`: Standard configuration with more examples
  - 2 training epochs
  - Medium sequence length (256)
  - Uses distilgpt2 model with LoRA

- `quick_test_config.json`: Faster configuration for quick testing
  - 1 training epoch
  - Smaller sequence length (128)
  - Uses distilgpt2 model with LoRA

## Expected Output

The pipeline will:
1. Load and preprocess the synthetic movie reviews
2. Fine-tune a distilgpt2 model with LoRA adaptation
3. Evaluate the model's performance
4. Save the results and model checkpoints

## Notes

- The synthetic dataset contains 1000 samples split into:
  - 700 training samples
  - 150 validation samples
  - 150 test samples
- The model is fine-tuned with a small batch size for quick testing
- Both configurations use DistilGPT-2 for faster training

## Troubleshooting

If you encounter errors, there are several resources to help:

1. **Run the validation tool** to check your configuration:
```bash
python validation/validate_config.py your_config.json
```

2. **Run the diagnostic tool** to identify environment issues:
```bash
python validation/diagnose_environment.py
```

3. **Check the troubleshooting guide** for solutions to common problems:
[TROUBLESHOOTING.md](TROUBLESHOOTING.md)

4. **Review the changelog** to see recent fixes:
[CHANGELOG.md](CHANGELOG.md)

Common issues include:
- Missing dependencies: Install them with `setup_validation.py`
- Model compatibility: We now use DistilGPT-2 with proper LoRA target modules
- Memory issues: Reduced batch size and sequence length in both configurations
- GPU availability: Works on CPU if no GPU is detected
- Performance warnings: Add `huggingface_hub[hf_xet]` for faster downloads
- Configuration structure: Use `validate_config.py` to check your configuration files

# Validation Framework Updates

This document summarizes the changes made to fix issues in the LLM finetuning validation framework.

## Overview of Fixes

We've made significant updates to address various issues encountered during validation testing:

1. **Configuration Structure Fixes**: 
   - Fixed configuration structure to match the pipeline's expectations
   - Updated nesting of components under the "data" section
   - Fixed field names to match the expected formats in each component

2. **Model Configuration Updates**:
   - Changed model field from `model_name_or_path` to `name_or_path`
   - Added proper model type specifications using both `type` and `model_type`
   - Fixed target modules for LoRA adapters to work with the GPT-2 architecture

3. **Dataset Loading Improvements**:
   - Fixed dataset loading configuration to correctly access local files
   - Updated JSON handling to properly read synthetic test dataset
   - Added appropriate text and label column specifications

4. **Pipeline Execution Enhancements**:
   - Improved model loading behavior to handle errors gracefully
   - Fixed HPO configuration to use the correct parameter naming
   - Added better error handling in key pipeline components
   - Enhanced type checking to prevent "'str' object has no attribute 'get'" errors

5. **Documentation Updates**:
   - Created a comprehensive troubleshooting guide
   - Added best practices for configuration file creation
   - Documented common errors and their solutions
   - Created a configuration validation tool

6. **Performance Optimizations**:
   - Added support for Xet Storage for faster Hugging Face Hub downloads
   - Disabled HPO initially to simplify validation and debugging
   - Reduced model size and sequence length for faster testing

## Summary of Changed Files

1. **Configuration Files**:
   - `quick_test_config.json`: Fixed structure, disabled HPO, updated model fields
   - `synthetic_test_config.json`: Fixed structure, disabled HPO, updated model fields

2. **Pipeline Components**:
   - `pipeline/executor.py`: Improved model loading, updated HPO handling, enhanced error logging
   - `models/factory.py`: Added debug logging to diagnose model loading issues

3. **Documentation**:
   - `TROUBLESHOOTING.md`: Comprehensive guide for resolving common issues
   - `README.md`: Updated with latest information (this file)
   - `CHANGELOG.md`: Record of all changes and fixes

4. **Utilities**:
   - `validate_config.py`: New tool to check configuration file structure and types
   - `setup_validation.py`: Updated to support optional performance packages

## Running the Validation Tests

To test the framework with the updated configuration:

```bash
python validation/test_pipeline.py --config quick_test_config.json
```

## Next Steps

1. **Further Testing**: Continue testing with more complex configurations
2. **Performance Optimization**: Re-enable HPO once basic functionality is verified
3. **Additional Models**: Test with different model architectures (BERT, RoBERTa, etc.)
4. **Custom Datasets**: Validate with various dataset formats

## Conclusion

The updated validation framework now has better error handling, clearer configuration requirements, and more comprehensive documentation. This should make it easier for users to get started with the LLM finetuning framework and avoid common pitfalls. 