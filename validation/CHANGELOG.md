# LLM Finetuning Framework Changelog

## Version 0.2.1 (2025-04-07)

### Bug Fixes
- Fixed "'str' object has no attribute 'get'" error by adding robust type checking and error handling
- Enhanced pipeline executor with better error detection and reporting
- Added comprehensive validation for configuration objects before use
- Updated troubleshooting guide with information about proper JSON configuration syntax
- Added guidance on correct value types (objects, arrays, strings, booleans, numbers) in configuration

### Code Improvements
- Added detailed logging to help diagnose configuration issues
- Improved error messages to better pinpoint the source of problems
- Added graceful failure paths to prevent cascading errors
- Enhanced validation of nested configuration structures

### New Features
- Added `validate_config.py` utility to check configuration files before pipeline execution
- Enhanced `diagnose_environment.py` with comprehensive system and dependency checks
- Updated documentation to include information about the new validation tools
- Added more examples of correct configuration structure in the troubleshooting guide

## Version 0.2.0 (2025-04-07)

### Major Changes

#### Configuration Structure Updates
- Reorganized configuration structure to ensure consistent nesting and field naming
- Fixed the data section to properly group preprocessing, tokenization, and dataset configurations
- Updated model configuration to use `name_or_path` instead of `model_name_or_path`
- Corrected field names in HPO configuration to use `parameters` instead of `search_space`

#### Pipeline Execution Enhancements
- Improved model loading process in `pipeline/executor.py` to handle errors gracefully
- Added debug logging to help diagnose configuration and loading issues
- Fixed hyperparameter optimization to properly update training config with best parameters
- Made HPO optional via the `enabled` flag for better control over validation process

#### Error Handling Improvements
- Added better error reporting throughout the pipeline execution process
- Enhanced logging to provide clearer insights into what's happening during execution
- Fixed common issues that would cause pipeline failures (model loading, dataset access, etc.)

### Specific Fixes

#### Tokenizer Type Fix
- Changed tokenizer type from "auto" to "huggingface" to match what's expected by the factory
- Added validation to ensure tokenizer's `model_name_or_path` matches the model being used

#### Dataset Loading Fix 
- Added `load_from_disk: false` to correctly handle loading datasets from individual files
- Fixed JSON handling to properly read synthetic dataset files
- Added proper text_column and label_column specifications in configuration

#### Model Configuration Fix
- Fixed model type specification to include both framework type and architecture type
- Updated target modules for LoRA to match the GPT-2 architecture (`c_attn` and `c_proj`)
- Changed from `gpt2` to `distilgpt2` for reduced memory usage and faster testing

#### Performance Optimizations
- Added support for Xet Storage (`huggingface_hub[hf_xet]`) for faster downloads
- Reduced sequence length and batch size for lower memory usage
- Disabled FP16 training by default to avoid precision issues during validation
- Made HPO optional and disabled by default for initial testing

### Documentation Updates
- Created comprehensive [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide
- Updated [README.md](README.md) with latest information and best practices
- Added configuration structure examples to help users create valid configurations
- Created a field-by-field guide to configuration options and their expected values

### Utility Script Improvements
- Enhanced `setup_validation.py` to include optional performance packages
- Added `--performance` flag to install Xet Storage support for faster downloads
- Improved error handling and feedback during setup process

## Version 0.1.0 (2025-04-06)

- Initial release of the LLM Finetuning Framework
- Basic support for tokenization, model loading, training, and evaluation
- Initial configuration files and validation scripts 