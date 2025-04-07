# Troubleshooting Guide

This document helps diagnose and fix common issues with the LLM finetuning validation pipeline.

## Diagnostic Tool

First, run the diagnostic tool to identify potential issues:

```bash
python validation/diagnose_environment.py
```

This will check your environment, dependencies, and configuration files.

## Common Errors and Solutions

### ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'datasets'
```

**Solution:**
Run the setup script to install required dependencies:
```bash
python validation/setup_validation.py
```

### Tokenizer Type Warning

**Warning:**
```
WARNING - Unknown tokenizer type: auto
```

**Solution:**
This happens because the tokenizer factory only recognizes certain tokenizer types. We've updated the config files to use "huggingface" instead of "auto" or "pretrained" for the tokenizer type.

If you see this warning, check your configuration files and ensure:
- The tokenizer type is set to "huggingface" (not "auto" or "pretrained")
- The model_name_or_path is valid and points to a supported model

### Dataset Loading Warning

**Warning:**
```
WARNING - No valid dataset loading method specified in configuration
```

**Solution:**
This warning occurs because the dataset loader factory requires a specific loading method to be specified. We've updated the configs to use the correct structure:

```json
"data": {
  "dataset": {
    "load_from_disk": false,
    "dir": "data/synthetic",
    "train_file": "train.json",
    "validation_file": "validation.json",
    "test_file": "test.json",
    "dataset_format": "json"
  }
}
```

Make sure your dataset configuration correctly specifies either:
- `"load_from_disk": false` for loading from individual files
- `"load_from_hub": true` and `"hf_dataset_name": "dataset_name"` for loading from HuggingFace Hub 

### String Object Has No Attribute 'get' Error

**Error:**
```
ERROR - Error running pipeline: 'str' object has no attribute 'get'
```

**Solution:**
This error occurs when the pipeline is trying to use a string value as if it were a dictionary. This typically happens when:

1. A configuration section is provided as a string instead of an object/dictionary:
   ```json
   "data": "some_value"  // WRONG: This should be an object
   ```
   
   Instead, it should be:
   ```json
   "data": {  // CORRECT: This is an object
     "preprocessing": { ... },
     "dataset": { ... }
   }
   ```

2. A nested configuration is provided as a string:
   ```json
   "model": {
     "adapter": "lora"  // WRONG: This should be an object
   }
   ```
   
   Instead, it should be:
   ```json
   "model": {
     "adapter": {  // CORRECT: This is an object
       "type": "lora",
       "r": 8
     }
   }
   ```

3. A list is provided where an object is expected:
   ```json
   "hpo": ["learning_rate", "batch_size"]  // WRONG: This should be an object
   ```
   
   Instead, it should be:
   ```json
   "hpo": {  // CORRECT: This is an object
     "enabled": true,
     "parameters": { ... }
   }
   ```

To fix this error:
1. Check that all sections of your configuration are properly defined as JSON objects (with curly braces `{}`) and not strings or arrays
2. Ensure that all required nested configuration objects are properly specified
3. Verify that you're not using shorthand notation for complex configuration objects

The pipeline has been updated with better error handling to pinpoint where exactly this error is happening.

### Model Type Warning

**Warning:**
```
WARNING - Unknown model type: causal_lm
```

**Solution:**
The model factory only recognizes specific model types. Model types are specified in two places:
1. The `type` field specifies the model framework (e.g., "huggingface")
2. The `model_type` field specifies the architecture (e.g., "causal_lm" or "seq2seq_lm")

Your configuration should look like:
```json
"model": {
  "name_or_path": "distilgpt2",
  "model_type": "causal_lm", 
  "type": "huggingface"
}
```

### Xet Storage Warning

**Warning:**
```
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
```

**Solution:**
This warning indicates that the Hugging Face model or dataset repository you're accessing uses Xet Storage for more efficient downloads, but you don't have the required package installed. This doesn't break functionality but may affect download speed.

To improve performance, install the recommended package:
```bash
pip install huggingface_hub[hf_xet]
```
or
```bash
pip install hf_xet
```

You can add this to your `requirements.txt` file or install it separately. The framework will still work without it, but downloads from Hugging Face Hub may be slower.

### NoneType Object Error or 'No model loaded' Error

**Error:**
```
ERROR - 'NoneType' object has no attribute 'model'
```

or 

```
ERROR - No model loaded. Call load_model() first.
```

**Solution:**
This error occurs when one of the components (e.g., model, tokenizer) failed to initialize properly due to configuration issues. Make sure:

1. The model configuration is correct:
   ```json
   "model": {
     "name_or_path": "distilgpt2",  // Use name_or_path not model_name_or_path
     "model_type": "causal_lm",
     "type": "huggingface",
     "adapter": { ... }
   }
   ```

2. The configuration structure follows this pattern:
   ```json
   {
     "output_dir": "...",
     "data": {
       "preprocessing": { ... },
       "tokenization": { ... },
       "dataset": { ... }
     },
     "model": { ... },
     "training": { ... },
     "evaluation": { ... },
     "hpo": { ... }
   }
   ```

3. Your config has the correct nested structure, especially the "data" field.

### Target Modules for LoRA

**Error:**
```
ValueError: Target modules ['q_proj', 'v_proj'] not found in the base model.
```

**Solution:**
This happens because the target modules in the config don't match the architecture of the model. We've fixed this in the config files by:

1. Using `distilgpt2` instead of `gpt2` (smaller and faster)
2. Setting the correct target modules: `["c_attn", "c_proj"]` for GPT-2 models

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
We've fixed this in the config files by:
- Reducing batch size from 8 to 4
- Reducing sequence length from 512 to 256 or 128
- Disabling mixed precision training (`"fp16": false`)
- Disabling hyperparameter optimization (`"enabled": false`)

### Unrecognized Arguments

**Error:**
```
TypeError: __init__() got an unexpected keyword argument 'XXX'
```

**Solution:**
This usually happens because the config contains parameters that aren't supported by the model or trainer. Our updates have removed any potentially problematic parameters.

## Configuration Structure

The pipeline expects a specific configuration structure:

```json
{
  "output_dir": "path/to/outputs",
  "data": {
    "preprocessing": {
      "type": "text",
      "max_length": 128,
      "text_column": "text",
      "label_column": "label"
    },
    "tokenization": {
      "type": "huggingface",
      "model_name_or_path": "distilgpt2",
      "padding": "max_length",
      "truncation": true,
      "max_length": 128
    },
    "dataset": {
      "load_from_disk": false,
      "dir": "data/synthetic",
      "train_file": "train.json",
      "validation_file": "validation.json",
      "test_file": "test.json",
      "dataset_format": "json"
    }
  },
  "model": {
    "name_or_path": "distilgpt2",
    "model_type": "causal_lm",
    "type": "huggingface",
    "adapter": {
      "type": "lora",
      "r": 8,
      "lora_alpha": 32,
      "target_modules": ["c_attn", "c_proj"],
      "lora_dropout": 0.1
    }
  },
  "training": {
    "epochs": 2,
    "batch_size": 4,
    "optimizer": "adamw",
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "fp16": false,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "is_regression": false
  },
  "evaluation": {
    "metrics": ["accuracy", "f1"],
    "batch_size": 4
  },
  "hpo": {
    "enabled": false,
    "max_trials": 4,
    "strategy": "grid_search",
    "parameters": {
      "learning_rate": ["1e-5", "5e-5"],
      "batch_size": [2, 4]
    }
  }
}
```

## Best Practices for Configuration Files

To avoid common issues with the framework, follow these best practices:

1. **Correct Structure**: Ensure all components are correctly nested within their appropriate sections (data, model, training, etc.)

2. **Model Configuration**:
   - Always specify `name_or_path` (not `model_name_or_path`) for the model
   - Include both `type` (framework) and `model_type` (architecture type)
   - Use correct target modules for adapters depending on model architecture

3. **Dataset Configuration**:
   - Always specify a valid loading method (`load_from_disk` or `load_from_hub`)
   - For local files, provide correct paths to data files
   - Include text_column and label_column names that match your data

4. **Performance Optimizations**:
   - Start with `"fp16": false` to avoid precision issues
   - Keep batch sizes small (4 or less) for initial tests
   - Disable HPO for initial testing with `"enabled": false`
   - Use smaller sequence lengths (128 or 256) to reduce memory usage

5. **Gradual Testing**:
   - Start with the `quick_test_config.json` to verify basic functionality
   - Test components individually before running the full pipeline
   - Increase complexity gradually (larger models, HPO, etc.)

6. **Field Names**: Use the exact field names expected by the framework:
   - HPO: "enabled", "max_trials", "parameters" (not "search_space")
   - Model: "name_or_path" (not "model_name" or "model_path")
   - Adapter: "lora_alpha" and "lora_dropout" (not just "alpha" and "dropout")

7. **Value Types**: Make sure all values have the correct type:
   - Objects/dictionaries should be enclosed in curly braces `{}`
   - Arrays/lists should be enclosed in square brackets `[]`
   - Strings should be enclosed in quotes `""`
   - Booleans should be lowercase `true` or `false` (not quoted)
   - Numbers should not be enclosed in quotes unless they are meant to be strings

## If Issues Persist

If you still encounter issues after applying these fixes:

1. **Try the Minimal Test Config:**
   ```bash
   python validation/test_pipeline.py --config quick_test_config.json
   ```

2. **Run With Verbose Logging:**
   Add this at the top of your Python scripts to see more detailed error messages:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Common Model-Specific Issues:**

   - **GPT-2 Models:**
     - Make sure target_modules are `["c_attn", "c_proj"]`
     - These models require more memory, so use smaller sequence lengths

   - **RoBERTa/BERT Models:**
     - Change target_modules to `["query", "key", "value"]`
     - Update model_type to "masked_lm" instead of "causal_lm"

4. **Known Package Conflicts:**
   - Ensure transformers, peft, and datasets versions are compatible
   - Try specifically: transformers==4.28.0, peft==0.3.0, datasets==2.10.0

## Changes Made to Config Files

We've made several changes to the configuration files to address common issues:

1. **Model Changes:**
   - Changed from `gpt2` to `distilgpt2` (smaller and faster)
   - Specified `model_type` as "causal_lm" and `type` as "huggingface"
   - Updated target modules for LoRA
   - Changed `model_name_or_path` to `name_or_path` to match expected structure

2. **Training Changes:**
   - Reduced batch size
   - Reduced sequence length
   - Disabled fp16 training
   - Fewer epochs
   - Shorter warmup period

3. **Tokenization Changes:**
   - Changed tokenizer type from "auto" to "huggingface" to match what's expected in the factory
   - Fixed structure under "data" > "tokenization"

4. **Dataset Changes:**
   - Set `"load_from_disk": false`
   - Fixed structure to be under "data" > "dataset"
   - Added proper train/validation/test file references

5. **HPO Changes:**
   - Fixed parameter names to match expected format
   - Simplified search space
   - Disabled HPO initially to debug other components

6. **Overall Structure:**
   - Fixed nesting of components under "data" section
   - Ensured proper field names match what's expected by the framework components

These changes should make the validation process more stable and less resource-intensive. 

CRITICAL ISSUES FOUND:
✗ Missing required packages: scikit-learn 

✓ Synthetic data directory exists     
✗ train split
✗ validation split
✗ test split 

Low memory: 2.18 GB available (min recommended: 4 GB) 