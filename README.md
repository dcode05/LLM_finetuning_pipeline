# LLM Finetuning Framework

A comprehensive framework for fine-tuning large language models with various strategies, hyperparameter optimization, and evaluation tools.

## Overview

This framework provides a modular approach to LLM finetuning, with components for:
- Data preprocessing and tokenization
- Model loading and adaptation
- Training with different techniques  
- Hyperparameter optimization
- Model evaluation and metrics
- Pipeline orchestration

## Features

- **Modular Architecture**: Each component is designed to be replaceable and extensible
- **Multiple Model Support**: Works with popular models like GPT-2, T5, BERT, and more
- **Adapter Integration**: Includes PEFT techniques like LoRA, QLoRA, and Adapters
- **Hyperparameter Optimization**: Supports grid search and Ray Tune for finding optimal training parameters
- **Comprehensive Evaluation**: Built-in metrics and evaluation procedures
- **Pipeline Approach**: Streamlined process from data preparation to model evaluation
- **Validation Tools**: Configuration validators and environment diagnostics to help troubleshoot issues

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/llm-finetuning-framework.git
cd llm-finetuning-framework

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from pipeline.builder import create_pipeline
from pipeline.executor import execute_pipeline
import json

# Load configuration
with open('configs/example_config.json', 'r') as f:
    config = json.load(f)

# Create the pipeline
pipeline = create_pipeline(config)

# Execute the pipeline
results = execute_pipeline(pipeline, config)

# Print evaluation results
print(results['evaluation'])
```

## Example

See the `example.py` script for a complete demonstration of how to use the framework:

```bash
python example.py
```

## Configuration

The framework is configured through a JSON configuration file. Here's an example configuration:

```json
{
  "output_dir": "outputs/example_run",
  "data": {
    "preprocessing": {
      "type": "text",
      "max_length": 512,
      "text_column": "text",
      "label_column": "label"
    },
    "tokenization": {
      "type": "huggingface",
      "model_name_or_path": "gpt2",
      "padding": "max_length",
      "truncation": true,
      "max_length": 512
    },
    "dataset": {
      "type": "huggingface",
      "name": "imdb",
      "load_from_hub": true,
      "split": {
        "train": "train[:1000]",
        "validation": "test[:200]",
        "test": "test[200:400]"
      }
    }
  },
  "model": {
    "name_or_path": "gpt2",
    "type": "huggingface",
    "adapter": {
      "type": "lora",
      "r": 8,
      "alpha": 16,
      "dropout": 0.05,
      "target_modules": ["c_attn", "c_proj"]
    }
  },
  "training": {
    "type": "huggingface",
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "lr_scheduler": "linear",
    "warmup_steps": 100,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "logging_steps": 50
  },
  "evaluation": {
    "type": "huggingface",
    "metrics": ["accuracy", "f1"]
  },
  "hpo": {
    "enabled": true,
    "type": "grid",
    "max_evals": 4,
    "metric": "eval_loss",
    "direction": "minimize",
    "parameters": {
      "learning_rate": [1e-5, 5e-5, 1e-4],
      "weight_decay": [0.0, 0.01],
      "batch_size": [4, 8]
    }
  }
}
```

## Validation and Diagnostics

Before running a full training job, it's recommended to validate your setup:

```bash
# Check your configuration file for errors
python validation/validate_config.py your_config.json

# Diagnose your system and environment
python validation/diagnose_environment.py

# Run a quick test with synthetic data
python validation/run_validation.py --config quick_test_config.json
```

The validation suite includes:

- **Configuration Validator**: Checks your JSON config files for structural errors and common mistakes
- **Environment Diagnostics**: Verifies your dependencies, GPU setup, and system resources
- **Synthetic Data Tests**: Runs a minimal end-to-end test with generated data
- **Troubleshooting Guide**: Comprehensive documentation of common issues and solutions
- **Quick Start Configs**: Pre-made configurations for testing different aspects of the framework

For detailed information, see the [validation README](validation/README.md) and the [troubleshooting guide](validation/TROUBLESHOOTING.md).

## Components

### Data Processing

- **Preprocessors**: Transform raw data into model-ready format
- **Tokenizers**: Convert text to token IDs
- **Dataset Loaders**: Load and prepare datasets from various sources

### Models

- **Model Loaders**: Load pre-trained models from HuggingFace or local files
- **Adapters**: Apply efficient finetuning methods like LoRA

### Training

- **Trainers**: Handle the training process with various optimization strategies
- **Hyperparameter Optimization**: Find optimal training parameters

### Evaluation

- **Evaluators**: Assess model performance
- **Metrics**: Measure model quality with standard and custom metrics

## Hyperparameter Optimization

The framework supports multiple HPO strategies:

### Grid Search

Simple grid search over the parameter space:

```json
"hpo": {
  "type": "grid",
  "max_evals": 4,
  "search_space": {
    "learning_rate": [1e-5, 5e-5, 1e-4],
    "weight_decay": [0.0, 0.01]
  }
}
```

### Ray Tune

Advanced HPO using Ray Tune with various search algorithms:

```json
"hpo": {
  "type": "ray_tune",
  "search_algorithm": "bayesopt",
  "num_samples": 10,
  "max_concurrent_trials": 2,
  "scheduler": "asha",
  "search_space": {
    "learning_rate": ["float", 1e-5, 1e-3, "log"],
    "weight_decay": ["float", 0.0, 0.1]
  }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 