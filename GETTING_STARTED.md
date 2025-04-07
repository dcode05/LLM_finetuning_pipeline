# Getting Started with the LLM Finetuning Framework

This guide will help you understand how to use the LLM Finetuning Framework, validate your setup, and get started with your first finetuning task.

## Contents

1. [Setup and Installation](#setup-and-installation)
2. [Validating Your Environment](#validating-your-environment)
3. [Understanding the Framework Components](#understanding-the-framework-components)
4. [Creating Your First Configuration](#creating-your-first-configuration)
5. [Running a Finetuning Job](#running-a-finetuning-job)
6. [Troubleshooting Common Issues](#troubleshooting-common-issues)
7. [Next Steps](#next-steps)

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended but not required)
- 8GB+ RAM (16GB+ recommended for larger models)
- 20GB+ disk space for models and datasets

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm-finetuning-framework.git
   cd llm-finetuning-framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   For minimal installation (validation only):
   ```bash
   python validation/setup_validation.py
   ```

3. Verify installation:
   ```bash
   python validation/diagnose_environment.py
   ```

## Validating Your Environment

The framework includes several tools to validate your environment and configurations before running expensive training jobs:

### 1. Environment Diagnostics

Run the diagnostics tool to check your system, dependencies, and GPU setup:

```bash
python validation/diagnose_environment.py
```

This will check:
- Required and optional package versions
- GPU availability and CUDA setup
- System resources (RAM, disk space)
- Workspace structure and permissions
- HuggingFace cache status

Review the output for any critical issues or warnings that need to be addressed.

### 2. Configuration Validation

Before running a finetuning job, validate your configuration file:

```bash
python validation/validate_config.py path/to/your_config.json
```

This helps catch common configuration issues:
- Missing required fields
- Incorrect data types
- Invalid nested structures
- Common configuration mistakes

### 3. Quick Validation Run

Test the full pipeline with synthetic data to verify everything works end-to-end:

```bash
# Generate synthetic data (if not already created)
python validation/generate_synthetic_data.py

# Run a quick test
python validation/run_validation.py --config quick_test_config.json
```

This will:
1. Load a small synthetic dataset
2. Initialize a small model with LoRA adapter
3. Run a single training epoch
4. Evaluate and report metrics

## Understanding the Framework Components

The framework is modular and consists of several key components:

### Data Pipeline Components

- **Preprocessors** (`data/preprocessing/`): Transform raw data into a format suitable for tokenization.
- **Tokenizers** (`data/tokenization/`): Convert text data into token IDs that models can process.
- **Dataset Loaders** (`data/dataset/`): Load datasets from various sources (HuggingFace Hub, local files, etc.).

### Model Components

- **Model Loaders** (`models/`): Load pre-trained models and prepare them for finetuning.
- **Adapters** (`models/adapters/`): Apply parameter-efficient finetuning techniques like LoRA.

### Training Components

- **Trainers** (`training/`): Handle the training process with various optimization strategies.
- **Hyperparameter Optimization** (`hpo/`): Find optimal training parameters using strategies like grid search or Bayesian optimization.

### Evaluation Components

- **Evaluators** (`evaluation/`): Assess model performance on validation and test data.
- **Metrics** (`evaluation/metrics/`): Calculate standard and custom performance metrics.

### Pipeline Orchestration

- **Pipeline Builder** (`pipeline/builder.py`): Assembles the components based on configuration.
- **Pipeline Executor** (`pipeline/executor.py`): Orchestrates the execution of the pipeline components.

## Creating Your First Configuration

The framework uses JSON configuration files to define the entire finetuning process. Start by modifying one of the example configurations:

```bash
cp validation/quick_test_config.json configs/my_first_config.json
```

Then edit `configs/my_first_config.json` to customize:

1. **Dataset:** Choose between loading from Hugging Face Hub or local files.
2. **Model:** Select a pretrained model to finetune.
3. **Adapter:** Configure parameter-efficient finetuning approach (LoRA, QLoRA, etc.).
4. **Training:** Set hyperparameters like learning rate, batch size, and epochs.
5. **Evaluation:** Define metrics to assess performance.
6. **HPO (optional):** Configure hyperparameter optimization.

Here's a simple example for sentiment classification:

```json
{
  "output_dir": "outputs/sentiment_classifier",
  "data": {
    "preprocessing": {
      "type": "text",
      "max_length": 512,
      "text_column": "text",
      "label_column": "label"
    },
    "tokenization": {
      "type": "huggingface",
      "model_name_or_path": "distilbert-base-uncased",
      "padding": "max_length",
      "truncation": true,
      "max_length": 512
    },
    "dataset": {
      "type": "huggingface",
      "name": "imdb",
      "load_from_hub": true,
      "split": {
        "train": "train[:5000]",
        "validation": "test[:500]",
        "test": "test[500:1000]"
      }
    }
  },
  "model": {
    "name_or_path": "distilbert-base-uncased",
    "type": "huggingface",
    "adapter": {
      "type": "lora",
      "r": 8,
      "alpha": 16,
      "dropout": 0.05,
      "target_modules": ["query", "key", "value"]
    }
  },
  "training": {
    "type": "huggingface",
    "epochs": 3,
    "batch_size": 16,
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
    "enabled": false
  }
}
```

Always validate your configuration before running:

```bash
python validation/validate_config.py configs/my_first_config.json
```

## Running a Finetuning Job

Once your configuration is ready, you can run the finetuning job using the example script:

```bash
python example.py --config configs/my_first_config.json
```

Or create a custom script:

```python
from pipeline.builder import create_pipeline
from pipeline.executor import execute_pipeline
import json
import argparse
import time

# Parse arguments
parser = argparse.ArgumentParser(description="Run LLM finetuning")
parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as f:
    config = json.load(f)

# Create pipeline
pipeline = create_pipeline(config)

# Execute pipeline and time it
start_time = time.time()
results = execute_pipeline(pipeline, config)
elapsed_time = time.time() - start_time

# Print results
print(f"Training completed in {elapsed_time:.2f} seconds")
print(f"Evaluation results:")
for metric, value in results.get('evaluation', {}).items():
    print(f"  {metric}: {value}")

# If HPO was enabled, print best parameters
if config.get('hpo', {}).get('enabled', False):
    print(f"Best hyperparameters:")
    for param, value in results.get('hpo', {}).get('best_params', {}).items():
        print(f"  {param}: {value}")
```

## Troubleshooting Common Issues

If you encounter issues, follow these steps:

1. Run the diagnostics tool:
   ```bash
   python validation/diagnose_environment.py
   ```

2. Validate your configuration:
   ```bash
   python validation/validate_config.py path/to/your_config.json
   ```

3. Check the troubleshooting guide:
   ```bash
   cat validation/TROUBLESHOOTING.md
   ```

4. Common issues include:
   - **Out-of-memory errors**: Reduce batch size, model size, or sequence length
   - **Dataset loading errors**: Verify dataset paths and loading method
   - **Tokenizer mismatches**: Ensure tokenizer matches the model type
   - **Missing target modules**: Check target modules match model architecture
   - **Configuration structure issues**: Use the validator to identify problems

## Next Steps

Once you've successfully run your first finetuning job:

1. **Experiment with different models**: Try models like GPT-2, RoBERTa, T5, etc.
2. **Explore adapter techniques**: Compare LoRA, QLoRA, and other PEFT methods
3. **Use hyperparameter optimization**: Enable HPO to find optimal parameters
4. **Custom datasets**: Prepare and load your own datasets
5. **Advanced configurations**: Explore more advanced training and evaluation options
6. **Extend the framework**: Add new components for your specific needs

Refer to the full API documentation for more advanced usage and customization options.

## Additional Resources

- [Framework Documentation](README.md)
- [Validation Suite](validation/README.md)
- [Troubleshooting Guide](validation/TROUBLESHOOTING.md)
- [Example Configurations](validation/examples/)
- [Changelog](validation/CHANGELOG.md) 