#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Finetuning Framework
========================

A comprehensive framework for fine-tuning large language models with various strategies,
hyperparameter optimization, and evaluation tools.

This framework provides a modular approach to LLM finetuning, with components for:
- Data preprocessing and tokenization
- Model loading and adaptation
- Training with different techniques
- Hyperparameter optimization
- Model evaluation and metrics
- Pipeline orchestration
"""

# Import pipeline components
from pipeline.builder import create_pipeline, Pipeline
from pipeline.executor import execute_pipeline

# Import factory functions for creating components
from data.preprocessing.factory import create_preprocessor
from data.tokenization.factory import create_tokenizer
from data.dataset.factory import create_dataset_loader
from models.factory import create_model, create_adapter
from training.factory import create_trainer
from evaluation.factory import create_evaluator
from hpo.factory import create_hpo_optimizer

# Make everything available at the top level
__all__ = [
    # Pipeline
    'Pipeline',
    'create_pipeline',
    'execute_pipeline',
    
    # Component factories
    'create_preprocessor',
    'create_tokenizer',
    'create_dataset_loader',
    'create_model',
    'create_adapter',
    'create_trainer',
    'create_evaluator',
    'create_hpo_optimizer',
]

# Version information
__version__ = '0.1.0' 