#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HuggingFace evaluator implementation.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Union

import numpy as np
import torch
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, default_data_collator
import evaluate

from evaluation.base import BaseEvaluator
from evaluation.metrics import load_metric


class HuggingFaceEvaluator(BaseEvaluator):
    """
    Evaluator implementation using HuggingFace's Transformers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace evaluator.
        
        Args:
            config: Configuration dictionary for the evaluator
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        
        # Load metrics
        self.metrics_config = config.get("metrics", ["accuracy"])
        self.metrics = {}
        for metric_name in self.metrics_config:
            try:
                self.metrics[metric_name] = load_metric(metric_name)
            except Exception as e:
                self.logger.warning(f"Failed to load metric {metric_name}: {e}")
    
    def evaluate(self, model, tokenizer, dataset: Optional[DatasetDict] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            dataset: The dataset to evaluate on
            config: Optional configuration override
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        config = config or self.config
        self.model = model
        self.tokenizer = tokenizer
        
        self.logger.info("Setting up evaluation")
        
        # Get evaluation dataset
        if dataset is None:
            self.logger.error("No dataset provided for evaluation")
            return {"error": "No dataset provided"}
        
        eval_dataset = dataset.get("validation") or dataset.get("test")
        if eval_dataset is None:
            self.logger.error("No validation or test split found in dataset")
            return {"error": "No validation/test split found"}
        
        # Set up evaluation arguments
        output_dir = config.get("output_dir", "outputs/evaluation")
        os.makedirs(output_dir, exist_ok=True)
        
        generation_config = config.get("generation_config", {})
        
        # Create Trainer for evaluation
        trainer_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=config.get("batch_size", 8),
            predict_with_generate=True if generation_config else False,
        )
        
        trainer = Trainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=self._compute_metrics_wrapper,
        )
        
        # Evaluate
        self.logger.info("Starting evaluation")
        start_time = time.time()
        
        if generation_config:
            # Generation-based evaluation
            metrics = self._evaluate_with_generation(
                model, tokenizer, eval_dataset, generation_config
            )
        else:
            # Standard evaluation
            metrics = trainer.evaluate(eval_dataset)
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        metrics["eval_time"] = eval_time
        
        self.logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        self.logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def compute_metrics(self, predictions, references) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        metrics = {}
        
        # Compute metrics based on configuration
        for metric_name, metric in self.metrics.items():
            try:
                metric_value = metric.compute(predictions=predictions, references=references)
                metrics.update(metric_value)
            except Exception as e:
                self.logger.warning(f"Failed to compute metric {metric_name}: {e}")
        
        return metrics
    
    def _compute_metrics_wrapper(self, eval_preds):
        """
        Wrapper for compute_metrics to handle transformers' format.
        
        Args:
            eval_preds: Evaluation predictions from the trainer
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        predictions, labels = eval_preds
        
        # Handle different prediction formats
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Get predicted class or token IDs
        if predictions.shape[-1] > 1:
            # Classification task
            predicted_classes = np.argmax(predictions, axis=-1)
        else:
            # Already class indices
            predicted_classes = predictions
        
        # Replace -100 padding in labels with something metrics can handle
        labels = np.where(labels != -100, labels, 0)
        
        return self.compute_metrics(predicted_classes, labels)
    
    def _evaluate_with_generation(self, model, tokenizer, eval_dataset, generation_config):
        """
        Evaluate the model with text generation.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use
            eval_dataset: The dataset to evaluate on
            generation_config: Generation configuration
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        self.logger.info("Evaluating with text generation")
        self.logger.info(f"Generation config: {generation_config}")
        
        # Set generation parameters
        max_new_tokens = generation_config.get("max_new_tokens", 20)
        do_sample = generation_config.get("do_sample", True)
        temperature = generation_config.get("temperature", 0.7)
        top_p = generation_config.get("top_p", 0.9)
        top_k = generation_config.get("top_k", 50)
        
        # Get input and output columns
        input_column = generation_config.get("input_column", "input_ids")
        target_column = generation_config.get("target_column", "labels")
        
        # Generate predictions
        all_predictions = []
        all_references = []
        
        # Process in batches
        batch_size = generation_config.get("batch_size", 8)
        
        for i in range(0, len(eval_dataset), batch_size):
            batch = eval_dataset[i:i+batch_size]
            
            # Get inputs
            inputs = batch[input_column]
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode generated text
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_predictions.extend(decoded_outputs)
            
            # Decode reference text
            references = batch[target_column]
            decoded_references = tokenizer.batch_decode(references, skip_special_tokens=True)
            all_references.extend(decoded_references)
        
        # Compute metrics
        metrics = {}
        
        if "rouge" in self.metrics_config:
            rouge = evaluate.load("rouge")
            rouge_scores = rouge.compute(
                predictions=all_predictions,
                references=all_references,
                use_stemmer=True,
            )
            metrics.update({f"rouge_{k}": v for k, v in rouge_scores.items()})
        
        if "bleu" in self.metrics_config:
            bleu = evaluate.load("bleu")
            bleu_score = bleu.compute(
                predictions=all_predictions,
                references=[[ref] for ref in all_references],
            )
            metrics["bleu"] = bleu_score["bleu"]
        
        if "meteor" in self.metrics_config:
            meteor = evaluate.load("meteor")
            meteor_score = meteor.compute(
                predictions=all_predictions,
                references=all_references,
            )
            metrics["meteor"] = meteor_score["meteor"]
        
        return metrics 