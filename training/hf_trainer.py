#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HuggingFace Transformers Trainer implementation.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Union, Callable

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    default_data_collator
)
import evaluate

from training.base import BaseTrainer
from training.optimizers import create_optimizer, create_scheduler


class HuggingFaceTrainer(BaseTrainer):
    """
    Trainer implementation using HuggingFace's Transformers Trainer.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace trainer.
        
        Args:
            config: Configuration dictionary for the trainer
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.trainer = None
        self.model = None
        self.tokenizer = None
    
    def train(self, model, tokenizer, dataset: Optional[DatasetDict] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the model using the HuggingFace Trainer.
        
        Args:
            model: The model to train
            tokenizer: The tokenizer to use
            dataset: The dataset to train on
            config: Optional configuration override
            
        Returns:
            Dict[str, Any]: Training results
        """
        config = config or self.config
        self.model = model
        self.tokenizer = tokenizer
        
        self.logger.info("Setting up training")
        
        # Create output dir if it doesn't exist
        output_dir = config.get("output_dir", "outputs/model")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up training arguments
        training_args = self._create_training_arguments(config)
        
        # Set up data collator
        data_collator = self._create_data_collator(model, tokenizer, config)
        
        # Set up optimizer
        optimizer = create_optimizer(model, config.get("optimizer", {}))
        
        # Set up callbacks
        callbacks = self._create_callbacks(config)
        
        self.logger.info("Creating Trainer")
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"] if dataset and "train" in dataset else None,
            eval_dataset=dataset["validation"] if dataset and "validation" in dataset else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, None),  # We'll create the scheduler later
            callbacks=callbacks,
        )
        
        # Create LR scheduler
        if optimizer:
            scheduler_config = config.get("lr_scheduler", {})
            num_training_steps = self.trainer.state.max_steps
            scheduler_config["num_training_steps"] = num_training_steps
            scheduler = create_scheduler(optimizer, scheduler_config)
            self.trainer.optimizers = (optimizer, scheduler)
        
        # Train the model
        self.logger.info("Starting training")
        start_time = time.time()
        train_result = self.trainer.train()
        end_time = time.time()
        
        # Log training results
        train_time = end_time - start_time
        train_metrics = train_result.metrics
        train_metrics["train_time"] = train_time
        
        self.logger.info(f"Training completed in {train_time:.2f} seconds")
        self.logger.info(f"Training metrics: {train_metrics}")
        
        # Save model
        self.logger.info(f"Saving model to {output_dir}")
        self.save_model(output_dir)
        
        # Save training metrics
        for key, value in train_metrics.items():
            self.logger.info(f"  {key} = {value}")
        
        return train_metrics
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model.
        
        Args:
            output_dir: Output directory
        """
        if self.trainer is None:
            raise ValueError("No trainer available. Train the model first.")
        
        self.trainer.save_model(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Model and tokenizer saved to {output_dir}")
    
    def compute_metrics(self, eval_preds):
        """
        Compute evaluation metrics.
        
        Args:
            eval_preds: Evaluation predictions from the trainer
            
        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """
        metrics_config = self.config.get("evaluation", {}).get("metrics", ["accuracy"])
        
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Post-process predictions and labels
        if hasattr(self.tokenizer, "pad_token_id"):
            # Replace -100 with pad token id
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # Initialize metrics
        metrics = {}
        
        # Decode predictions for text-based metrics
        decoded_preds = self.tokenizer.batch_decode(predictions.argmax(-1), skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute metrics
        if "accuracy" in metrics_config:
            accuracy_metric = evaluate.load("accuracy")
            metrics.update(accuracy_metric.compute(predictions=predictions.argmax(-1).flatten(), 
                                              references=labels.flatten()))
        
        if "rouge" in metrics_config:
            rouge_metric = evaluate.load("rouge")
            rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
            metrics.update({k: v for k, v in rouge_result.items()})
        
        if "bleu" in metrics_config:
            bleu_metric = evaluate.load("bleu")
            metrics["bleu"] = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)["bleu"]
        
        return metrics
    
    def _create_training_arguments(self, config: Dict[str, Any]) -> TrainingArguments:
        """
        Create training arguments for the Trainer.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            TrainingArguments: Training arguments
        """
        output_dir = config.get("output_dir", "outputs/model")
        per_device_train_batch_size = config.get("per_device_train_batch_size", 8)
        per_device_eval_batch_size = config.get("per_device_eval_batch_size", 8)
        num_train_epochs = config.get("num_train_epochs", 3)
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        
        # Learning rate and scheduler
        learning_rate = config.get("optimizer", {}).get("learning_rate", 5e-5)
        
        # Checkpointing
        save_strategy = config.get("save_strategy", "steps")
        save_steps = config.get("save_steps", 500)
        save_total_limit = config.get("save_total_limit", 3)
        
        # Evaluation
        evaluation_strategy = config.get("evaluation_strategy", "steps")
        eval_steps = config.get("eval_steps", 500)
        
        # Logging
        logging_dir = os.path.join(output_dir, "logs")
        logging_steps = config.get("logging_steps", 100)
        
        # Mixed precision
        fp16 = config.get("fp16", False)
        bf16 = config.get("bf16", False)
        
        # Weight decay
        weight_decay = config.get("optimizer", {}).get("weight_decay", 0.0)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            logging_dir=logging_dir,
            logging_steps=logging_steps,
            fp16=fp16,
            bf16=bf16,
            report_to=config.get("report_to", ["tensorboard"]),
            load_best_model_at_end=config.get("load_best_model_at_end", True),
            metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=config.get("greater_is_better", False),
        )
        
        return training_args
    
    def _create_data_collator(self, model, tokenizer, config: Dict[str, Any]):
        """
        Create a data collator for the trainer.
        
        Args:
            model: The model to train
            tokenizer: The tokenizer to use
            config: Configuration dictionary
            
        Returns:
            Data collator
        """
        model_type = getattr(model.config, "model_type", "")
        
        # Use different data collators based on model type
        if model_type in ["t5", "bart", "mt5", "pegasus"]:
            # Seq2Seq models
            return DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding="longest",
                max_length=tokenizer.model_max_length,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
        else:
            # Causal LM models
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
    
    def _create_callbacks(self, config: Dict[str, Any]) -> List:
        """
        Create callbacks for the trainer.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List: List of callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping_config = config.get("early_stopping", {})
        if early_stopping_config.get("enabled", False):
            self.logger.info("Adding early stopping callback")
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=early_stopping_config.get("patience", 3),
                early_stopping_threshold=early_stopping_config.get("threshold", 0.0)
            )
            callbacks.append(early_stopping_callback)
        
        return callbacks 