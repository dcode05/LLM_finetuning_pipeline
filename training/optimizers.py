#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimizers and learning rate schedulers for the training module.
"""

import logging
from typing import Dict, Any, Optional, Union

import torch
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR, 
    LinearLR, 
    ReduceLROnPlateau,
    LRScheduler
)
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


def create_optimizer(model, config: Dict[str, Any]) -> Optional[Optimizer]:
    """
    Create an optimizer based on configuration.
    
    Args:
        model: The model to optimize
        config: Configuration dictionary for the optimizer
        
    Returns:
        Optimizer: An optimizer instance
    """
    logger = logging.getLogger(__name__)
    
    # Check if we have trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning("No trainable parameters found in the model")
        return None
    
    # Get optimizer parameters
    optimizer_name = config.get("name", "adamw").lower()
    learning_rate = config.get("learning_rate", 5e-5)
    weight_decay = config.get("weight_decay", 0.0)
    
    logger.info(f"Creating optimizer: {optimizer_name}")
    logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    
    # Create optimizer
    if optimizer_name == "adamw":
        beta1 = config.get("beta1", 0.9)
        beta2 = config.get("beta2", 0.999)
        epsilon = config.get("epsilon", 1e-8)
        
        optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        momentum = config.get("momentum", 0.9)
        
        optimizer = SGD(
            trainable_params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        logger.warning(f"Unknown optimizer: {optimizer_name}. Using AdamW as default.")
        optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    return optimizer


def create_scheduler(optimizer: Optimizer, config: Dict[str, Any]) -> Optional[LRScheduler]:
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer to schedule
        config: Configuration dictionary for the scheduler
        
    Returns:
        LRScheduler: A learning rate scheduler instance
    """
    logger = logging.getLogger(__name__)
    
    # Get scheduler parameters
    scheduler_name = config.get("name", "linear").lower()
    num_warmup_steps = config.get("num_warmup_steps", 0)
    num_training_steps = config.get("num_training_steps", 1000)
    
    logger.info(f"Creating LR scheduler: {scheduler_name}")
    logger.info(f"Warmup steps: {num_warmup_steps}, Training steps: {num_training_steps}")
    
    # Create scheduler
    if scheduler_name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_name == "cosine_with_restarts":
        from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
        num_cycles = config.get("num_cycles", 1)
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles
        )
    elif scheduler_name == "constant":
        from transformers import get_constant_schedule
        return get_constant_schedule(optimizer)
    elif scheduler_name == "constant_with_warmup":
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps
        )
    elif scheduler_name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=config.get("mode", "min"),
            factor=config.get("factor", 0.1),
            patience=config.get("patience", 10),
            verbose=True
        )
    else:
        logger.warning(f"Unknown scheduler: {scheduler_name}. Using linear as default.")
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        ) 