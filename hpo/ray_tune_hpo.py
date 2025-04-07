#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ray Tune hyperparameter optimization implementation.

This module provides an implementation of hyperparameter optimization
using Ray Tune, which is scalable, efficient, and supports various
search algorithms.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, List, Union

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
    from ray.tune.search.bayesopt import BayesOptSearch
    from ray.tune.search.hyperopt import HyperOptSearch
except ImportError:
    logging.warning("Ray Tune not installed. Install with `pip install ray[tune]`")

from hpo.base import BaseHPO


class RayTuneHPO(BaseHPO):
    """
    Hyperparameter optimization using Ray Tune.
    
    This class implements hyperparameter optimization using Ray Tune,
    supporting various search algorithms including grid search, random search,
    Bayesian optimization, and HyperOpt.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Ray Tune HPO.
        
        Args:
            config: Dictionary containing configuration for the HPO,
                   including:
                   - search_algorithm: One of "random", "grid", "bayesopt", "hyperopt"
                   - num_samples: Number of samples to take from the search space
                   - max_concurrent_trials: Maximum number of concurrent trials
                   - scheduler: One of "asha", "hyperband", or None
                   - resources_per_trial: Dict of resources per trial (e.g., {"cpu": 1, "gpu": 0.5})
        """
        super().__init__(config)
        
        # Check if Ray Tune is installed
        try:
            import ray
        except ImportError:
            self.logger.error("Ray Tune not installed. Install with `pip install ray[tune]`")
            raise ImportError("Ray Tune not installed. Install with `pip install ray[tune]`")
        
        # Get configuration parameters
        self.search_algorithm = self.config.get("search_algorithm", "random")
        self.num_samples = self.config.get("num_samples", 10)
        self.max_concurrent_trials = self.config.get("max_concurrent_trials", 1)
        self.scheduler_type = self.config.get("scheduler", None)
        self.resources_per_trial = self.config.get("resources_per_trial", {"cpu": 1})
        
        # Tracking variables
        self.results = None
        self.experiment_name = f"llm_finetune_hpo_{int(time.time())}"
    
    def optimize(self, objective_fn: Callable[[Dict[str, Any]], float], 
                search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run hyperparameter optimization using Ray Tune.
        
        Args:
            objective_fn: Function that takes hyperparameters and returns a score to maximize
            search_space: Dictionary defining the search space for hyperparameters
            
        Returns:
            Dictionary containing the best hyperparameters found
        """
        self.logger.info(f"Starting Ray Tune HPO with {self.search_algorithm} search algorithm")
        self.logger.info(f"Search space: {search_space}")
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Set up the search algorithm
        search_alg = self._setup_search_algorithm(search_space)
        
        # Set up the scheduler
        scheduler = self._setup_scheduler()
        
        # Define the reporter
        reporter = CLIReporter(
            parameter_columns=list(search_space.keys()),
            metric_columns=["score", "training_iteration"]
        )
        
        # Wrap the objective function for Ray Tune
        def tune_objective(config, checkpoint_dir=None):
            # Ray Tune expects the objective function to report results to tune
            score = objective_fn(config)
            tune.report(score=score)
        
        # Run the optimization
        try:
            analysis = tune.run(
                tune_objective,
                config=search_space,
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=self.num_samples,
                resources_per_trial=self.resources_per_trial,
                max_concurrent_trials=self.max_concurrent_trials,
                progress_reporter=reporter,
                name=self.experiment_name,
                verbose=1
            )
            
            # Store results
            self.results = analysis
            self.best_params = analysis.best_config
            self.best_score = analysis.best_result["score"]
            
            self.logger.info(f"HPO completed. Best score: {self.best_score}")
            self.logger.info(f"Best parameters: {self.best_params}")
            
            return self.best_params
            
        except Exception as e:
            self.logger.error(f"Error during Ray Tune optimization: {e}")
            raise
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best hyperparameters found during optimization.
        
        Returns:
            Dictionary containing the best hyperparameters
        """
        if self.best_params is None:
            self.logger.warning("No optimization has been run yet.")
            return {}
        
        return self.best_params
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization results.
        
        Returns:
            Dictionary containing summary information about the optimization run
        """
        if self.results is None:
            self.logger.warning("No optimization has been run yet.")
            return {}
        
        summary = {
            "best_score": self.best_score,
            "best_params": self.best_params,
            "num_trials": len(self.results.trials),
            "experiment_name": self.experiment_name,
            "search_algorithm": self.search_algorithm,
            "num_samples": self.num_samples,
        }
        
        return summary
    
    def _setup_search_algorithm(self, search_space: Dict[str, Any]) -> Optional[Any]:
        """
        Set up the search algorithm based on the configuration.
        
        Args:
            search_space: Dictionary defining the search space for hyperparameters
            
        Returns:
            Search algorithm object or None (for random search)
        """
        if self.search_algorithm == "random":
            return None
        elif self.search_algorithm == "bayesopt":
            return BayesOptSearch(
                metric="score", 
                mode="max",
                random_search_steps=self.config.get("random_search_steps", 10)
            )
        elif self.search_algorithm == "hyperopt":
            return HyperOptSearch(
                metric="score", 
                mode="max"
            )
        else:
            self.logger.warning(f"Unknown search algorithm: {self.search_algorithm}. Using random search.")
            return None
    
    def _setup_scheduler(self) -> Optional[Any]:
        """
        Set up the trial scheduler based on the configuration.
        
        Returns:
            Scheduler object or None
        """
        if self.scheduler_type == "asha":
            return ASHAScheduler(
                metric="score",
                mode="max",
                max_t=self.config.get("max_iterations", 100),
                grace_period=self.config.get("grace_period", 10),
                reduction_factor=self.config.get("reduction_factor", 2)
            )
        elif self.scheduler_type == "hyperband":
            return HyperBandScheduler(
                metric="score",
                mode="max",
                max_t=self.config.get("max_iterations", 100)
            )
        else:
            return None 