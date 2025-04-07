#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grid Search hyperparameter optimization implementation.

This module provides a basic implementation of grid search for
hyperparameter optimization, which is useful as a baseline method
or when the search space is small.
"""

import logging
import itertools
import time
from typing import Dict, Any, List, Callable, Tuple, Optional

from hpo.base import BaseHPO


class GridSearchHPO(BaseHPO):
    """
    Hyperparameter optimization using grid search.
    
    This class implements hyperparameter optimization using grid search,
    which exhaustively searches through all possible combinations of
    hyperparameters in the search space.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Grid Search HPO.
        
        Args:
            config: Dictionary containing configuration for the HPO,
                   including:
                   - max_evals: Maximum number of evaluations (optional, default is to try all combinations)
                   - parallel: Whether to run evaluations in parallel (optional, default is False)
                   - parallel_backend: Backend to use for parallelization (optional, default is "joblib")
                   - n_jobs: Number of parallel jobs (optional, default is -1, which uses all available cores)
        """
        super().__init__(config)
        
        # Configuration parameters
        self.max_evals = self.config.get("max_evals", None)
        self.parallel = self.config.get("parallel", False)
        self.parallel_backend = self.config.get("parallel_backend", "joblib")
        self.n_jobs = self.config.get("n_jobs", -1)
        
        # Results tracking
        self.all_results = []
        self.start_time = None
        self.end_time = None
    
    def optimize(self, objective_fn: Callable[[Dict[str, Any]], float], 
                search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run hyperparameter optimization using grid search.
        
        Args:
            objective_fn: Function that takes hyperparameters and returns a score to maximize
            search_space: Dictionary defining the search space for hyperparameters,
                          where each key is a parameter name and each value is a list
                          of possible values for that parameter.
            
        Returns:
            Dictionary containing the best hyperparameters found
        """
        self.logger.info(f"Starting Grid Search HPO")
        self.logger.info(f"Search space: {search_space}")
        
        # Verify the search space format
        for param, values in search_space.items():
            if not isinstance(values, list):
                self.logger.error(f"Parameter {param} values must be a list, got {type(values)}")
                raise ValueError(f"Parameter {param} values must be a list, got {type(values)}")
        
        # Generate all combinations of parameters
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        # Generate all combinations
        param_combinations = list(itertools.product(*param_values))
        self.logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        # Limit the number of evaluations if specified
        if self.max_evals is not None and len(param_combinations) > self.max_evals:
            self.logger.warning(
                f"Number of combinations ({len(param_combinations)}) exceeds max_evals ({self.max_evals}). "
                f"Truncating to {self.max_evals} combinations."
            )
            param_combinations = param_combinations[:self.max_evals]
        
        # Run the evaluations
        self.start_time = time.time()
        
        if self.parallel and len(param_combinations) > 1:
            self._run_parallel(objective_fn, param_names, param_combinations)
        else:
            self._run_sequential(objective_fn, param_names, param_combinations)
        
        self.end_time = time.time()
        
        # Find the best parameters
        if not self.all_results:
            self.logger.warning("No evaluations were run!")
            return {}
        
        best_idx = max(range(len(self.all_results)), key=lambda i: self.all_results[i][1])
        self.best_params = self.all_results[best_idx][0]
        self.best_score = self.all_results[best_idx][1]
        
        self.logger.info(f"HPO completed in {self.end_time - self.start_time:.2f} seconds")
        self.logger.info(f"Best score: {self.best_score}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def _run_sequential(self, objective_fn: Callable, param_names: List[str], 
                       param_combinations: List[Tuple]) -> None:
        """
        Run evaluations sequentially.
        
        Args:
            objective_fn: Function that takes hyperparameters and returns a score to maximize
            param_names: List of parameter names
            param_combinations: List of parameter value combinations
        """
        for i, combination in enumerate(param_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Evaluate objective function
            self.logger.info(f"Evaluating combination {i+1}/{len(param_combinations)}: {params}")
            try:
                score = objective_fn(params)
                self.all_results.append((params, score))
                self.logger.info(f"Combination {i+1} score: {score}")
            except Exception as e:
                self.logger.error(f"Error evaluating combination {i+1}: {e}")
    
    def _run_parallel(self, objective_fn: Callable, param_names: List[str], 
                     param_combinations: List[Tuple]) -> None:
        """
        Run evaluations in parallel.
        
        Args:
            objective_fn: Function that takes hyperparameters and returns a score to maximize
            param_names: List of parameter names
            param_combinations: List of parameter value combinations
        """
        if self.parallel_backend == "joblib":
            try:
                from joblib import Parallel, delayed
                
                # Wrapper function for parallel execution
                def evaluate_params(combination):
                    params = dict(zip(param_names, combination))
                    try:
                        score = objective_fn(params)
                        return (params, score)
                    except Exception as e:
                        self.logger.error(f"Error evaluating params {params}: {e}")
                        return (params, float('-inf'))
                
                # Run in parallel
                self.logger.info(f"Running {len(param_combinations)} evaluations in parallel with joblib")
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(evaluate_params)(combination) for combination in param_combinations
                )
                
                # Filter out failed evaluations
                self.all_results = [r for r in results if r[1] != float('-inf')]
                
            except ImportError:
                self.logger.warning("joblib not installed. Falling back to sequential execution.")
                self._run_sequential(objective_fn, param_names, param_combinations)
        else:
            self.logger.warning(f"Unknown parallel backend: {self.parallel_backend}. Falling back to sequential execution.")
            self._run_sequential(objective_fn, param_names, param_combinations)
    
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
        if not self.all_results:
            self.logger.warning("No optimization has been run yet.")
            return {}
        
        # Sort results by score in descending order
        sorted_results = sorted(self.all_results, key=lambda x: x[1], reverse=True)
        
        # Create results table
        summary = {
            "best_score": self.best_score,
            "best_params": self.best_params,
            "num_evaluations": len(self.all_results),
            "duration_seconds": self.end_time - self.start_time if self.end_time else None,
            "top_k_results": sorted_results[:min(5, len(sorted_results))],
            "parallel": self.parallel,
            "parallel_backend": self.parallel_backend if self.parallel else None,
        }
        
        return summary 