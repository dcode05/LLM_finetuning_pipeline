"""
Logging utilities for the LLM finetuning pipeline.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file
        log_format: Format string for the log messages
    """
    # Get the numeric log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Define log format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
    
    # Log initial message
    logging.info(f"Logging initialized at level {log_level}")


class LoggerAdapter(logging.LoggerAdapter):
    """
    Adapter for adding context information to log messages.
    """
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]) -> None:
        """
        Initialize the adapter with a logger and context.
        
        Args:
            logger: The logger to adapt
            context: Context information to add to log messages
        """
        super().__init__(logger, context)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process the log message by adding context information.
        
        Args:
            msg: The log message
            kwargs: Keyword arguments
            
        Returns:
            Tuple of (modified_message, modified_kwargs)
        """
        # Add context to message
        context_str = " ".join([f"{k}={v}" for k, v in self.extra.items()])
        return f"{msg} [{context_str}]", kwargs


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional context information.
    
    Args:
        name: The name of the logger
        context: Optional context information
        
    Returns:
        Logger: A logger instance
    """
    logger = logging.getLogger(name)
    
    if context:
        return LoggerAdapter(logger, context)
    
    return logger 