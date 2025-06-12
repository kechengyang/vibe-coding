"""
Logger Module for Employee Health Monitoring System.

This module handles application logging configuration.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from .config import get_config

def setup_logger(
    name: str = "employee_health_monitor",
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_size: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3
) -> logging.Logger:
    """Set up and configure a logger.
    
    Args:
        name: Logger name
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses default path)
        max_size: Maximum log file size in bytes before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        logging.Logger: Configured logger
    """
    # Get configuration
    config = get_config()
    
    # Use config values if not provided
    if log_level is None:
        log_level = config.get("app.log_level", "INFO")
    
    if log_file is None:
        log_dir = Path(config.get_data_dir())
        log_file = str(log_dir / "app.log")
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Get logger
    logger = logging.getLogger(name)
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(levelname)s - %(message)s"
    )
    
    # Create file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger '{name}' configured with level {log_level}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    If the logger doesn't exist, it will be created with default configuration.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        # If it's a child of the main logger, inherit its settings
        if name.startswith("employee_health_monitor"):
            logger.propagate = True
        else:
            # Otherwise, set up a new logger
            setup_logger(name)
    
    return logger


# Example usage
if __name__ == "__main__":
    # Set up main logger
    main_logger = setup_logger()
    
    # Log some messages
    main_logger.debug("This is a debug message")
    main_logger.info("This is an info message")
    main_logger.warning("This is a warning message")
    main_logger.error("This is an error message")
    main_logger.critical("This is a critical message")
    
    # Get a child logger
    child_logger = get_logger("employee_health_monitor.test")
    
    # Log some messages with the child logger
    child_logger.info("This is a message from the child logger")
