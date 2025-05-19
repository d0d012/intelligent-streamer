"""
Logging configuration for the application.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any

from .config import get_config

def setup_logging() -> None:
    """
    Configure application-wide logging based on configuration.
    """
    config = get_config()
    log_config = config.get_section("logging")
    
    # Get log level
    level_name = log_config.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if log_config.get("file_enabled", False):
        log_path = log_config.get("file_path", "logs/app.log")
        log_dir = os.path.dirname(log_path)
        
        # Create log directory if it doesn't exist
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Log configuration complete
    root_logger.info(f"Logging configured at level {level_name}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)