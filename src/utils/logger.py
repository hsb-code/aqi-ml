"""
Logging configuration and utilities for AQI ML system.
Provides consistent logging across all modules with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from ..config.settings import config


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional specific log file name (without path)
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        console: Whether to log to console
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Download started")
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or config.logging.log_level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=config.logging.log_format,
        datefmt=config.logging.log_date_format
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.logging.log_to_file:
        # Determine log file path
        if log_file:
            log_path = config.paths.logs / log_file
        else:
            log_path = config.paths.logs / f"{name.replace('.', '_')}.log"
        
        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler (prevents huge log files)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.logging.log_file_max_bytes,
            backupCount=config.logging.log_file_backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments.
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> @log_function_call(logger)
        >>> def download_data(product, start_date):
        >>>     pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            args_str = ', '.join([repr(a) for a in args])
            kwargs_str = ', '.join([f"{k}={v!r}" for k, v in kwargs.items()])
            all_args = ', '.join(filter(None, [args_str, kwargs_str]))
            
            logger.debug(f"Calling {func.__name__}({all_args})")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    
    Example:
        >>> class Downloader(LoggerMixin):
        >>>     def __init__(self):
        >>>         self._init_logger()
        >>>     
        >>>     def download(self):
        >>>         self.logger.info("Starting download")
    """
    
    def _init_logger(self, name: Optional[str] = None):
        """Initialize logger for the class"""
        logger_name = name or self.__class__.__name__
        self.logger = setup_logger(logger_name)


if __name__ == "__main__":
    # Test logging
    logger = setup_logger("test_logger")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print(f"\nLog file location: {config.paths.logs}")
