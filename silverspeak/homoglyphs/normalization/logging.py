"""
Logging configuration for homoglyph normalization utilities.

This module provides functions and constants for configuring logging
in the homoglyph normalization system.

Author: Aldan Creo (ACMC) <os@acmc.fyi>
"""

import logging
from typing import Optional

# Default valid log levels
VALID_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Configure logging with a standardized format for production use
logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """
    Configure the logging system for the library.

    Args:
        level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_string (Optional[str]): Custom format string for log messages.
            If None, a default format will be used.

    Returns:
        None
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"

    log_level = VALID_LOG_LEVELS.get(level.upper(), logging.INFO)

    # Configure the root logger
    logging.basicConfig(level=log_level, format=format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Set the level for this module's logger
    logger.setLevel(log_level)

    # Prevent logging propagation if needed
    # logger.propagate = False
