"""Structured logging for QueryFrame."""

import logging
import os

_LOG_FORMAT = "[%(name)s] %(levelname)s: %(message)s"


def get_logger(name: str = "queryframe") -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    level = os.environ.get("QF_LOG_LEVEL", "WARNING").upper()
    logger.setLevel(getattr(logging, level, logging.WARNING))
    return logger
