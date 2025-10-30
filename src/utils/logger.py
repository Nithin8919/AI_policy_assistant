"""Logging configuration"""
from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stderr,
    format="{time} {level} {message}",
    level="INFO"
)

def get_logger(name: str = None):
    """Get a logger instance (compatibility function)."""
    return logger




