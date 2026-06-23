"""
Utils package for the manga translation worker.

Public API for logging:
- setup_logging(): Initialize logging system (call once at startup)
- get_logger(): Get a logger for a module
- set_request_id(): Context manager to set request ID
- request_id_var: ContextVar for request ID tracking

Example:
    # In handler.py (or main entry point):
    from utils import setup_logging, get_logger, set_request_id
    setup_logging(log_level="INFO", log_format="json")
    logger = get_logger()
    logger.info("Application started")

    # In any pipeline module:
    from utils import get_logger, set_request_id
    logger = get_logger(__name__)

    # In request handler:
    with set_request_id("job-12345"):
        logger.info("Processing job")
"""

from .logger import (
    JSONLogFormatter,
    TextLogFormatter,
    get_logger,
    request_id_var,
    set_request_id,
    setup_logging,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "set_request_id",
    "request_id_var",
    "JSONLogFormatter",
    "TextLogFormatter",
]
