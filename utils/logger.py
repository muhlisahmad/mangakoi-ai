"""
Logging utilities for the manga translation worker.

This module provides a centralized logging system with support for:
- JSON-formatted structured logs
- Request ID tracking via context variables (thread-safe)
- Environment-based configuration (LOG_LEVEL, LOG_FORMAT)
- Proper logger initialization and reuse

Architecture:
    - JSONLogFormatter: Outputs logs in JSON format with request context
    - TextLogFormatter: Outputs logs in human-readable format
    - request_id_var: ContextVar for tracking request IDs across async/threading boundaries
    - get_logger(): Factory function to retrieve configured loggers
    - setup_logging(): One-time initialization of the logging system
    - set_request_id(): Context manager to set request ID for current scope

Usage Example:
    # At application startup (handler.py):
    from utils.logging import setup_logging, get_logger
    setup_logging(log_level="INFO", log_format="json")

    # In any module:
    from utils.logging import get_logger, set_request_id
    logger = get_logger(__name__)

    # In request handler:
    with set_request_id("job-12345"):
        logger.info("Processing job")  # Will include request ID in logs

Notes:
    - Call setup_logging() exactly ONCE at application startup
    - get_logger() can be called multiple times; returns cached logger
    - Context variables (request_id_var, set_request_id) are thread-safe
"""

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Literal, override


def _get_log_filename() -> str:
    """
    Generate a log filename with ISO 8601 date format.

    Format: worker-YYYY-MM-DD.log

    Benefits:
    - Sortable by filename (lexicographic order = chronological order)
    - Easy to rotate logs by day
    - ISO 8601 standard compliant
    - Won't overwrite previous day's logs

    Returns:
        Filename in format: worker-2026-06-23.log
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    return f"worker-{date_str}.log"


# ── Context Variables ────────────────────────────────────────────────
# Thread-safe and async-safe way to track request IDs across call stack
request_id_var: ContextVar[str] = ContextVar("request_id", default="N/A")

# Track whether logging has been initialized (prevents duplicate setup)
_logging_initialized: bool = False


# ── Formatters ───────────────────────────────────────────────────────


class JSONLogFormatter(logging.Formatter):
    """
    Outputs logs in JSON format with structured fields.

    Includes:
    - timestamp: ISO 8601 UTC timestamp
    - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - request_id: Current request ID from context variable
    - message: Log message
    - exception: Full traceback if logging an exception
    - extra: Any additional fields passed via LogRecord.extra_data
    """

    @override
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="seconds"
            ),
            "level": record.levelname,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }

        # Include optional extra data if present
        if hasattr(record, "extra_data"):
            log_record["extra"] = (
                record.extra_data  # pyright: ignore[reportAttributeAccessIssue]
            )

        # Include full traceback for exceptions
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


class TextLogFormatter(logging.Formatter):
    """
    Outputs logs in human-readable text format.

    Format: YYYY-MM-DD HH:MM:SS - LEVEL - [Request: request_id] - message

    Useful for development and local testing.
    """

    @override
    def format(self, record: logging.LogRecord) -> str:
        record.request_id = request_id_var.get()
        return super().format(record)


# ── Public API ───────────────────────────────────────────────────────


def setup_logging(
    log_dir: str = "/runpod-volume/logs",
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    log_format: Literal["json", "text"] = "json",
) -> logging.Logger:
    """
    Initialize the logging system. Call this ONCE at application startup.

    Args:
        log_dir: Directory for log files. Uses ISO 8601 date-based filenames.
                 Defaults to "/runpod-volume/logs" (per RunPod best practices).
                 Example files: worker-2026-06-23.log, worker-2026-06-24.log
        log_level: Logging level. Defaults to "INFO" for production.
                   Use "DEBUG" for development.
                   Can be overridden by LOG_LEVEL environment variable.
        log_format: Output format: "json" for structured logs (default, recommended for production),
                    or "text" for human-readable logs (useful for development).
                    Can be overridden by LOG_FORMAT environment variable.

    Returns:
        The configured logger instance for the "runpod_worker" application.

    Example:
        >>> logger = setup_logging(log_dir="/var/logs", log_level="INFO", log_format="json")
        >>> logger.info("Application started")

    Note:
        - Idempotent: Safe to call multiple times (will only initialize once)
        - Sets up both stdout handler (for RunPod console) and file handler (persistent logs)
        - Log files use ISO 8601 date format: worker-YYYY-MM-DD.log
        - Log directory is created automatically if it doesn't exist
    """
    global _logging_initialized

    # Read configuration from environment variables (can override defaults)
    env_log_level = os.environ.get("LOG_LEVEL", log_level).upper()
    env_log_format = os.environ.get("LOG_FORMAT", log_format).lower()

    # Validate log level
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if env_log_level not in valid_levels:
        raise ValueError(
            f"Invalid LOG_LEVEL: {env_log_level}. Must be one of {valid_levels}"
        )

    # Validate log format
    valid_formats = {"json", "text"}
    if env_log_format not in valid_formats:
        raise ValueError(
            f"Invalid LOG_FORMAT: {env_log_format}. Must be one of {valid_formats}"
        )

    # Get or create the logger
    logger = logging.getLogger("runpod_worker")

    # Prevent duplicate handler setup if called multiple times
    if _logging_initialized or logger.handlers:
        return logger

    _logging_initialized = True

    os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists

    # Configure logger level
    logger.setLevel(getattr(logging, env_log_level))

    # Create stdout handler (RunPod captures stdout/stderr automatically)
    console_handler = logging.StreamHandler()

    # Create file handler with ISO 8601 date-based filename
    log_filename = _get_log_filename()
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))

    # Select and attach formatter
    if env_log_format == "json":
        formatter = JSONLogFormatter()
    else:
        formatter = TextLogFormatter(
            fmt="%(asctime)s - %(levelname)s - [Request: %(request_id)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if not logger.handlers:
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (we handle everything ourselves)
    logger.propagate = False

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance. This is the recommended way to get a logger in any module.

    Args:
        name: Logger name, typically __name__ of the calling module.
              If None, returns the root "runpod_worker" logger.

    Returns:
        A configured logging.Logger instance.

    Example:
        In pipeline/detect.py:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Detection started")

    Note:
        - Returns a child logger under "runpod_worker" if name is provided
        - Child loggers inherit handlers and level from parent
        - Safe to call from any module, any number of times
    """
    if name is None:
        return logging.getLogger("runpod_worker")
    return logging.getLogger(f"runpod_worker.{name}")


@contextmanager
def set_request_id(request_id: str):
    """
    Context manager to set request ID for the current scope.

    Thread-safe and async-safe. Automatically resets request_id when exiting scope.

    Args:
        request_id: The request/job ID to associate with logs in this scope.

    Example:
        >>> from utils.logger import set_request_id, get_logger
        >>> logger = get_logger()
        >>> with set_request_id("job-12345"):
        ...     logger.info("Processing")  # Logs include "request_id": "job-12345"

    Note:
        - Works correctly with nested contexts
        - Thread-safe and async-safe (uses ContextVar)
        - Automatically restores previous value on exit
    """
    token = request_id_var.set(request_id)
    try:
        yield
    finally:
        request_id_var.reset(token)
