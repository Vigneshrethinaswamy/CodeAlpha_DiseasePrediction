"""
utils.py
========
Shared utility functions used across the CodeAlpha_DiseasePrediction
pipeline: logging configuration, joblib persistence helpers, and small
reusable helpers for timing and JSON I/O.

Centralizing these utilities avoids duplicating boilerplate (e.g. logger
setup) in every module and ensures consistent behavior across the project.
"""

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

import joblib

from src import config


def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a configured logger instance.

    The logger writes to both the console (stdout) and a persistent log
    file at `config.LOG_FILE_PATH`. Calling this function multiple times
    with the same `name` returns the same logger without duplicating
    handlers, which prevents repeated log lines when modules are imported
    more than once (e.g. during interactive notebook use).

    Args:
        name: Name of the logger, typically `__name__` of the calling module.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger already configured (e.g. re-imported module); avoid
        # attaching duplicate handlers which would cause repeated log lines.
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    formatter = logging.Formatter(fmt=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (best-effort; if the logs directory cannot be created,
    # fall back to console-only logging rather than crashing the pipeline).
    try:
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        file_handler = logging.FileHandler(config.LOG_FILE_PATH, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass

    logger.propagate = False
    return logger


_logger = get_logger(__name__)


@contextmanager
def timer(task_name: str) -> Generator[None, None, None]:
    """
    Context manager that logs the wall-clock duration of a code block.

    Usage:
        with timer("Training Random Forest"):
            model.fit(X_train, y_train)

    Args:
        task_name: Human-readable description of the task being timed.

    Yields:
        None.
    """
    start_time = time.time()
    _logger.info("Starting: %s", task_name)
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        _logger.info("Finished: %s (elapsed: %.2f seconds)", task_name, elapsed)


def save_joblib(obj: Any, path: str) -> None:
    """
    Persist a Python object to disk using joblib, creating parent
    directories as needed.

    Args:
        obj: The object to serialize (e.g. a fitted model, encoder, or scaler).
        path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    _logger.info("Saved artifact to: %s", path)


def load_joblib(path: str) -> Any:
    """
    Load a Python object previously saved with joblib.

    Args:
        path: Path to the serialized object.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist at `path`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required artifact not found at '{path}'. "
            f"Did you run the training pipeline (main.py) first?"
        )
    obj = joblib.load(path)
    _logger.info("Loaded artifact from: %s", path)
    return obj


def save_json(obj: Any, path: str) -> None:
    """
    Save a JSON-serializable Python object to disk.

    Args:
        obj: A JSON-serializable object (dict, list, etc.).
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    _logger.info("Saved JSON to: %s", path)


def load_json(path: str) -> Any:
    """
    Load a JSON file from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialized JSON content.

    Raises:
        FileNotFoundError: If the file does not exist at `path`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required JSON file not found at '{path}'.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_seconds(seconds: float) -> str:
    """
    Format a duration in seconds into a human-readable "Xm Ys" string.

    Args:
        seconds: Duration in seconds.

    Returns:
        A human-readable string, e.g. "2m 15s".
    """
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


def safe_filename(name: str) -> str:
    """
    Convert an arbitrary string (e.g. a disease or model name) into a
    filesystem-safe filename fragment by replacing problematic characters.

    Args:
        name: Raw string to sanitize.

    Returns:
        A sanitized string safe for use in file paths.
    """
    keep_chars = (" ", ".", "_", "-")
    sanitized = "".join(c if c.isalnum() or c in keep_chars else "_" for c in name)
    sanitized = sanitized.strip().replace(" ", "_")
    return sanitized.lower()
