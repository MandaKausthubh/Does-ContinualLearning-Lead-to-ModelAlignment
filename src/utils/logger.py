"""
Logging utilities with WandB integration.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import wandb


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory to save log files (None for no file logging)
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class WandBLogger:
    """Wrapper for WandB logging."""

    def __init__(
        self,
        project: str = "llm-continual-alignment",
        name: Optional[str] = None,
        config: Optional[dict] = None,
        enabled: bool = True
    ):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            name: Run name
            config: Configuration dictionary
            enabled: Whether to enable logging
        """
        self.enabled = enabled
        self.run = None

        if enabled:
            try:
                self.run = wandb.init(
                    project=project,
                    name=name,
                    config=config
                )
            except Exception as e:
                print(f"Warning: Could not initialize WandB: {e}")
                self.enabled = False

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to WandB."""
        if self.enabled and self.run:
            wandb.log(metrics, step=step)

    def log_table(self, key: str, data: list, columns: list):
        """Log a table to WandB."""
        if self.enabled and self.run:
            table = wandb.Table(data=data, columns=columns)
            wandb.log({key: table})

    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run:
            wandb.finish()


class MetricsLogger:
    """Logger for tracking metrics during training."""

    def __init__(self, log_dir: str):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to save metrics
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.metrics_history = []

    def log_metric(self, step: int, metrics: dict):
        """Log a metric at a specific step."""
        entry = {"step": step, **metrics}
        self.metrics_history.append(entry)

    def save(self, filename: str = "metrics.json"):
        """Save metrics history to file."""
        import json

        output_path = os.path.join(self.log_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def get_summary(self) -> dict:
        """Get summary statistics of logged metrics."""
        if not self.metrics_history:
            return {}

        import numpy as np

        summary = {}

        # Get all metric keys
        keys = set()
        for entry in self.metrics_history:
            keys.update(k for k in entry.keys() if k != "step")

        for key in keys:
            values = [e[key] for e in self.metrics_history if key in e]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "last": float(values[-1])
                }

        return summary
