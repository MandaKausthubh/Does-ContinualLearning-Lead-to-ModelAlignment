"""
File naming conventions and path management for the project.

This module centralizes all file paths and naming conventions to ensure
consistency across the codebase.
"""

import os
from typing import Optional
from pathlib import Path
from datetime import datetime


class PathManager:
    """Centralized path management for the project."""

    # Base directories
    BASE_DIR = Path("/Users/mandakausthubh/ML_Research/NLPDL")
    CACHE_DIR = BASE_DIR / "cache"
    EXPERIMENTS_DIR = BASE_DIR / "experiments"
    RESULTS_DIR = BASE_DIR / "results"
    CONFIG_DIR = BASE_DIR / "configs"
    DATA_DIR = BASE_DIR / "data"

    # Cache subdirectories
    MODEL_CACHE = CACHE_DIR / "models"
    DATASET_CACHE = CACHE_DIR / "datasets"
    EVAL_CACHE = CACHE_DIR / "eval"

    # Results subdirectories
    METRICS_DIR = RESULTS_DIR / "metrics"
    PLOTS_DIR = RESULTS_DIR / "plots"
    LOGS_DIR = RESULTS_DIR / "logs"

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories if they don't exist."""
        dirs = [
            cls.CACHE_DIR,
            cls.MODEL_CACHE,
            cls.DATASET_CACHE,
            cls.EVAL_CACHE,
            cls.EXPERIMENTS_DIR,
            cls.RESULTS_DIR,
            cls.METRICS_DIR,
            cls.PLOTS_DIR,
            cls.LOGS_DIR,
            cls.DATA_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_experiment_dir(cls, experiment_name: str) -> Path:
        """Get directory for a specific experiment."""
        return cls.EXPERIMENTS_DIR / experiment_name

    @classmethod
    def get_checkpoint_dir(cls, experiment_name: str, phase: Optional[str] = None) -> Path:
        """Get checkpoint directory for an experiment."""
        exp_dir = cls.get_experiment_dir(experiment_name)
        if phase:
            return exp_dir / phase / "checkpoints"
        return exp_dir / "checkpoints"


class DatasetPaths:
    """Paths for dataset storage and caching."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else PathManager.DATASET_CACHE

    def get_dataset_cache(self, dataset_name: str) -> Path:
        """Get cache directory for a specific dataset."""
        # Normalize dataset name
        name = dataset_name.lower().replace("-", "_").replace("/", "_")
        return self.cache_dir / name

    def get_processed_path(self, dataset_name: str, split: str = "train") -> Path:
        """Get path for processed dataset file."""
        cache_dir = self.get_dataset_cache(dataset_name)
        return cache_dir / f"{split}_processed.json"

    def get_gender_pairs_path(self, dataset_name: str, split: str = "train") -> Path:
        """Get path for gender-swapped pairs file."""
        cache_dir = self.get_dataset_cache(dataset_name)
        return cache_dir / f"{split}_gender_pairs.json"

    def get_bias_subset_path(self, dataset_name: str, n_samples: int) -> Path:
        """Get path for bias evaluation subset."""
        cache_dir = self.get_dataset_cache(dataset_name)
        return cache_dir / f"bias_subset_{n_samples}.json"


class ExperimentPaths:
    """Paths for experiment outputs."""

    EXPERIMENT_NAMES = {
        "baseline": "exp_01_baseline",
        "sft": "exp_02_sft",
        "sdft": "exp_03_sdft",
    }

    def __init__(self, experiment_type: str, custom_name: Optional[str] = None):
        """
        Initialize experiment paths.

        Args:
            experiment_type: Type of experiment (baseline, sft, sdft)
            custom_name: Optional custom experiment name
        """
        if custom_name:
            self.experiment_name = custom_name
        else:
            self.experiment_name = self.EXPERIMENT_NAMES.get(
                experiment_type.lower(),
                f"exp_{experiment_type}"
            )

        self.experiment_dir = PathManager.get_experiment_dir(self.experiment_name)
        self.log_dir = self.experiment_dir / "logs"
        self.config_path = self.experiment_dir / "config.json"

    def get_phase_dir(self, phase: str) -> Path:
        """Get directory for a specific training phase."""
        return self.experiment_dir / phase

    def get_checkpoint_path(self, phase: str, step: Optional[int] = None, epoch: Optional[int] = None) -> Path:
        """Get checkpoint file path."""
        checkpoint_dir = self.get_phase_dir(phase) / "checkpoints"

        if step is not None:
            return checkpoint_dir / f"checkpoint_step_{step}.pt"
        elif epoch is not None:
            return checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        else:
            return checkpoint_dir / "checkpoint_latest.pt"

    def get_final_model_path(self, phase: str) -> Path:
        """Get path for final model of a phase."""
        return self.get_phase_dir(phase) / "final"

    def get_log_file(self, name: str = "train") -> Path:
        """Get log file path."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / f"{name}.log"

    def get_results_path(self, dataset: Optional[str] = None) -> Path:
        """Get results file path."""
        if dataset:
            return self.experiment_dir / f"results_{dataset}.json"
        return self.experiment_dir / "results.json"

    def get_bias_metrics_path(self, dataset: str, phase: Optional[str] = None) -> Path:
        """Get path for bias metrics file."""
        if phase:
            return self.get_phase_dir(phase) / f"bias_metrics_{dataset}.json"
        return self.experiment_dir / f"bias_metrics_{dataset}.json"

    def get_outputs_path(self, dataset: str) -> Path:
        """Get path for model outputs file."""
        return self.experiment_dir / f"outputs_{dataset}.json"


class ResultsPaths:
    """Paths for aggregated results and comparisons."""

    def __init__(self):
        self.results_dir = PathManager.RESULTS_DIR

    def get_comparison_path(self, name: str = "comparison") -> Path:
        """Get path for experiment comparison file."""
        return PathManager.METRICS_DIR / f"{name}.json"

    def get_plot_path(self, name: str, format: str = "png") -> Path:
        """Get path for plot file."""
        PathManager.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        return PathManager.PLOTS_DIR / f"{name}.{format}"

    def get_metrics_summary_path(self, name: str = "metrics_summary") -> Path:
        """Get path for metrics summary file."""
        return PathManager.METRICS_DIR / f"{name}.json"

    def get_report_path(self, name: str = "report") -> Path:
        """Get path for report file."""
        return self.results_dir / f"{name}.md"


class ConfigPaths:
    """Paths for configuration files."""

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get configuration directory."""
        return PathManager.CONFIG_DIR

    @classmethod
    def get_model_config(cls) -> Path:
        """Get model configuration file path."""
        return cls.get_config_dir() / "model_config.yaml"

    @classmethod
    def get_training_config(cls) -> Path:
        """Get training configuration file path."""
        return cls.get_config_dir() / "training_config.yaml"

    @classmethod
    def get_eval_config(cls) -> Path:
        """Get evaluation configuration file path."""
        return cls.get_config_dir() / "eval_config.yaml"

    @classmethod
    def get_custom_config(cls, name: str) -> Path:
        """Get custom configuration file path."""
        return cls.get_config_dir() / f"{name}.yaml"


class CachePaths:
    """Paths for caching intermediate results."""

    def __init__(self):
        self.cache_dir = PathManager.CACHE_DIR

    def get_model_cache(self, model_name: str) -> Path:
        """Get cache directory for a specific model."""
        # Normalize model name
        name = model_name.replace("/", "_").replace("-", "_")
        return PathManager.MODEL_CACHE / name

    def get_eval_cache(self, eval_name: str) -> Path:
        """Get cache directory for evaluation results."""
        return PathManager.EVAL_CACHE / eval_name

    def get_generation_cache(self, model_name: str, dataset: str) -> Path:
        """Get cache file for generated outputs."""
        cache_dir = self.get_eval_cache(f"{model_name}_{dataset}")
        return cache_dir / "generations.json"


class LogPaths:
    """Paths for logging."""

    def __init__(self):
        self.log_dir = PathManager.LOGS_DIR

    def get_run_log(self, run_name: str, timestamp: bool = True) -> Path:
        """Get log file path for a specific run."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return self.log_dir / f"{run_name}_{ts}.log"
        return self.log_dir / f"{run_name}.log"

    def get_wandb_dir(self) -> Path:
        """Get Weights & Biases log directory."""
        wandb_dir = self.log_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        return wandb_dir


# Convenience functions for common operations
def get_experiment_paths(experiment_type: str, custom_name: Optional[str] = None) -> ExperimentPaths:
    """Get paths for an experiment."""
    return ExperimentPaths(experiment_type, custom_name)


def get_dataset_paths(cache_dir: Optional[str] = None) -> DatasetPaths:
    """Get paths for dataset operations."""
    return DatasetPaths(cache_dir)


def ensure_all_dirs():
    """Ensure all project directories exist."""
    PathManager.ensure_dirs()


def get_checkpoint_pattern(experiment_name: str, phase: str) -> str:
    """Get glob pattern for checkpoint files."""
    return str(PathManager.EXPERIMENTS_DIR / experiment_name / phase / "checkpoints" / "checkpoint_*.pt")


# File name conventions
class NamingConventions:
    """Standard naming conventions for files."""

    @staticmethod
    def checkpoint_name(step: Optional[int] = None, epoch: Optional[int] = None) -> str:
        """Generate checkpoint file name."""
        if step is not None:
            return f"checkpoint_step_{step}.pt"
        elif epoch is not None:
            return f"checkpoint_epoch_{epoch}.pt"
        return "checkpoint_latest.pt"

    @staticmethod
    def results_name(dataset: str, timestamp: bool = False) -> str:
        """Generate results file name."""
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"results_{dataset}_{ts}.json"
        return f"results_{dataset}.json"

    @staticmethod
    def metrics_name(metric_type: str, dataset: str) -> str:
        """Generate metrics file name."""
        return f"{metric_type}_metrics_{dataset}.json"

    @staticmethod
    def plot_name(plot_type: str, dataset: Optional[str] = None) -> str:
        """Generate plot file name."""
        if dataset:
            return f"{plot_type}_{dataset}"
        return plot_type

    @staticmethod
    def log_name(prefix: str, timestamp: bool = True) -> str:
        """Generate log file name."""
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{prefix}_{ts}.log"
        return f"{prefix}.log"


# Initialize directories on module import
PathManager.ensure_dirs()
