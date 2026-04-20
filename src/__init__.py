"""
LLM Continual Learning for Model Alignment

A research project investigating whether continual learning via self-distillation
can serve as a stable mechanism for debiasing and model alignment in Large Language Models.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from . import data
from . import models
from . import training
from . import evaluation
from . import pipelines
from . import utils

__all__ = ['data', 'models', 'training', 'evaluation', 'pipelines', 'utils']