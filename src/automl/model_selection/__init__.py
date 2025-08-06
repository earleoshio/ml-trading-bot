"""
Model Selection Module

This module provides automated model selection and optimization specifically
designed for cryptocurrency trading, including model registry, hyperparameter
optimization, ensemble management, and meta-learning.
"""

from .model_registry import ModelRegistry
from .hyperparameter_optimizer import HyperparameterOptimizer
from .ensemble_manager import EnsembleManager
from .meta_learner import MetaLearner

__all__ = [
    "ModelRegistry",
    "HyperparameterOptimizer", 
    "EnsembleManager",
    "MetaLearner"
]