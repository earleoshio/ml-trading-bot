"""
Custom AutoML System for Cryptocurrency Trading

This package provides a comprehensive AutoML system specifically designed for
cryptocurrency trading patterns, including automated feature engineering,
model selection, regime detection, and strategy generation.
"""

__version__ = "0.1.0"
__author__ = "ML Trading Bot Team"

from .automl_pipeline import AutoMLPipeline

__all__ = ["AutoMLPipeline"]