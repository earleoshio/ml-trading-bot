"""
Regime Detection Module

This module provides automated market regime detection specifically designed
for cryptocurrency markets, including volatility regimes, trend detection,
and microstructure regime changes.
"""

from .volatility_regimes import VolatilityRegimeDetector
from .trend_detection import TrendDetector
from .regime_classifier import RegimeClassifier

__all__ = [
    "VolatilityRegimeDetector",
    "TrendDetector",
    "RegimeClassifier"
]