"""
Feature Engineering Module

This module provides automated feature engineering specifically designed for
cryptocurrency trading, including technical indicators, market microstructure
features, sentiment analysis, and pattern recognition.
"""

from .technical_indicators import TechnicalIndicators
from .market_microstructure import MarketMicrostructure
from .sentiment_features import SentimentFeatures
from .pattern_recognition import PatternRecognition
from .feature_selector import FeatureSelector

__all__ = [
    "TechnicalIndicators",
    "MarketMicrostructure", 
    "SentimentFeatures",
    "PatternRecognition",
    "FeatureSelector"
]