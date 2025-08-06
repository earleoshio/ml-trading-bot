"""
Technical Indicators Module

Implements comprehensive technical analysis indicators specifically optimized
for cryptocurrency trading patterns.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple
from loguru import logger


class TechnicalIndicators:
    """
    Technical indicators pipeline for cryptocurrency trading.
    
    Provides automated generation of technical indicators including:
    - Trend indicators (MA, EMA, MACD, ADX)
    - Momentum indicators (RSI, Stochastic, CCI)
    - Volatility indicators (Bollinger Bands, ATR, VIX-like)
    - Volume indicators (OBV, Volume Profile, VWAP)
    - Crypto-specific indicators (Funding rates, Perpetual basis)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize technical indicators with configuration.
        
        Args:
            config: Configuration dictionary with indicator parameters
        """
        self.config = config or self._default_config()
        self.indicators = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for technical indicators."""
        return {
            'trend_indicators': {
                'sma_periods': [5, 10, 20, 50, 100, 200],
                'ema_periods': [5, 10, 20, 50, 100, 200],
                'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
                'adx_period': 14
            },
            'momentum_indicators': {
                'rsi_periods': [14, 21, 30],
                'stoch_params': {'fastk': 14, 'slowk': 3, 'slowd': 3},
                'cci_period': 20,
                'williams_r_period': 14
            },
            'volatility_indicators': {
                'bb_period': 20,
                'bb_std': 2,
                'atr_period': 14,
                'keltner_period': 20,
                'keltner_multiplier': 2
            },
            'volume_indicators': {
                'obv_enabled': True,
                'vwap_periods': [20, 50],
                'mfi_period': 14
            },
            'crypto_specific': {
                'funding_rate_lookback': 24,
                'basis_calculation': 'perpetual_spot',
                'whale_threshold': 1000000  # USD
            }
        }
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical indicator features from OHLCV data.
        
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with original data plus technical indicator features
        """
        logger.info("Generating technical indicator features...")
        
        result_df = data.copy()
        
        # Generate trend indicators
        result_df = self._add_trend_indicators(result_df)
        
        # Generate momentum indicators  
        result_df = self._add_momentum_indicators(result_df)
        
        # Generate volatility indicators
        result_df = self._add_volatility_indicators(result_df)
        
        # Generate volume indicators
        result_df = self._add_volume_indicators(result_df)
        
        # Generate crypto-specific indicators
        result_df = self._add_crypto_specific_indicators(result_df)
        
        logger.info(f"Generated {len(result_df.columns) - len(data.columns)} technical features")
        return result_df
    
    def _add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        result = data.copy()
        
        # Simple Moving Averages
        for period in self.config['trend_indicators']['sma_periods']:
            result[f'sma_{period}'] = talib.SMA(data['close'], timeperiod=period)
            
        # Exponential Moving Averages
        for period in self.config['trend_indicators']['ema_periods']:
            result[f'ema_{period}'] = talib.EMA(data['close'], timeperiod=period)
            
        # MACD
        macd_params = self.config['trend_indicators']['macd_params']
        macd, macd_signal, macd_hist = talib.MACD(
            data['close'],
            fastperiod=macd_params['fast'],
            slowperiod=macd_params['slow'], 
            signalperiod=macd_params['signal']
        )
        result['macd'] = macd
        result['macd_signal'] = macd_signal
        result['macd_histogram'] = macd_hist
        
        # ADX (Average Directional Index)
        result['adx'] = talib.ADX(
            data['high'], data['low'], data['close'],
            timeperiod=self.config['trend_indicators']['adx_period']
        )
        
        # Parabolic SAR
        result['sar'] = talib.SAR(data['high'], data['low'])
        
        return result
    
    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillators."""
        result = data.copy()
        
        # RSI
        for period in self.config['momentum_indicators']['rsi_periods']:
            result[f'rsi_{period}'] = talib.RSI(data['close'], timeperiod=period)
            
        # Stochastic
        stoch_params = self.config['momentum_indicators']['stoch_params']
        slowk, slowd = talib.STOCH(
            data['high'], data['low'], data['close'],
            fastk_period=stoch_params['fastk'],
            slowk_period=stoch_params['slowk'],
            slowd_period=stoch_params['slowd']
        )
        result['stoch_k'] = slowk
        result['stoch_d'] = slowd
        
        # CCI (Commodity Channel Index)
        result['cci'] = talib.CCI(
            data['high'], data['low'], data['close'],
            timeperiod=self.config['momentum_indicators']['cci_period']
        )
        
        # Williams %R
        result['williams_r'] = talib.WILLR(
            data['high'], data['low'], data['close'],
            timeperiod=self.config['momentum_indicators']['williams_r_period']
        )
        
        # Rate of Change
        result['roc'] = talib.ROC(data['close'], timeperiod=10)
        
        return result
    
    def _add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        result = data.copy()
        
        # Bollinger Bands
        bb_period = self.config['volatility_indicators']['bb_period']
        bb_std = self.config['volatility_indicators']['bb_std']
        
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            data['close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std
        )
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle  
        result['bb_lower'] = bb_lower
        result['bb_width'] = (bb_upper - bb_lower) / bb_middle
        result['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Average True Range
        result['atr'] = talib.ATR(
            data['high'], data['low'], data['close'],
            timeperiod=self.config['volatility_indicators']['atr_period']
        )
        
        # Keltner Channels
        kelt_period = self.config['volatility_indicators']['keltner_period']
        kelt_mult = self.config['volatility_indicators']['keltner_multiplier']
        
        kelt_middle = talib.EMA(data['close'], timeperiod=kelt_period)
        kelt_atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=kelt_period)
        
        result['keltner_upper'] = kelt_middle + (kelt_mult * kelt_atr)
        result['keltner_lower'] = kelt_middle - (kelt_mult * kelt_atr)
        
        # Normalized ATR (volatility regime indicator)
        result['atr_normalized'] = result['atr'] / data['close']
        
        return result
    
    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        result = data.copy()
        
        # On Balance Volume
        if self.config['volume_indicators']['obv_enabled']:
            result['obv'] = talib.OBV(data['close'], data['volume'])
            
        # Volume Weighted Average Price
        for period in self.config['volume_indicators']['vwap_periods']:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            result[f'vwap_{period}'] = (
                (typical_price * data['volume']).rolling(period).sum() /
                data['volume'].rolling(period).sum()
            )
            
        # Money Flow Index
        result['mfi'] = talib.MFI(
            data['high'], data['low'], data['close'], data['volume'],
            timeperiod=self.config['volume_indicators']['mfi_period']
        )
        
        # Volume Rate of Change
        result['volume_roc'] = talib.ROC(data['volume'], timeperiod=10)
        
        # Price-Volume Trend
        result['pvt'] = ((data['close'] - data['close'].shift(1)) / 
                        data['close'].shift(1) * data['volume']).cumsum()
        
        return result
    
    def _add_crypto_specific_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add cryptocurrency-specific indicators."""
        result = data.copy()
        
        # Add placeholder columns for crypto-specific data that would come from external sources
        
        # Funding rate features (would be populated from exchange APIs)
        result['funding_rate'] = 0.0  # Placeholder
        result['funding_rate_ma'] = 0.0  # Placeholder
        result['funding_rate_std'] = 0.0  # Placeholder
        
        # Perpetual-spot basis
        result['perp_spot_basis'] = 0.0  # Placeholder
        result['basis_ma'] = 0.0  # Placeholder
        
        # Order book imbalance (would require level 2 data)
        result['order_book_imbalance'] = 0.0  # Placeholder
        
        # Large trade detection (whale movements)
        whale_threshold = self.config['crypto_specific']['whale_threshold']
        result['whale_trades'] = (data['volume'] * data['close'] > whale_threshold).astype(int)
        
        # Cross-exchange arbitrage opportunities (placeholder)
        result['arbitrage_opportunity'] = 0.0  # Placeholder
        
        return result
    
    def get_feature_importance(self, target: pd.Series, features: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance using mutual information.
        
        Args:
            target: Target variable (returns, signals, etc.)
            features: Feature matrix
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        from sklearn.feature_selection import mutual_info_regression
        
        # Remove NaN values
        valid_idx = ~(target.isna() | features.isna().any(axis=1))
        clean_target = target[valid_idx]
        clean_features = features[valid_idx]
        
        if len(clean_features) == 0:
            return {}
            
        importance_scores = mutual_info_regression(clean_features, clean_target)
        
        return dict(zip(features.columns, importance_scores))
    
    def analyze_regime_dependency(self, data: pd.DataFrame, 
                                 regime_column: str = 'regime') -> Dict[str, Dict]:
        """
        Analyze how indicator effectiveness varies by market regime.
        
        Args:
            data: DataFrame with features and regime labels
            regime_column: Name of column containing regime labels
            
        Returns:
            Dictionary with regime-specific statistics for each indicator
        """
        if regime_column not in data.columns:
            logger.warning(f"Regime column '{regime_column}' not found in data")
            return {}
            
        results = {}
        indicator_columns = [col for col in data.columns if col.startswith(
            ('sma_', 'ema_', 'rsi_', 'bb_', 'atr', 'macd', 'stoch_')
        )]
        
        for indicator in indicator_columns:
            results[indicator] = {}
            for regime in data[regime_column].unique():
                regime_data = data[data[regime_column] == regime][indicator]
                results[indicator][regime] = {
                    'mean': regime_data.mean(),
                    'std': regime_data.std(), 
                    'min': regime_data.min(),
                    'max': regime_data.max(),
                    'count': len(regime_data)
                }
                
        return results