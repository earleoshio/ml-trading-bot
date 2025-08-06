"""
Pattern Recognition Module

Implements advanced pattern recognition for cryptocurrency trading,
including candlestick patterns, chart patterns, and support/resistance levels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
import talib
from scipy import signal
from scipy.stats import linregress


class PatternRecognition:
    """
    Pattern recognition system for cryptocurrency trading.
    
    Detects and quantifies various market patterns including:
    - Candlestick patterns
    - Chart patterns (triangles, flags, head and shoulders)
    - Support and resistance levels
    - Trend lines and channels
    - Fractals and geometric patterns
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pattern recognition system.
        
        Args:
            config: Configuration dictionary with pattern detection parameters
        """
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default configuration for pattern recognition."""
        return {
            'candlestick': {
                'body_threshold': 0.001,  # Minimum body size as fraction of price
                'wick_threshold': 0.0005,  # Minimum wick size
                'doji_threshold': 0.0002   # Maximum body size for doji
            },
            'support_resistance': {
                'lookback_periods': [20, 50, 100],
                'touch_threshold': 0.002,  # Price tolerance for level touches
                'min_touches': 2,
                'strength_periods': [5, 10, 20]
            },
            'chart_patterns': {
                'triangle_min_periods': 20,
                'triangle_max_periods': 100,
                'flag_min_periods': 5,
                'flag_max_periods': 20,
                'head_shoulders_periods': 50
            },
            'trend_lines': {
                'min_periods': 10,
                'max_periods': 100,
                'r_squared_threshold': 0.7,
                'angle_threshold': 0.1  # Minimum trend line slope
            },
            'fractals': {
                'williams_fractal_periods': 5,
                'alligator_jaw_period': 13,
                'alligator_teeth_period': 8,
                'alligator_lips_period': 5
            }
        }
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate pattern recognition features from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with pattern recognition features added
        """
        logger.info("Generating pattern recognition features...")
        
        result_df = data.copy()
        
        # Add candlestick pattern features
        result_df = self._add_candlestick_patterns(result_df)
        
        # Add support and resistance levels
        result_df = self._add_support_resistance_features(result_df)
        
        # Add chart pattern detection
        result_df = self._add_chart_patterns(result_df)
        
        # Add trend line analysis
        result_df = self._add_trend_line_features(result_df)
        
        # Add fractal analysis
        result_df = self._add_fractal_features(result_df)
        
        # Add geometric pattern features
        result_df = self._add_geometric_patterns(result_df)
        
        logger.info(f"Generated {len(result_df.columns) - len(data.columns)} pattern recognition features")
        return result_df
    
    def _add_candlestick_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition features."""
        result = data.copy()
        
        # Basic candlestick patterns using TA-Lib
        candlestick_patterns = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'hanging_man': talib.CDLHANGINGMAN,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'engulfing_bull': talib.CDLENGULFING,
            'engulfing_bear': lambda o, h, l, c: -talib.CDLENGULFING(o, h, l, c),
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'harami': talib.CDLHARAMI,
            'dark_cloud': talib.CDLDARKCLOUDCOVER,
            'piercing': talib.CDLPIERCING
        }
        
        for pattern_name, pattern_func in candlestick_patterns.items():
            try:
                result[f'candle_{pattern_name}'] = pattern_func(
                    data['open'], data['high'], data['low'], data['close']
                )
            except:
                result[f'candle_{pattern_name}'] = 0
        
        # Custom candlestick analysis
        body_size = np.abs(data['close'] - data['open'])
        upper_wick = data['high'] - np.maximum(data['open'], data['close'])
        lower_wick = np.minimum(data['open'], data['close']) - data['low']
        total_range = data['high'] - data['low']
        
        # Candlestick metrics
        result['body_ratio'] = body_size / (total_range + 1e-8)
        result['upper_wick_ratio'] = upper_wick / (total_range + 1e-8)
        result['lower_wick_ratio'] = lower_wick / (total_range + 1e-8)
        result['candle_color'] = (data['close'] > data['open']).astype(int)
        
        # Consecutive candle patterns
        result['consecutive_green'] = self._count_consecutive(result['candle_color'], 1)
        result['consecutive_red'] = self._count_consecutive(result['candle_color'], 0)
        
        return result
    
    def _add_support_resistance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance level features."""
        result = data.copy()
        
        for lookback in self.config['support_resistance']['lookback_periods']:
            # Find local minima (support) and maxima (resistance)
            support_levels = self._find_support_resistance(data['low'], lookback, 'support')
            resistance_levels = self._find_support_resistance(data['high'], lookback, 'resistance')
            
            result[f'support_{lookback}'] = support_levels
            result[f'resistance_{lookback}'] = resistance_levels
            
            # Distance to nearest support/resistance
            result[f'dist_to_support_{lookback}'] = (data['close'] - support_levels) / data['close']
            result[f'dist_to_resistance_{lookback}'] = (resistance_levels - data['close']) / data['close']
            
            # Support/resistance strength
            result[f'support_strength_{lookback}'] = self._calculate_level_strength(
                data, support_levels, 'support'
            )
            result[f'resistance_strength_{lookback}'] = self._calculate_level_strength(
                data, resistance_levels, 'resistance'
            )
        
        return result
    
    def _add_chart_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add chart pattern detection features."""
        result = data.copy()
        
        # Triangle patterns
        result['ascending_triangle'] = self._detect_triangle_pattern(data, 'ascending')
        result['descending_triangle'] = self._detect_triangle_pattern(data, 'descending')
        result['symmetric_triangle'] = self._detect_triangle_pattern(data, 'symmetric')
        
        # Flag and pennant patterns
        result['bull_flag'] = self._detect_flag_pattern(data, 'bull')
        result['bear_flag'] = self._detect_flag_pattern(data, 'bear')
        
        # Head and shoulders patterns
        result['head_shoulders'] = self._detect_head_shoulders(data, 'normal')
        result['inverse_head_shoulders'] = self._detect_head_shoulders(data, 'inverse')
        
        # Double top/bottom patterns
        result['double_top'] = self._detect_double_pattern(data, 'top')
        result['double_bottom'] = self._detect_double_pattern(data, 'bottom')
        
        # Wedge patterns
        result['rising_wedge'] = self._detect_wedge_pattern(data, 'rising')
        result['falling_wedge'] = self._detect_wedge_pattern(data, 'falling')
        
        return result
    
    def _add_trend_line_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend line analysis features."""
        result = data.copy()
        
        for period in range(self.config['trend_lines']['min_periods'], 
                          self.config['trend_lines']['max_periods'], 10):
            
            # Upper trend line (resistance trend)
            upper_trend = self._fit_trend_line(data['high'], period, 'upper')
            result[f'upper_trend_{period}'] = upper_trend['values']
            result[f'upper_trend_slope_{period}'] = upper_trend['slope']
            result[f'upper_trend_r2_{period}'] = upper_trend['r_squared']
            
            # Lower trend line (support trend)
            lower_trend = self._fit_trend_line(data['low'], period, 'lower')
            result[f'lower_trend_{period}'] = lower_trend['values']
            result[f'lower_trend_slope_{period}'] = lower_trend['slope']
            result[f'lower_trend_r2_{period}'] = lower_trend['r_squared']
            
            # Channel analysis
            channel_width = upper_trend['values'] - lower_trend['values']
            result[f'channel_width_{period}'] = channel_width / data['close']
            result[f'channel_position_{period}'] = (
                (data['close'] - lower_trend['values']) / (channel_width + 1e-8)
            )
        
        return result
    
    def _add_fractal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add fractal analysis features."""
        result = data.copy()
        
        # Williams Fractals
        fractal_period = self.config['fractals']['williams_fractal_periods']
        result['fractal_high'] = self._detect_williams_fractals(data['high'], fractal_period, 'high')
        result['fractal_low'] = self._detect_williams_fractals(data['low'], fractal_period, 'low')
        
        # Alligator indicator (Bill Williams)
        jaw_period = self.config['fractals']['alligator_jaw_period']
        teeth_period = self.config['fractals']['alligator_teeth_period']
        lips_period = self.config['fractals']['alligator_lips_period']
        
        result['alligator_jaw'] = talib.SMA(data['close'], timeperiod=jaw_period).shift(8)
        result['alligator_teeth'] = talib.SMA(data['close'], timeperiod=teeth_period).shift(5)
        result['alligator_lips'] = talib.SMA(data['close'], timeperiod=lips_period).shift(3)
        
        # Alligator sleeping/hunting phases
        jaw_teeth_diff = result['alligator_jaw'] - result['alligator_teeth']
        teeth_lips_diff = result['alligator_teeth'] - result['alligator_lips']
        
        result['alligator_sleeping'] = (
            (np.abs(jaw_teeth_diff) < data['close'] * 0.001) &
            (np.abs(teeth_lips_diff) < data['close'] * 0.001)
        ).astype(int)
        
        # Awesome Oscillator
        result['awesome_oscillator'] = (
            talib.SMA((data['high'] + data['low']) / 2, timeperiod=5) -
            talib.SMA((data['high'] + data['low']) / 2, timeperiod=34)
        )
        
        return result
    
    def _add_geometric_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add geometric pattern features."""
        result = data.copy()
        
        # Fibonacci retracements
        for period in [20, 50, 100]:
            high_period = data['high'].rolling(period).max()
            low_period = data['low'].rolling(period).min()
            range_period = high_period - low_period
            
            # Common Fibonacci levels
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            for level in fib_levels:
                fib_retracement = high_period - (range_period * level)
                result[f'fib_{level}_{period}'] = fib_retracement
                result[f'dist_to_fib_{level}_{period}'] = np.abs(data['close'] - fib_retracement) / data['close']
        
        # Golden ratio analysis
        result['golden_ratio_ma'] = talib.EMA(data['close'], timeperiod=21) / talib.EMA(data['close'], timeperiod=13)
        
        # Harmonic patterns (simplified Gartley detection)
        result['gartley_pattern'] = self._detect_gartley_pattern(data)
        
        return result
    
    def _count_consecutive(self, series: pd.Series, value: Union[int, float]) -> pd.Series:
        """Count consecutive occurrences of a value."""
        consecutive = series.copy().astype(float)
        consecutive[series != value] = 0
        
        result = pd.Series(0, index=series.index)
        count = 0
        for i in range(len(series)):
            if series.iloc[i] == value:
                count += 1
                result.iloc[i] = count
            else:
                count = 0
        
        return result
    
    def _find_support_resistance(self, prices: pd.Series, lookback: int, 
                                level_type: str) -> pd.Series:
        """Find support or resistance levels using local extrema."""
        if level_type == 'support':
            extrema = prices.rolling(lookback, center=True).min()
            condition = prices == extrema
        else:  # resistance
            extrema = prices.rolling(lookback, center=True).max()
            condition = prices == extrema
        
        levels = pd.Series(np.nan, index=prices.index)
        levels[condition] = prices[condition]
        levels = levels.fillna(method='ffill')
        
        return levels
    
    def _calculate_level_strength(self, data: pd.DataFrame, levels: pd.Series, 
                                level_type: str) -> pd.Series:
        """Calculate strength of support/resistance levels."""
        strength = pd.Series(0.0, index=data.index)
        threshold = self.config['support_resistance']['touch_threshold']
        
        for i in range(len(data)):
            if pd.isna(levels.iloc[i]):
                continue
                
            level = levels.iloc[i]
            start_idx = max(0, i - 20)
            
            if level_type == 'support':
                touches = np.sum(np.abs(data['low'].iloc[start_idx:i+1] - level) < level * threshold)
            else:
                touches = np.sum(np.abs(data['high'].iloc[start_idx:i+1] - level) < level * threshold)
            
            strength.iloc[i] = touches
        
        return strength
    
    def _detect_triangle_pattern(self, data: pd.DataFrame, triangle_type: str) -> pd.Series:
        """Detect triangle patterns."""
        # Simplified triangle detection - would need more sophisticated implementation
        result = pd.Series(0, index=data.index)
        
        # This is a placeholder - real implementation would analyze trend lines
        # and convergence patterns over the specified periods
        
        return result
    
    def _detect_flag_pattern(self, data: pd.DataFrame, flag_type: str) -> pd.Series:
        """Detect flag patterns."""
        # Simplified flag detection - would need more sophisticated implementation
        result = pd.Series(0, index=data.index)
        
        # This is a placeholder - real implementation would analyze price channels
        # following strong moves
        
        return result
    
    def _detect_head_shoulders(self, data: pd.DataFrame, pattern_type: str) -> pd.Series:
        """Detect head and shoulders patterns."""
        # Simplified head and shoulders detection
        result = pd.Series(0, index=data.index)
        
        # This is a placeholder - real implementation would identify three peaks/troughs
        # with specific height relationships
        
        return result
    
    def _detect_double_pattern(self, data: pd.DataFrame, pattern_type: str) -> pd.Series:
        """Detect double top/bottom patterns."""
        # Simplified double pattern detection
        result = pd.Series(0, index=data.index)
        
        # This is a placeholder - real implementation would identify two similar
        # peaks/troughs at approximately same level
        
        return result
    
    def _detect_wedge_pattern(self, data: pd.DataFrame, wedge_type: str) -> pd.Series:
        """Detect wedge patterns."""
        # Simplified wedge detection
        result = pd.Series(0, index=data.index)
        
        # This is a placeholder - real implementation would analyze converging
        # trend lines with specific slope characteristics
        
        return result
    
    def _fit_trend_line(self, prices: pd.Series, period: int, line_type: str) -> Dict:
        """Fit trend line to price data."""
        result = {
            'values': pd.Series(np.nan, index=prices.index),
            'slope': pd.Series(np.nan, index=prices.index),
            'r_squared': pd.Series(np.nan, index=prices.index)
        }
        
        for i in range(period, len(prices)):
            y = prices.iloc[i-period:i].values
            x = np.arange(len(y))
            
            if len(y) > 1:
                slope, intercept, r_value, _, _ = linregress(x, y)
                
                result['slope'].iloc[i] = slope
                result['r_squared'].iloc[i] = r_value ** 2
                result['values'].iloc[i] = slope * (len(y) - 1) + intercept
        
        return result
    
    def _detect_williams_fractals(self, prices: pd.Series, period: int, 
                                 fractal_type: str) -> pd.Series:
        """Detect Williams fractals."""
        fractals = pd.Series(0, index=prices.index)
        
        for i in range(period, len(prices) - period):
            window = prices.iloc[i-period:i+period+1]
            
            if fractal_type == 'high':
                if prices.iloc[i] == window.max():
                    fractals.iloc[i] = 1
            else:  # low
                if prices.iloc[i] == window.min():
                    fractals.iloc[i] = 1
        
        return fractals
    
    def _detect_gartley_pattern(self, data: pd.DataFrame) -> pd.Series:
        """Detect simplified Gartley harmonic patterns."""
        # Simplified Gartley detection - would need more sophisticated harmonic analysis
        result = pd.Series(0, index=data.index)
        
        # This is a placeholder for complex harmonic pattern detection
        # Real implementation would analyze specific Fibonacci relationships
        
        return result