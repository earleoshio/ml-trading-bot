"""
Trend Detection Module

Implements comprehensive trend detection for cryptocurrency markets,
including trend strength analysis, trend reversal detection, and 
multi-timeframe trend analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.signal import find_peaks, savgol_filter


class TrendDetector:
    """
    Advanced trend detection system for cryptocurrency markets.
    
    Provides comprehensive trend analysis including:
    - Trend direction and strength detection
    - Support and resistance level identification
    - Trend reversal pattern recognition
    - Multi-timeframe trend analysis
    - Momentum-based trend confirmation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trend detection system.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or self._default_config()
        self.trend_history = []
        self.support_resistance_levels = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for trend detection."""
        return {
            'trend_detection': {
                'methods': ['linear_regression', 'moving_average', 'peak_trough'],
                'lookback_periods': [20, 50, 100, 200],
                'strength_threshold': 0.6,  # Minimum R² for trend significance
                'angle_threshold': 0.1  # Minimum slope for trend identification
            },
            'moving_averages': {
                'short_period': 20,
                'medium_period': 50,
                'long_period': 200,
                'ema_alpha': 0.1
            },
            'support_resistance': {
                'window_size': 20,
                'min_touches': 2,
                'touch_tolerance': 0.02,  # 2% tolerance
                'strength_periods': [10, 20, 50]
            },
            'reversal_detection': {
                'enabled': True,
                'divergence_threshold': 0.3,
                'volume_confirmation': True,
                'pattern_recognition': True
            },
            'multi_timeframe': {
                'timeframes': ['1h', '4h', '1d'],
                'consensus_weight': 0.7,  # Weight for timeframe consensus
                'conflict_resolution': 'higher_timeframe'  # 'higher_timeframe', 'majority', 'weighted'
            }
        }
    
    def detect_trend(self, price_data: pd.DataFrame, 
                    method: str = 'ensemble') -> Dict[str, Union[str, float, Dict]]:
        """
        Detect overall trend direction and characteristics.
        
        Args:
            price_data: DataFrame with OHLCV data
            method: Detection method ('linear_regression', 'moving_average', 'ensemble')
            
        Returns:
            Dictionary with trend information
        """
        logger.info(f"Detecting trend using {method} method...")
        
        if method == 'ensemble':
            return self._ensemble_trend_detection(price_data)
        elif method == 'linear_regression':
            return self._linear_regression_trend(price_data)
        elif method == 'moving_average':
            return self._moving_average_trend(price_data)
        elif method == 'peak_trough':
            return self._peak_trough_trend(price_data)
        else:
            logger.warning(f"Unknown method {method}, using ensemble")
            return self._ensemble_trend_detection(price_data)
    
    def _ensemble_trend_detection(self, price_data: pd.DataFrame) -> Dict:
        """Combine multiple trend detection methods."""
        
        methods = self.config['trend_detection']['methods']
        trend_results = {}
        
        # Apply each method
        if 'linear_regression' in methods:
            trend_results['linear_regression'] = self._linear_regression_trend(price_data)
        
        if 'moving_average' in methods:
            trend_results['moving_average'] = self._moving_average_trend(price_data)
        
        if 'peak_trough' in methods:
            trend_results['peak_trough'] = self._peak_trough_trend(price_data)
        
        # Combine results
        ensemble_result = self._combine_trend_results(trend_results)
        
        return ensemble_result
    
    def _linear_regression_trend(self, price_data: pd.DataFrame) -> Dict:
        """Detect trend using linear regression on multiple timeframes."""
        
        results = {}
        prices = price_data['close']
        
        for period in self.config['trend_detection']['lookback_periods']:
            if len(prices) < period:
                continue
            
            # Get recent data
            recent_prices = prices.tail(period)
            x = np.arange(len(recent_prices)).reshape(-1, 1)
            y = recent_prices.values
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(x, y)
            
            # Calculate metrics
            slope = model.coef_[0]
            r_squared = model.score(x, y)
            
            # Determine trend direction and strength
            if r_squared >= self.config['trend_detection']['strength_threshold']:
                if abs(slope) >= self.config['trend_detection']['angle_threshold']:
                    direction = 'uptrend' if slope > 0 else 'downtrend'
                    strength = min(r_squared, 1.0)
                else:
                    direction = 'sideways'
                    strength = 1 - r_squared  # Inverse for sideways
            else:
                direction = 'unclear'
                strength = r_squared
            
            results[f'period_{period}'] = {
                'direction': direction,
                'strength': strength,
                'slope': slope,
                'r_squared': r_squared
            }
        
        # Overall assessment
        directions = [r['direction'] for r in results.values() if r['direction'] != 'unclear']
        if directions:
            direction_counts = {d: directions.count(d) for d in set(directions)}
            overall_direction = max(direction_counts, key=direction_counts.get)
            overall_strength = np.mean([r['strength'] for r in results.values()])
        else:
            overall_direction = 'unclear'
            overall_strength = 0.0
        
        return {
            'method': 'linear_regression',
            'overall_direction': overall_direction,
            'overall_strength': overall_strength,
            'period_results': results
        }
    
    def _moving_average_trend(self, price_data: pd.DataFrame) -> Dict:
        """Detect trend using moving averages."""
        
        prices = price_data['close']
        
        # Calculate moving averages
        short_ma = prices.rolling(self.config['moving_averages']['short_period']).mean()
        medium_ma = prices.rolling(self.config['moving_averages']['medium_period']).mean()
        long_ma = prices.rolling(self.config['moving_averages']['long_period']).mean()
        
        # Current values
        current_price = prices.iloc[-1]
        current_short_ma = short_ma.iloc[-1]
        current_medium_ma = medium_ma.iloc[-1] 
        current_long_ma = long_ma.iloc[-1]
        
        # Trend determination
        ma_alignment = 0
        
        # Check MA alignment for uptrend
        if current_price > current_short_ma > current_medium_ma > current_long_ma:
            direction = 'uptrend'
            ma_alignment = 3  # All aligned
        elif current_price > current_short_ma > current_medium_ma:
            direction = 'uptrend'
            ma_alignment = 2
        elif current_price > current_short_ma:
            direction = 'uptrend'
            ma_alignment = 1
        
        # Check MA alignment for downtrend
        elif current_price < current_short_ma < current_medium_ma < current_long_ma:
            direction = 'downtrend'
            ma_alignment = 3
        elif current_price < current_short_ma < current_medium_ma:
            direction = 'downtrend'
            ma_alignment = 2
        elif current_price < current_short_ma:
            direction = 'downtrend'
            ma_alignment = 1
        
        else:
            direction = 'sideways'
            ma_alignment = 0
        
        # Calculate strength based on MA separation
        ma_separation = abs(current_short_ma - current_long_ma) / current_price
        strength = min(ma_separation * 10, 1.0)  # Normalize
        
        # MA crossover signals
        short_medium_cross = self._detect_ma_crossover(short_ma, medium_ma)
        medium_long_cross = self._detect_ma_crossover(medium_ma, long_ma)
        
        return {
            'method': 'moving_average',
            'overall_direction': direction,
            'overall_strength': strength,
            'ma_alignment': ma_alignment,
            'ma_values': {
                'short_ma': current_short_ma,
                'medium_ma': current_medium_ma,
                'long_ma': current_long_ma
            },
            'crossovers': {
                'short_medium': short_medium_cross,
                'medium_long': medium_long_cross
            }
        }
    
    def _peak_trough_trend(self, price_data: pd.DataFrame) -> Dict:
        """Detect trend using peak and trough analysis."""
        
        prices = price_data['close']
        
        # Smooth prices to reduce noise
        smoothed_prices = savgol_filter(prices, window_length=min(21, len(prices)//4), polyorder=3)
        
        # Find peaks and troughs
        peaks, peak_properties = find_peaks(smoothed_prices, distance=10, prominence=smoothed_prices.std()/2)
        troughs, trough_properties = find_peaks(-smoothed_prices, distance=10, prominence=smoothed_prices.std()/2)
        
        if len(peaks) < 2 or len(troughs) < 2:
            return {
                'method': 'peak_trough',
                'overall_direction': 'unclear',
                'overall_strength': 0.0,
                'peaks_count': len(peaks),
                'troughs_count': len(troughs)
            }
        
        # Analyze peak and trough progression
        recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
        recent_troughs = troughs[-3:] if len(troughs) >= 3 else troughs
        
        # Higher highs and higher lows = uptrend
        peak_trend = self._analyze_level_progression(smoothed_prices[recent_peaks])
        trough_trend = self._analyze_level_progression(smoothed_prices[recent_troughs])
        
        # Combine peak and trough analysis
        if peak_trend == 'rising' and trough_trend == 'rising':
            direction = 'uptrend'
            strength = 0.8
        elif peak_trend == 'falling' and trough_trend == 'falling':
            direction = 'downtrend'
            strength = 0.8
        elif peak_trend == 'rising' and trough_trend == 'falling':
            direction = 'expanding_range'
            strength = 0.6
        elif peak_trend == 'falling' and trough_trend == 'rising':
            direction = 'contracting_range'
            strength = 0.4
        else:
            direction = 'sideways'
            strength = 0.3
        
        return {
            'method': 'peak_trough',
            'overall_direction': direction,
            'overall_strength': strength,
            'peaks_count': len(peaks),
            'troughs_count': len(troughs),
            'peak_trend': peak_trend,
            'trough_trend': trough_trend,
            'recent_peaks': recent_peaks.tolist(),
            'recent_troughs': recent_troughs.tolist()
        }
    
    def _analyze_level_progression(self, levels: np.ndarray) -> str:
        """Analyze if levels are rising, falling, or sideways."""
        
        if len(levels) < 2:
            return 'unclear'
        
        # Calculate trend of levels
        x = np.arange(len(levels))
        slope, _, r_value, _, _ = stats.linregress(x, levels)
        
        if r_value ** 2 < 0.5:  # Low R²
            return 'sideways'
        elif slope > 0:
            return 'rising'
        else:
            return 'falling'
    
    def _detect_ma_crossover(self, ma1: pd.Series, ma2: pd.Series) -> str:
        """Detect moving average crossover."""
        
        if len(ma1) < 2 or len(ma2) < 2:
            return 'none'
        
        # Check recent crossover
        current_diff = ma1.iloc[-1] - ma2.iloc[-1]
        previous_diff = ma1.iloc[-2] - ma2.iloc[-2]
        
        if previous_diff <= 0 < current_diff:
            return 'bullish_cross'
        elif previous_diff >= 0 > current_diff:
            return 'bearish_cross'
        else:
            return 'none'
    
    def _combine_trend_results(self, trend_results: Dict) -> Dict:
        """Combine results from multiple trend detection methods."""
        
        if not trend_results:
            return {
                'method': 'ensemble',
                'overall_direction': 'unclear',
                'overall_strength': 0.0,
                'consensus': 0.0,
                'individual_results': {}
            }
        
        # Collect directions and strengths
        directions = []
        strengths = []
        
        for method_name, result in trend_results.items():
            if result['overall_direction'] != 'unclear':
                directions.append(result['overall_direction'])
                strengths.append(result['overall_strength'])
        
        if not directions:
            return {
                'method': 'ensemble',
                'overall_direction': 'unclear',
                'overall_strength': 0.0,
                'consensus': 0.0,
                'individual_results': trend_results
            }
        
        # Calculate consensus
        direction_counts = {d: directions.count(d) for d in set(directions)}
        most_common_direction = max(direction_counts, key=direction_counts.get)
        consensus = direction_counts[most_common_direction] / len(directions)
        
        # Average strength
        overall_strength = np.mean(strengths)
        
        return {
            'method': 'ensemble',
            'overall_direction': most_common_direction,
            'overall_strength': overall_strength,
            'consensus': consensus,
            'direction_distribution': direction_counts,
            'individual_results': trend_results
        }
    
    def identify_support_resistance(self, price_data: pd.DataFrame, 
                                  level_type: str = 'both') -> Dict:
        """
        Identify support and resistance levels.
        
        Args:
            price_data: OHLCV price data
            level_type: 'support', 'resistance', or 'both'
            
        Returns:
            Dictionary with identified levels
        """
        
        levels = {}
        
        if level_type in ['support', 'both']:
            levels['support'] = self._find_support_levels(price_data)
        
        if level_type in ['resistance', 'both']:
            levels['resistance'] = self._find_resistance_levels(price_data)
        
        return levels
    
    def _find_support_levels(self, price_data: pd.DataFrame) -> List[Dict]:
        """Find support levels in price data."""
        
        lows = price_data['low']
        window = self.config['support_resistance']['window_size']
        tolerance = self.config['support_resistance']['touch_tolerance']
        min_touches = self.config['support_resistance']['min_touches']
        
        support_levels = []
        
        # Find local minima
        local_minima = []
        for i in range(window, len(lows) - window):
            if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                local_minima.append((i, lows.iloc[i]))
        
        # Group similar levels
        for i, (idx1, level1) in enumerate(local_minima):
            touches = [(idx1, level1)]
            
            for j, (idx2, level2) in enumerate(local_minima[i+1:], i+1):
                if abs(level2 - level1) / level1 <= tolerance:
                    touches.append((idx2, level2))
            
            if len(touches) >= min_touches:
                avg_level = np.mean([touch[1] for touch in touches])
                strength = self._calculate_level_strength(price_data, avg_level, 'support')
                
                support_levels.append({
                    'level': avg_level,
                    'strength': strength,
                    'touches': len(touches),
                    'first_touch_idx': min(touch[0] for touch in touches),
                    'last_touch_idx': max(touch[0] for touch in touches)
                })
        
        # Sort by strength
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return support_levels
    
    def _find_resistance_levels(self, price_data: pd.DataFrame) -> List[Dict]:
        """Find resistance levels in price data."""
        
        highs = price_data['high']
        window = self.config['support_resistance']['window_size']
        tolerance = self.config['support_resistance']['touch_tolerance']
        min_touches = self.config['support_resistance']['min_touches']
        
        resistance_levels = []
        
        # Find local maxima
        local_maxima = []
        for i in range(window, len(highs) - window):
            if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                local_maxima.append((i, highs.iloc[i]))
        
        # Group similar levels
        for i, (idx1, level1) in enumerate(local_maxima):
            touches = [(idx1, level1)]
            
            for j, (idx2, level2) in enumerate(local_maxima[i+1:], i+1):
                if abs(level2 - level1) / level1 <= tolerance:
                    touches.append((idx2, level2))
            
            if len(touches) >= min_touches:
                avg_level = np.mean([touch[1] for touch in touches])
                strength = self._calculate_level_strength(price_data, avg_level, 'resistance')
                
                resistance_levels.append({
                    'level': avg_level,
                    'strength': strength,
                    'touches': len(touches),
                    'first_touch_idx': min(touch[0] for touch in touches),
                    'last_touch_idx': max(touch[0] for touch in touches)
                })
        
        # Sort by strength
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return resistance_levels
    
    def _calculate_level_strength(self, price_data: pd.DataFrame, 
                                 level: float, level_type: str) -> float:
        """Calculate the strength of a support/resistance level."""
        
        strength_periods = self.config['support_resistance']['strength_periods']
        tolerance = self.config['support_resistance']['touch_tolerance']
        
        total_strength = 0.0
        
        for period in strength_periods:
            if len(price_data) < period:
                continue
            
            recent_data = price_data.tail(period)
            
            if level_type == 'support':
                # Count how many times price approached but didn't break support
                approaches = ((recent_data['low'] <= level * (1 + tolerance)) & 
                            (recent_data['close'] > level)).sum()
                breaks = (recent_data['low'] < level * (1 - tolerance)).sum()
            else:  # resistance
                # Count how many times price approached but didn't break resistance
                approaches = ((recent_data['high'] >= level * (1 - tolerance)) & 
                            (recent_data['close'] < level)).sum()
                breaks = (recent_data['high'] > level * (1 + tolerance)).sum()
            
            # Strength increases with approaches and decreases with breaks
            period_strength = max(0, approaches - breaks * 2) / period
            total_strength += period_strength
        
        return total_strength / len(strength_periods)
    
    def detect_trend_reversal(self, price_data: pd.DataFrame) -> Dict:
        """Detect potential trend reversal signals."""
        
        if not self.config['reversal_detection']['enabled']:
            return {'reversal_signals': [], 'reversal_probability': 0.0}
        
        reversal_signals = []
        
        # Divergence analysis
        divergence = self._detect_divergence(price_data)
        if divergence['detected']:
            reversal_signals.append(divergence)
        
        # Volume confirmation
        if self.config['reversal_detection']['volume_confirmation']:
            volume_signal = self._analyze_volume_reversal(price_data)
            if volume_signal['signal']:
                reversal_signals.append(volume_signal)
        
        # Pattern recognition
        if self.config['reversal_detection']['pattern_recognition']:
            pattern_signals = self._detect_reversal_patterns(price_data)
            reversal_signals.extend(pattern_signals)
        
        # Calculate overall reversal probability
        if reversal_signals:
            signal_strengths = [signal.get('strength', 0.5) for signal in reversal_signals]
            reversal_probability = min(1.0, np.mean(signal_strengths) * len(reversal_signals) / 3)
        else:
            reversal_probability = 0.0
        
        return {
            'reversal_signals': reversal_signals,
            'reversal_probability': reversal_probability,
            'signal_count': len(reversal_signals)
        }
    
    def _detect_divergence(self, price_data: pd.DataFrame) -> Dict:
        """Detect price-momentum divergence."""
        
        prices = price_data['close']
        
        # Calculate momentum (RSI proxy)
        returns = prices.pct_change()
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        if len(prices) < 50 or len(rsi) < 50:
            return {'detected': False, 'type': None, 'strength': 0.0}
        
        # Look for divergence in recent periods
        recent_period = 20
        recent_prices = prices.tail(recent_period)
        recent_rsi = rsi.tail(recent_period)
        
        # Price trend
        price_trend = self._get_simple_trend(recent_prices)
        rsi_trend = self._get_simple_trend(recent_rsi)
        
        # Check for divergence
        if price_trend == 'rising' and rsi_trend == 'falling':
            return {
                'detected': True,
                'type': 'bearish_divergence',
                'strength': 0.7,
                'description': 'Price making higher highs while momentum declining'
            }
        elif price_trend == 'falling' and rsi_trend == 'rising':
            return {
                'detected': True,
                'type': 'bullish_divergence',
                'strength': 0.7,
                'description': 'Price making lower lows while momentum improving'
            }
        
        return {'detected': False, 'type': None, 'strength': 0.0}
    
    def _get_simple_trend(self, series: pd.Series) -> str:
        """Get simple trend direction for a series."""
        
        if len(series) < 2:
            return 'unclear'
        
        first_half = series.iloc[:len(series)//2].mean()
        second_half = series.iloc[len(series)//2:].mean()
        
        if second_half > first_half * 1.02:
            return 'rising'
        elif second_half < first_half * 0.98:
            return 'falling'
        else:
            return 'sideways'
    
    def _analyze_volume_reversal(self, price_data: pd.DataFrame) -> Dict:
        """Analyze volume for reversal confirmation."""
        
        if 'volume' not in price_data.columns:
            return {'signal': False, 'strength': 0.0}
        
        volumes = price_data['volume']
        prices = price_data['close']
        
        # Calculate volume moving average
        vol_ma = volumes.rolling(20).mean()
        
        # Look for volume spikes during price extremes
        recent_vol = volumes.tail(5)
        recent_vol_ma = vol_ma.tail(5)
        
        # High volume on reversal
        volume_spike = (recent_vol > recent_vol_ma * 1.5).any()
        
        if volume_spike:
            return {
                'signal': True,
                'type': 'volume_reversal',
                'strength': 0.6,
                'description': 'High volume detected during potential reversal'
            }
        
        return {'signal': False, 'strength': 0.0}
    
    def _detect_reversal_patterns(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect candlestick reversal patterns."""
        
        patterns = []
        
        if len(price_data) < 10:
            return patterns
        
        recent_data = price_data.tail(10)
        
        # Simple pattern detection (placeholder for more sophisticated patterns)
        
        # Doji detection
        for i in range(len(recent_data)):
            candle = recent_data.iloc[i]
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if body_size < total_range * 0.1:  # Small body relative to range
                patterns.append({
                    'type': 'doji',
                    'strength': 0.5,
                    'description': 'Doji pattern detected - potential reversal'
                })
        
        # Hammer/Shooting star detection
        last_candle = recent_data.iloc[-1]
        body_size = abs(last_candle['close'] - last_candle['open'])
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        total_range = last_candle['high'] - last_candle['low']
        
        if lower_wick > body_size * 2 and upper_wick < body_size:
            patterns.append({
                'type': 'hammer',
                'strength': 0.6,
                'description': 'Hammer pattern - potential bullish reversal'
            })
        elif upper_wick > body_size * 2 and lower_wick < body_size:
            patterns.append({
                'type': 'shooting_star',
                'strength': 0.6,
                'description': 'Shooting star pattern - potential bearish reversal'
            })
        
        return patterns
    
    def multi_timeframe_analysis(self, price_data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Perform multi-timeframe trend analysis."""
        
        if not self.config['multi_timeframe']:
            return {'analysis': 'Multi-timeframe analysis disabled'}
        
        timeframe_results = {}
        
        # Analyze each timeframe
        for timeframe, data in price_data_dict.items():
            timeframe_results[timeframe] = self.detect_trend(data, method='ensemble')
        
        # Combine results
        combined_analysis = self._combine_multi_timeframe_results(timeframe_results)
        
        return combined_analysis
    
    def _combine_multi_timeframe_results(self, timeframe_results: Dict) -> Dict:
        """Combine multi-timeframe analysis results."""
        
        if not timeframe_results:
            return {'overall_direction': 'unclear', 'confidence': 0.0}
        
        # Extract directions and strengths
        directions = []
        strengths = []
        
        for tf, result in timeframe_results.items():
            if result.get('overall_direction') != 'unclear':
                directions.append(result['overall_direction'])
                strengths.append(result['overall_strength'])
        
        if not directions:
            return {
                'overall_direction': 'unclear',
                'confidence': 0.0,
                'timeframe_results': timeframe_results
            }
        
        # Calculate consensus
        direction_counts = {d: directions.count(d) for d in set(directions)}
        consensus_direction = max(direction_counts, key=direction_counts.get)
        consensus_ratio = direction_counts[consensus_direction] / len(directions)
        
        # Weight by consensus threshold
        consensus_weight = self.config['multi_timeframe']['consensus_weight']
        confidence = consensus_ratio if consensus_ratio >= consensus_weight else 0.5
        
        return {
            'overall_direction': consensus_direction,
            'confidence': confidence,
            'consensus_ratio': consensus_ratio,
            'timeframe_agreement': direction_counts,
            'individual_results': timeframe_results
        }
    
    def export_trend_analysis(self, filepath: str) -> None:
        """Export trend analysis results."""
        
        analysis_data = {
            'config': self.config,
            'trend_history': self.trend_history,
            'support_resistance_levels': self.support_resistance_levels,
            'export_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        logger.info(f"Trend analysis exported to {filepath}")