"""
Market Microstructure Features Module

Implements market microstructure analysis for cryptocurrency trading,
including order book analysis, trade size distribution, and liquidity metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger


class MarketMicrostructure:
    """
    Market microstructure features for cryptocurrency trading.
    
    Analyzes order book dynamics, trade patterns, and liquidity conditions
    to generate features that capture market microstructure effects.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize market microstructure analyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default configuration for market microstructure analysis."""
        return {
            'order_book': {
                'depth_levels': 10,
                'imbalance_periods': [5, 10, 20],
                'pressure_periods': [5, 15, 30]
            },
            'trade_analysis': {
                'size_quantiles': [0.5, 0.75, 0.9, 0.95, 0.99],
                'aggressor_threshold': 0.6,
                'block_size_threshold': 100000  # USD
            },
            'liquidity': {
                'spread_periods': [1, 5, 15],
                'depth_threshold': [1000, 5000, 10000],  # USD
                'impact_sizes': [1000, 10000, 50000]  # USD
            },
            'volatility': {
                'realized_vol_periods': [5, 15, 30, 60],
                'microstructure_noise_threshold': 0.001
            }
        }
    
    def generate_features(self, 
                         ohlcv_data: pd.DataFrame,
                         order_book_data: Optional[pd.DataFrame] = None,
                         trades_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate market microstructure features.
        
        Args:
            ohlcv_data: OHLCV price data
            order_book_data: Order book snapshots (optional)
            trades_data: Individual trades data (optional)
            
        Returns:
            DataFrame with microstructure features
        """
        logger.info("Generating market microstructure features...")
        
        result_df = ohlcv_data.copy()
        
        # Add basic microstructure features from OHLCV
        result_df = self._add_basic_microstructure_features(result_df)
        
        # Add order book features if available
        if order_book_data is not None:
            result_df = self._add_order_book_features(result_df, order_book_data)
        
        # Add trade-level features if available
        if trades_data is not None:
            result_df = self._add_trade_features(result_df, trades_data)
        
        # Add liquidity measures
        result_df = self._add_liquidity_features(result_df)
        
        # Add volatility microstructure features
        result_df = self._add_volatility_microstructure_features(result_df)
        
        logger.info(f"Generated {len(result_df.columns) - len(ohlcv_data.columns)} microstructure features")
        return result_df
    
    def _add_basic_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic microstructure features from OHLCV data."""
        result = data.copy()
        
        # Price impact proxy (high-low spread)
        result['hl_spread'] = (data['high'] - data['low']) / data['close']
        result['hl_spread_ma'] = result['hl_spread'].rolling(20).mean()
        
        # Intrabar price efficiency
        result['price_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
        
        # Volume-price relationships
        result['volume_price_correlation'] = data['volume'].rolling(20).corr(data['close'])
        result['volume_volatility'] = data['volume'].rolling(20).std() / data['volume'].rolling(20).mean()
        
        # Tick direction inference (simplified)
        result['tick_direction'] = np.sign(data['close'].diff())
        result['tick_persistence'] = result['tick_direction'].rolling(10).sum() / 10
        
        # Realized volatility (high-frequency proxy)
        log_returns = np.log(data['close'] / data['close'].shift(1))
        for period in self.config['volatility']['realized_vol_periods']:
            result[f'realized_vol_{period}'] = log_returns.rolling(period).std() * np.sqrt(period)
        
        return result
    
    def _add_order_book_features(self, price_data: pd.DataFrame, 
                                order_book_data: pd.DataFrame) -> pd.DataFrame:
        """Add order book imbalance and pressure features."""
        result = price_data.copy()
        
        # Placeholder implementation - would need actual order book data structure
        # Order book imbalance (bid volume - ask volume) / (bid volume + ask volume)
        result['order_book_imbalance'] = 0.0  # Placeholder
        
        # Bid-ask spread
        result['bid_ask_spread'] = 0.0  # Placeholder
        result['relative_spread'] = 0.0  # Placeholder
        
        # Order book depth
        for threshold in self.config['liquidity']['depth_threshold']:
            result[f'depth_{threshold}'] = 0.0  # Placeholder
        
        # Order book pressure indicators
        for period in self.config['order_book']['pressure_periods']:
            result[f'buy_pressure_{period}'] = 0.0  # Placeholder
            result[f'sell_pressure_{period}'] = 0.0  # Placeholder
        
        # Order book slope (price impact per unit volume)
        result['ob_slope_bid'] = 0.0  # Placeholder
        result['ob_slope_ask'] = 0.0  # Placeholder
        
        return result
    
    def _add_trade_features(self, price_data: pd.DataFrame, 
                           trades_data: pd.DataFrame) -> pd.DataFrame:
        """Add trade-level microstructure features."""
        result = price_data.copy()
        
        # Placeholder implementation - would need actual trade data aggregation
        # Trade size distribution
        for quantile in self.config['trade_analysis']['size_quantiles']:
            result[f'trade_size_q{int(quantile*100)}'] = 0.0  # Placeholder
        
        # Aggressor side ratio (buy vs sell initiated trades)
        result['buy_ratio'] = 0.5  # Placeholder
        result['sell_ratio'] = 0.5  # Placeholder
        
        # Large trade (block trade) indicators
        result['block_trades_count'] = 0.0  # Placeholder
        result['block_trades_volume'] = 0.0  # Placeholder
        
        # Trade intensity (number of trades per unit time)
        result['trade_intensity'] = 0.0  # Placeholder
        result['trade_intensity_ma'] = 0.0  # Placeholder
        
        # Average trade size
        result['avg_trade_size'] = 0.0  # Placeholder
        result['trade_size_volatility'] = 0.0  # Placeholder
        
        return result
    
    def _add_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity-related features."""
        result = data.copy()
        
        # Amihud illiquidity measure: |return| / volume
        returns = np.abs(np.log(data['close'] / data['close'].shift(1)))
        result['amihud_illiquidity'] = returns / (data['volume'] + 1e-8)
        
        # Rolling liquidity measures
        for period in self.config['liquidity']['spread_periods']:
            result[f'liquidity_score_{period}'] = (
                1 / (result['amihud_illiquidity'].rolling(period).mean() + 1e-8)
            )
        
        # Volume-weighted average spread proxy
        result['vwas_proxy'] = result['hl_spread'] * data['volume'] / data['volume'].rolling(20).sum()
        
        # Price impact estimation
        volume_ma = data['volume'].rolling(20).mean()
        result['price_impact_proxy'] = result['hl_spread'] / (volume_ma + 1e-8)
        
        # Market depth proxy (inverse of price impact)
        result['market_depth_proxy'] = 1 / (result['price_impact_proxy'] + 1e-8)
        
        return result
    
    def _add_volatility_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility microstructure features."""
        result = data.copy()
        
        # Microstructure noise estimation
        returns = np.log(data['close'] / data['close'].shift(1))
        result['microstructure_noise'] = np.abs(returns - returns.rolling(3).mean())
        
        # Signal-to-noise ratio
        result['signal_noise_ratio'] = (
            returns.rolling(20).std() / 
            (result['microstructure_noise'].rolling(20).mean() + 1e-8)
        )
        
        # Volatility clustering detection
        squared_returns = returns ** 2
        result['volatility_clustering'] = squared_returns.rolling(10).corr(
            squared_returns.shift(1).rolling(10)
        )
        
        # Jump detection (simplified)
        vol_threshold = returns.rolling(20).std() * 3
        result['jump_indicator'] = (np.abs(returns) > vol_threshold).astype(int)
        result['jump_intensity'] = result['jump_indicator'].rolling(20).sum()
        
        # Intraday volatility patterns (hour-of-day effects)
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            hourly_vol = returns.groupby(data['hour']).std()
            result['hourly_vol_effect'] = data['hour'].map(hourly_vol)
        
        return result
    
    def calculate_market_impact(self, 
                              price_data: pd.DataFrame,
                              order_sizes: List[float]) -> Dict[float, pd.Series]:
        """
        Calculate market impact for different order sizes.
        
        Args:
            price_data: OHLCV price data
            order_sizes: List of order sizes in USD
            
        Returns:
            Dictionary mapping order sizes to estimated market impact
        """
        results = {}
        
        # Use Amihud measure as proxy for market impact
        returns = np.abs(np.log(price_data['close'] / price_data['close'].shift(1)))
        base_impact = returns / (price_data['volume'] + 1e-8)
        
        for size in order_sizes:
            # Scale impact by order size (square root law approximation)
            volume_ratio = size / (price_data['volume'] * price_data['close'])
            results[size] = base_impact * np.sqrt(volume_ratio)
        
        return results
    
    def detect_liquidity_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect different liquidity regimes based on microstructure indicators.
        
        Args:
            data: DataFrame with microstructure features
            
        Returns:
            Series with regime labels (high/medium/low liquidity)
        """
        # Use quantile-based regime classification
        liquidity_score = 1 / (data['amihud_illiquidity'] + 1e-8)
        
        high_threshold = liquidity_score.quantile(0.67)
        low_threshold = liquidity_score.quantile(0.33)
        
        regimes = pd.Series(index=data.index, dtype=str)
        regimes[liquidity_score >= high_threshold] = 'high_liquidity'
        regimes[(liquidity_score < high_threshold) & (liquidity_score >= low_threshold)] = 'medium_liquidity'
        regimes[liquidity_score < low_threshold] = 'low_liquidity'
        
        return regimes
    
    def analyze_microstructure_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Analyze microstructure patterns and relationships.
        
        Args:
            data: DataFrame with microstructure features
            
        Returns:
            Dictionary with pattern analysis results
        """
        analysis = {}
        
        # Volume-price correlation analysis
        analysis['volume_price_corr'] = data['volume'].corr(data['close'])
        
        # Spread-volatility relationship
        if 'bid_ask_spread' in data.columns and 'realized_vol_5' in data.columns:
            analysis['spread_vol_corr'] = data['bid_ask_spread'].corr(data['realized_vol_5'])
        
        # Order imbalance effectiveness
        if 'order_book_imbalance' in data.columns:
            returns = np.log(data['close'] / data['close'].shift(1)).shift(-1)  # Next period return
            analysis['imbalance_return_corr'] = data['order_book_imbalance'].corr(returns)
        
        # Liquidity timing patterns
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            analysis['hourly_liquidity'] = data.groupby('hour')['amihud_illiquidity'].mean().to_dict()
        
        return analysis