"""
Volatility Regimes Module

Implements volatility regime detection and clustering for cryptocurrency markets,
including GARCH-based modeling, volatility clustering detection, and regime switching.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch library not available. Some volatility modeling features will be limited.")

from sklearn.cluster import KMeans, GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks


class VolatilityRegimeDetector:
    """
    Volatility regime detection for cryptocurrency markets.
    
    Detects different volatility regimes using various methods including:
    - GARCH-based regime detection
    - Volatility clustering analysis
    - Hidden Markov Models for regime switching
    - Threshold-based regime identification
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize volatility regime detector.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or self._default_config()
        self.regimes = {}
        self.volatility_models = {}
        self.regime_history = []
        
    def _default_config(self) -> Dict:
        """Default configuration for volatility regime detection."""
        return {
            'detection_methods': {
                'threshold_based': True,
                'garch_based': ARCH_AVAILABLE,
                'clustering_based': True,
                'hmm_based': False  # Requires additional dependencies
            },
            'volatility_estimation': {
                'method': 'realized_volatility',  # 'realized_volatility', 'garch', 'ewma'
                'window': 24,  # Hours for volatility calculation
                'annualization_factor': 365 * 24  # For hourly data
            },
            'threshold_regimes': {
                'low_vol_percentile': 25,
                'high_vol_percentile': 75,
                'extreme_vol_percentile': 95
            },
            'clustering': {
                'n_clusters': 3,  # Low, Medium, High volatility
                'cluster_method': 'kmeans',  # 'kmeans', 'gmm'
                'features': ['volatility', 'volatility_ma', 'volatility_std']
            },
            'garch_model': {
                'model_type': 'GARCH',  # 'GARCH', 'EGARCH', 'GJR-GARCH'
                'p': 1,  # GARCH lag
                'q': 1,  # ARCH lag
                'update_frequency': 24  # Hours
            },
            'regime_persistence': {
                'min_duration': 4,  # Minimum hours in regime
                'smoothing': True,
                'persistence_threshold': 0.7
            }
        }
    
    def detect_regimes(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Detect volatility regimes in price data.
        
        Args:
            price_data: DataFrame with OHLCV price data
            
        Returns:
            Series with regime labels for each time period
        """
        logger.info("Detecting volatility regimes...")
        
        # Calculate volatility
        volatility = self._calculate_volatility(price_data)
        
        # Apply different detection methods
        regimes_dict = {}
        
        if self.config['detection_methods']['threshold_based']:
            regimes_dict['threshold'] = self._threshold_based_detection(volatility)
        
        if self.config['detection_methods']['garch_based'] and ARCH_AVAILABLE:
            regimes_dict['garch'] = self._garch_based_detection(price_data, volatility)
        
        if self.config['detection_methods']['clustering_based']:
            regimes_dict['clustering'] = self._clustering_based_detection(volatility)
        
        # Combine regime predictions (ensemble approach)
        final_regimes = self._combine_regime_predictions(regimes_dict, volatility.index)
        
        # Apply persistence filtering
        if self.config['regime_persistence']['smoothing']:
            final_regimes = self._smooth_regimes(final_regimes)
        
        # Store regime information
        self.regimes['current'] = final_regimes
        self.regime_history.append({
            'timestamp': datetime.now().isoformat(),
            'regimes': final_regimes,
            'volatility': volatility,
            'methods_used': list(regimes_dict.keys())
        })
        
        logger.info(f"Detected {len(final_regimes.unique())} volatility regimes")
        return final_regimes
    
    def _calculate_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate volatility using specified method."""
        
        method = self.config['volatility_estimation']['method']
        window = self.config['volatility_estimation']['window']
        annualization = self.config['volatility_estimation']['annualization_factor']
        
        if method == 'realized_volatility':
            # Using close-to-close returns
            returns = np.log(price_data['close'] / price_data['close'].shift(1))
            volatility = returns.rolling(window=window).std() * np.sqrt(annualization)
            
        elif method == 'garch' and ARCH_AVAILABLE:
            # Using GARCH model
            returns = np.log(price_data['close'] / price_data['close'].shift(1)).dropna() * 100
            
            try:
                model = arch_model(returns, vol='GARCH', p=1, q=1)
                fitted_model = model.fit(disp='off')
                volatility = fitted_model.conditional_volatility / 100
                volatility.index = price_data.index[1:len(volatility)+1]
            except:
                logger.warning("GARCH volatility estimation failed, falling back to realized volatility")
                returns = np.log(price_data['close'] / price_data['close'].shift(1))
                volatility = returns.rolling(window=window).std() * np.sqrt(annualization)
                
        elif method == 'ewma':
            # Exponentially weighted moving average
            returns = np.log(price_data['close'] / price_data['close'].shift(1))
            volatility = returns.ewm(span=window).std() * np.sqrt(annualization)
            
        else:
            # Default to realized volatility
            returns = np.log(price_data['close'] / price_data['close'].shift(1))
            volatility = returns.rolling(window=window).std() * np.sqrt(annualization)
        
        return volatility.fillna(method='bfill')
    
    def _threshold_based_detection(self, volatility: pd.Series) -> pd.Series:
        """Detect regimes using volatility thresholds."""
        
        # Calculate percentile thresholds
        low_threshold = volatility.quantile(self.config['threshold_regimes']['low_vol_percentile'] / 100)
        high_threshold = volatility.quantile(self.config['threshold_regimes']['high_vol_percentile'] / 100)
        extreme_threshold = volatility.quantile(self.config['threshold_regimes']['extreme_vol_percentile'] / 100)
        
        # Classify regimes
        regimes = pd.Series('medium_vol', index=volatility.index)
        regimes[volatility <= low_threshold] = 'low_vol'
        regimes[volatility >= high_threshold] = 'high_vol'
        regimes[volatility >= extreme_threshold] = 'extreme_vol'
        
        return regimes
    
    def _garch_based_detection(self, price_data: pd.DataFrame, volatility: pd.Series) -> pd.Series:
        """Detect regimes using GARCH model residuals and conditional volatility."""
        
        if not ARCH_AVAILABLE:
            logger.warning("ARCH package not available for GARCH-based detection")
            return pd.Series('medium_vol', index=volatility.index)
        
        returns = np.log(price_data['close'] / price_data['close'].shift(1)).dropna() * 100
        
        try:
            # Fit GARCH model
            model_type = self.config['garch_model']['model_type']
            p = self.config['garch_model']['p']
            q = self.config['garch_model']['q']
            
            if model_type == 'EGARCH':
                model = arch_model(returns, vol='EGARCH', p=p, o=1, q=q)
            elif model_type == 'GJR-GARCH':
                model = arch_model(returns, vol='GARCH', p=p, o=1, q=q)
            else:
                model = arch_model(returns, vol='GARCH', p=p, q=q)
            
            fitted_model = model.fit(disp='off')
            
            # Extract conditional volatility and standardized residuals
            conditional_vol = fitted_model.conditional_volatility
            standardized_residuals = fitted_model.resid / conditional_vol
            
            # Detect regime changes using structural breaks in conditional volatility
            vol_changes = np.abs(np.diff(conditional_vol))
            change_threshold = vol_changes.quantile(0.95)
            
            # Create regime series
            regimes = pd.Series('medium_vol', index=price_data.index[1:len(conditional_vol)+1])
            
            # High volatility periods
            high_vol_mask = conditional_vol > conditional_vol.quantile(0.75)
            regimes[high_vol_mask] = 'high_vol'
            
            # Low volatility periods
            low_vol_mask = conditional_vol < conditional_vol.quantile(0.25)
            regimes[low_vol_mask] = 'low_vol'
            
            # Extend to full index
            full_regimes = pd.Series('medium_vol', index=volatility.index)
            full_regimes.loc[regimes.index] = regimes
            
            return full_regimes
            
        except Exception as e:
            logger.warning(f"GARCH-based detection failed: {e}")
            return pd.Series('medium_vol', index=volatility.index)
    
    def _clustering_based_detection(self, volatility: pd.Series) -> pd.Series:
        """Detect regimes using clustering on volatility features."""
        
        # Prepare features
        features_df = pd.DataFrame()
        features_df['volatility'] = volatility
        features_df['volatility_ma'] = volatility.rolling(
            window=self.config['volatility_estimation']['window']
        ).mean()
        features_df['volatility_std'] = volatility.rolling(
            window=self.config['volatility_estimation']['window'] * 2
        ).std()
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        if len(features_df) < self.config['clustering']['n_clusters']:
            logger.warning("Insufficient data for clustering-based regime detection")
            return pd.Series('medium_vol', index=volatility.index)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # Apply clustering
        method = self.config['clustering']['cluster_method']
        n_clusters = self.config['clustering']['n_clusters']
        
        if method == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        
        cluster_labels = clusterer.fit_predict(features_scaled)
        
        # Map cluster labels to regime names
        regime_mapping = self._map_clusters_to_regimes(
            cluster_labels, features_df['volatility']
        )
        
        # Create regime series
        regimes = pd.Series(index=features_df.index)
        for i, label in enumerate(cluster_labels):
            regimes.iloc[i] = regime_mapping[label]
        
        # Extend to full index
        full_regimes = pd.Series('medium_vol', index=volatility.index)
        full_regimes.loc[regimes.index] = regimes
        
        return full_regimes
    
    def _map_clusters_to_regimes(self, cluster_labels: np.ndarray, 
                                volatility_values: pd.Series) -> Dict[int, str]:
        """Map cluster labels to meaningful regime names."""
        
        # Calculate average volatility for each cluster
        cluster_vol_means = {}
        for label in np.unique(cluster_labels):
            mask = cluster_labels == label
            cluster_vol_means[label] = volatility_values.iloc[mask].mean()
        
        # Sort clusters by average volatility
        sorted_clusters = sorted(cluster_vol_means.items(), key=lambda x: x[1])
        
        # Map to regime names
        regime_mapping = {}
        n_clusters = len(sorted_clusters)
        
        if n_clusters == 2:
            regime_mapping[sorted_clusters[0][0]] = 'low_vol'
            regime_mapping[sorted_clusters[1][0]] = 'high_vol'
        elif n_clusters == 3:
            regime_mapping[sorted_clusters[0][0]] = 'low_vol'
            regime_mapping[sorted_clusters[1][0]] = 'medium_vol'
            regime_mapping[sorted_clusters[2][0]] = 'high_vol'
        elif n_clusters == 4:
            regime_mapping[sorted_clusters[0][0]] = 'low_vol'
            regime_mapping[sorted_clusters[1][0]] = 'medium_vol'
            regime_mapping[sorted_clusters[2][0]] = 'high_vol'
            regime_mapping[sorted_clusters[3][0]] = 'extreme_vol'
        else:
            # Default mapping for other cluster counts
            for i, (label, _) in enumerate(sorted_clusters):
                regime_mapping[label] = f'regime_{i}'
        
        return regime_mapping
    
    def _combine_regime_predictions(self, regimes_dict: Dict[str, pd.Series], 
                                   index: pd.Index) -> pd.Series:
        """Combine predictions from different detection methods."""
        
        if not regimes_dict:
            return pd.Series('medium_vol', index=index)
        
        # Simple voting approach
        combined_regimes = pd.Series('medium_vol', index=index)
        
        for i in index:
            votes = {}
            for method, regimes in regimes_dict.items():
                if i in regimes.index:
                    regime = regimes.loc[i]
                    votes[regime] = votes.get(regime, 0) + 1
            
            if votes:
                # Select regime with most votes
                combined_regimes.loc[i] = max(votes.items(), key=lambda x: x[1])[0]
        
        return combined_regimes
    
    def _smooth_regimes(self, regimes: pd.Series) -> pd.Series:
        """Apply smoothing to reduce regime switching noise."""
        
        min_duration = self.config['regime_persistence']['min_duration']
        smoothed_regimes = regimes.copy()
        
        # Find regime changes
        regime_changes = regimes != regimes.shift(1)
        change_indices = regime_changes[regime_changes].index
        
        # Check regime duration and smooth short regimes
        for i, change_idx in enumerate(change_indices[1:], 1):
            prev_change_idx = change_indices[i-1]
            duration = change_idx - prev_change_idx
            
            if duration < pd.Timedelta(hours=min_duration):
                # Replace short regime with previous regime
                prev_regime = regimes.loc[prev_change_idx - pd.Timedelta(hours=1)]
                smoothed_regimes.loc[prev_change_idx:change_idx] = prev_regime
        
        return smoothed_regimes
    
    def detect_volatility_breakouts(self, volatility: pd.Series, 
                                   threshold_factor: float = 2.0) -> pd.Series:
        """
        Detect volatility breakouts and spikes.
        
        Args:
            volatility: Volatility time series
            threshold_factor: Multiplier for volatility threshold
            
        Returns:
            Series indicating breakout periods
        """
        
        # Calculate rolling statistics
        vol_ma = volatility.rolling(window=24).mean()
        vol_std = volatility.rolling(window=24).std()
        
        # Define breakout threshold
        breakout_threshold = vol_ma + (threshold_factor * vol_std)
        
        # Detect breakouts
        breakouts = volatility > breakout_threshold
        
        return breakouts.astype(int)
    
    def analyze_volatility_clustering(self, returns: pd.Series) -> Dict[str, float]:
        """
        Analyze volatility clustering characteristics.
        
        Args:
            returns: Returns time series
            
        Returns:
            Dictionary with clustering metrics
        """
        
        squared_returns = returns ** 2
        
        # Autocorrelation of squared returns (ARCH effect)
        arch_lags = [1, 5, 10, 20]
        arch_effects = {}
        
        for lag in arch_lags:
            if len(squared_returns) > lag:
                arch_effects[f'arch_{lag}'] = squared_returns.autocorr(lag=lag)
        
        # Ljung-Box test for ARCH effects
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(squared_returns.dropna(), lags=10, return_df=True)
            ljung_box_pvalue = lb_test['lb_pvalue'].min()
        except:
            ljung_box_pvalue = np.nan
        
        # Volatility persistence
        volatility = returns.rolling(window=24).std()
        vol_persistence = volatility.autocorr(lag=1) if len(volatility) > 1 else np.nan
        
        clustering_metrics = {
            'arch_effects': arch_effects,
            'ljung_box_pvalue': ljung_box_pvalue,
            'volatility_persistence': vol_persistence,
            'has_clustering': ljung_box_pvalue < 0.05 if not np.isnan(ljung_box_pvalue) else False
        }
        
        return clustering_metrics
    
    def get_regime_statistics(self, regimes: pd.Series, 
                             price_data: pd.DataFrame) -> Dict:
        """
        Calculate statistics for each volatility regime.
        
        Args:
            regimes: Regime classification series
            price_data: Price data for statistics calculation
            
        Returns:
            Dictionary with regime-specific statistics
        """
        
        returns = np.log(price_data['close'] / price_data['close'].shift(1))
        
        regime_stats = {}
        
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
                
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask]
            regime_prices = price_data.loc[regime_mask, 'close']
            
            if len(regime_returns) > 1:
                regime_stats[regime] = {
                    'count': len(regime_returns),
                    'duration_hours': len(regime_returns),
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std() * np.sqrt(365 * 24),  # Annualized
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() * np.sqrt(365 * 24) if regime_returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(regime_prices),
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurtosis()
                }
        
        return regime_stats
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown for a price series."""
        
        if len(prices) == 0:
            return 0.0
        
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Return maximum drawdown (most negative value)
        return drawdown.min()
    
    def export_regime_analysis(self, filepath: str) -> None:
        """Export regime analysis results."""
        
        analysis_data = {
            'config': self.config,
            'current_regimes': self.regimes,
            'regime_history': self.regime_history,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Convert pandas objects to serializable format
        import json
        
        def convert_for_json(obj):
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            else:
                return str(obj)
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=convert_for_json)
        
        logger.info(f"Regime analysis exported to {filepath}")
    
    def plot_volatility_regimes(self, price_data: pd.DataFrame, regimes: pd.Series):
        """Plot volatility regimes with price data (requires matplotlib)."""
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # Plot price data
            ax1.plot(price_data.index, price_data['close'], label='Price', color='black')
            ax1.set_ylabel('Price')
            ax1.set_title('Price and Volatility Regimes')
            ax1.legend()
            
            # Plot volatility with regime coloring
            volatility = self._calculate_volatility(price_data)
            
            # Color mapping for regimes
            regime_colors = {
                'low_vol': 'green',
                'medium_vol': 'blue',
                'high_vol': 'orange',
                'extreme_vol': 'red'
            }
            
            for regime in regimes.unique():
                if pd.isna(regime):
                    continue
                mask = regimes == regime
                ax2.scatter(volatility.index[mask], volatility[mask], 
                          c=regime_colors.get(regime, 'gray'), 
                          label=regime, alpha=0.6)
            
            ax2.set_ylabel('Volatility')
            ax2.set_xlabel('Time')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available. Cannot create regime plots.")
        except Exception as e:
            logger.warning(f"Error creating regime plot: {e}")