"""
Meta-Learner Module

Implements meta-learning algorithms for cryptocurrency trading models,
including model selection, adaptive learning, and transfer learning capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from loguru import logger
from datetime import datetime
import json
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from .model_registry import BaseModel, ModelRegistry


class MetaLearner:
    """
    Meta-learning system for adaptive model selection and optimization.
    
    Implements various meta-learning strategies including:
    - Model selection based on data characteristics
    - Learning to learn from previous trading strategies
    - Transfer learning between different market conditions
    - Adaptive algorithm selection
    - Meta-feature extraction and analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize meta-learner system.
        
        Args:
            config: Configuration dictionary with meta-learning parameters
        """
        self.config = config or self._default_config()
        self.meta_knowledge = {}
        self.model_performance_db = {}
        self.meta_features_history = []
        self.selection_history = []
        
    def _default_config(self) -> Dict:
        """Default configuration for meta-learning."""
        return {
            'meta_features': {
                'statistical': ['mean', 'std', 'skewness', 'kurtosis', 'autocorr'],
                'information_theory': ['entropy', 'mutual_info'],
                'complexity': ['effective_dimension', 'noise_ratio'],
                'temporal': ['trend_strength', 'seasonality', 'volatility_clustering']
            },
            'model_selection': {
                'selection_method': 'performance_based',  # 'performance_based', 'meta_model', 'ensemble_selection'
                'performance_window': 168,  # Hours for performance evaluation
                'min_samples_for_selection': 100,
                'exploration_rate': 0.1  # Epsilon for exploration vs exploitation
            },
            'transfer_learning': {
                'enabled': True,
                'similarity_threshold': 0.8,
                'adaptation_rate': 0.1,
                'knowledge_decay': 0.99
            },
            'adaptation': {
                'update_frequency': 24,  # Hours
                'performance_threshold': 0.05,  # Minimum improvement threshold
                'adaptation_methods': ['weight_adaptation', 'parameter_transfer', 'architecture_search']
            },
            'meta_model': {
                'algorithm': 'xgboost',  # 'xgboost', 'neural_network', 'ensemble'
                'features': 'auto',  # 'auto', 'custom', 'all'
                'cross_validation': True,
                'update_online': True
            }
        }
    
    def extract_meta_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Extract meta-features from dataset for model selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of meta-features
        """
        meta_features = {}
        
        # Statistical meta-features
        if 'statistical' in self.config['meta_features']:
            meta_features.update(self._extract_statistical_features(X, y))
        
        # Information theory meta-features
        if 'information_theory' in self.config['meta_features']:
            meta_features.update(self._extract_information_features(X, y))
        
        # Complexity meta-features
        if 'complexity' in self.config['meta_features']:
            meta_features.update(self._extract_complexity_features(X, y))
        
        # Temporal meta-features (crypto-specific)
        if 'temporal' in self.config['meta_features']:
            meta_features.update(self._extract_temporal_features(X, y))
        
        return meta_features
    
    def _extract_statistical_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Extract statistical meta-features."""
        features = {}
        
        try:
            # Target variable statistics
            features['target_mean'] = y.mean()
            features['target_std'] = y.std()
            features['target_skewness'] = y.skew()
            features['target_kurtosis'] = y.kurtosis()
            
            # Feature statistics
            features['n_features'] = len(X.columns)
            features['n_samples'] = len(X)
            features['feature_mean_std'] = X.std().mean()
            features['feature_correlation_mean'] = X.corr().abs().mean().mean()
            
            # Missing values
            features['missing_ratio'] = X.isnull().sum().sum() / (len(X) * len(X.columns))
            
            # Autocorrelation (for time series)
            if len(y) > 1:
                autocorr = y.autocorr(lag=1)
                features['target_autocorr'] = autocorr if not pd.isna(autocorr) else 0.0
                
        except Exception as e:
            logger.warning(f"Error extracting statistical features: {e}")
            
        return features
    
    def _extract_information_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Extract information theory meta-features."""
        features = {}
        
        try:
            from scipy.stats import entropy
            from sklearn.feature_selection import mutual_info_regression
            
            # Target entropy (discretized)
            y_discrete = pd.cut(y, bins=10, labels=False)
            features['target_entropy'] = entropy(pd.Series(y_discrete).value_counts())
            
            # Mutual information between features and target
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mean())
            
            common_idx = X_clean.index.intersection(y_clean.index)
            if len(common_idx) > 10:
                mi_scores = mutual_info_regression(X_clean.loc[common_idx], y_clean.loc[common_idx])
                features['mutual_info_mean'] = np.mean(mi_scores)
                features['mutual_info_max'] = np.max(mi_scores)
                features['mutual_info_std'] = np.std(mi_scores)
            
        except Exception as e:
            logger.warning(f"Error extracting information features: {e}")
            
        return features
    
    def _extract_complexity_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Extract dataset complexity meta-features."""
        features = {}
        
        try:
            # Effective dimensionality using PCA
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            X_clean = X.fillna(X.mean())
            if len(X_clean) > len(X_clean.columns):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)
                
                pca = PCA()
                pca.fit(X_scaled)
                
                # Find number of components for 95% variance
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                n_components_95 = np.argmax(cumsum >= 0.95) + 1
                features['effective_dimension'] = n_components_95 / len(X.columns)
            
            # Noise ratio estimation
            X_std = X.std()
            features['noise_ratio'] = (X_std == 0).sum() / len(X_std)
            
            # Feature redundancy
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = (upper_triangle > 0.9).sum().sum()
            total_pairs = len(X.columns) * (len(X.columns) - 1) / 2
            features['feature_redundancy'] = high_corr_pairs / total_pairs if total_pairs > 0 else 0
            
        except Exception as e:
            logger.warning(f"Error extracting complexity features: {e}")
            
        return features
    
    def _extract_temporal_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Extract temporal meta-features specific to crypto trading."""
        features = {}
        
        try:
            # Trend strength
            if len(y) > 2:
                x_values = np.arange(len(y))
                trend_slope = np.polyfit(x_values, y.values, 1)[0]
                features['trend_strength'] = abs(trend_slope)
            
            # Volatility clustering (GARCH effects)
            returns = y.pct_change().dropna()
            if len(returns) > 10:
                squared_returns = returns ** 2
                volatility_autocorr = squared_returns.autocorr(lag=1)
                features['volatility_clustering'] = volatility_autocorr if not pd.isna(volatility_autocorr) else 0
            
            # Seasonality detection (simplified)
            if len(y) > 24:  # At least 24 periods
                hourly_pattern = y.groupby(y.index % 24).mean()
                features['seasonality_strength'] = hourly_pattern.std() / hourly_pattern.mean()
            
            # Market regime stability
            if len(y) > 50:
                rolling_std = y.rolling(window=20).std()
                features['regime_stability'] = 1 / (1 + rolling_std.std())
            
        except Exception as e:
            logger.warning(f"Error extracting temporal features: {e}")
            
        return features
    
    def recommend_model(self, X: pd.DataFrame, y: pd.Series, 
                       available_models: List[str]) -> Tuple[str, float]:
        """
        Recommend the best model based on meta-learning.
        
        Args:
            X: Feature matrix
            y: Target variable
            available_models: List of available model names
            
        Returns:
            Tuple of (recommended_model, confidence_score)
        """
        # Extract meta-features
        meta_features = self.extract_meta_features(X, y)
        
        # Store meta-features for history
        self.meta_features_history.append({
            'timestamp': datetime.now().isoformat(),
            'meta_features': meta_features
        })
        
        selection_method = self.config['model_selection']['selection_method']
        
        if selection_method == 'performance_based':
            recommendation = self._performance_based_selection(meta_features, available_models)
        elif selection_method == 'meta_model':
            recommendation = self._meta_model_selection(meta_features, available_models)
        elif selection_method == 'ensemble_selection':
            recommendation = self._ensemble_selection(meta_features, available_models)
        else:
            # Default to performance-based
            recommendation = self._performance_based_selection(meta_features, available_models)
        
        # Store selection for learning
        self.selection_history.append({
            'timestamp': datetime.now().isoformat(),
            'meta_features': meta_features,
            'recommended_model': recommendation[0],
            'confidence': recommendation[1],
            'available_models': available_models
        })
        
        return recommendation
    
    def _performance_based_selection(self, meta_features: Dict[str, float], 
                                   available_models: List[str]) -> Tuple[str, float]:
        """Select model based on historical performance patterns."""
        
        if not self.model_performance_db:
            # No historical data, return random model
            return np.random.choice(available_models), 0.1
        
        # Calculate similarity to historical cases
        best_model = None
        best_similarity = 0.0
        
        for case in self.model_performance_db.values():
            similarity = self._calculate_feature_similarity(meta_features, case['meta_features'])
            
            if similarity > best_similarity and case['best_model'] in available_models:
                best_similarity = similarity
                best_model = case['best_model']
        
        if best_model is None:
            best_model = available_models[0]
            best_similarity = 0.1
        
        return best_model, best_similarity
    
    def _meta_model_selection(self, meta_features: Dict[str, float], 
                            available_models: List[str]) -> Tuple[str, float]:
        """Use meta-model to select the best model."""
        
        # This would implement a trained meta-model that predicts best model
        # For now, return a simple heuristic-based selection
        
        # Heuristic rules based on meta-features
        n_features = meta_features.get('n_features', 10)
        n_samples = meta_features.get('n_samples', 100)
        volatility_clustering = meta_features.get('volatility_clustering', 0)
        trend_strength = meta_features.get('trend_strength', 0)
        
        if 'lstm' in available_models and volatility_clustering > 0.3:
            return 'lstm', 0.8
        elif 'xgboost' in available_models and n_features > 20:
            return 'xgboost', 0.7
        elif 'svm' in available_models and n_samples < 500:
            return 'svm', 0.6
        elif 'randomforest' in available_models:
            return 'randomforest', 0.5
        else:
            return available_models[0], 0.3
    
    def _ensemble_selection(self, meta_features: Dict[str, float], 
                          available_models: List[str]) -> Tuple[str, float]:
        """Select multiple models for ensembling."""
        
        # For ensemble selection, return the most diverse set
        # This is simplified - would implement proper diversity-based selection
        
        if len(available_models) >= 3:
            # Select top 3 diverse models
            selected_models = available_models[:3]
            return f"ensemble_{'+'.join(selected_models)}", 0.9
        else:
            return available_models[0], 0.5
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], 
                                    features2: Dict[str, float]) -> float:
        """Calculate similarity between two meta-feature vectors."""
        
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        # Normalize features and calculate cosine similarity
        vec1 = np.array([features1[key] for key in common_keys])
        vec2 = np.array([features2[key] for key in common_keys])
        
        # Handle zero vectors
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return max(0, similarity)  # Ensure non-negative
    
    def update_knowledge(self, meta_features: Dict[str, float], 
                        model_performances: Dict[str, float]) -> None:
        """
        Update meta-learning knowledge base with new experience.
        
        Args:
            meta_features: Meta-features of the dataset
            model_performances: Performance scores for each model
        """
        
        if not model_performances:
            return
        
        # Find best performing model
        best_model = max(model_performances.items(), key=lambda x: x[1])
        
        # Create knowledge entry
        knowledge_entry = {
            'timestamp': datetime.now().isoformat(),
            'meta_features': meta_features,
            'model_performances': model_performances,
            'best_model': best_model[0],
            'best_performance': best_model[1]
        }
        
        # Store in knowledge base
        entry_id = f"entry_{len(self.model_performance_db)}"
        self.model_performance_db[entry_id] = knowledge_entry
        
        # Apply knowledge decay to older entries
        decay_factor = self.config['transfer_learning']['knowledge_decay']
        for entry in self.model_performance_db.values():
            if 'weight' in entry:
                entry['weight'] *= decay_factor
            else:
                entry['weight'] = 1.0
        
        logger.info(f"Updated meta-learning knowledge base. Total entries: {len(self.model_performance_db)}")
    
    def transfer_learning_adapt(self, source_meta_features: Dict[str, float], 
                               target_meta_features: Dict[str, float],
                               source_model_params: Dict) -> Dict:
        """
        Adapt model parameters using transfer learning.
        
        Args:
            source_meta_features: Meta-features from source domain
            target_meta_features: Meta-features from target domain
            source_model_params: Parameters from source model
            
        Returns:
            Adapted parameters for target domain
        """
        
        if not self.config['transfer_learning']['enabled']:
            return source_model_params
        
        # Calculate domain similarity
        similarity = self._calculate_feature_similarity(source_meta_features, target_meta_features)
        
        if similarity < self.config['transfer_learning']['similarity_threshold']:
            logger.info("Domains too dissimilar for transfer learning")
            return source_model_params
        
        # Adapt parameters based on similarity and adaptation rate
        adaptation_rate = self.config['transfer_learning']['adaptation_rate']
        adapted_params = source_model_params.copy()
        
        # Simple parameter adaptation (this would be more sophisticated in practice)
        for param_name, param_value in adapted_params.items():
            if isinstance(param_value, (int, float)):
                # Adjust parameter based on domain difference
                feature_diff = abs(
                    target_meta_features.get('target_std', 1) - 
                    source_meta_features.get('target_std', 1)
                )
                adjustment = 1 + (feature_diff * adaptation_rate)
                adapted_params[param_name] = param_value * adjustment
        
        logger.info(f"Applied transfer learning with similarity {similarity:.3f}")
        return adapted_params
    
    def analyze_meta_learning_effectiveness(self) -> Dict:
        """Analyze the effectiveness of meta-learning decisions."""
        
        if len(self.selection_history) < 5:
            return {'message': 'Insufficient data for analysis'}
        
        analysis = {
            'total_selections': len(self.selection_history),
            'model_selection_frequency': {},
            'average_confidence': 0.0,
            'improvement_over_random': 0.0
        }
        
        # Model selection frequency
        for selection in self.selection_history:
            model = selection['recommended_model']
            analysis['model_selection_frequency'][model] = analysis['model_selection_frequency'].get(model, 0) + 1
        
        # Average confidence
        confidences = [s['confidence'] for s in self.selection_history]
        analysis['average_confidence'] = np.mean(confidences)
        
        # Feature importance analysis
        if len(self.meta_features_history) > 1:
            all_features = set()
            for features in self.meta_features_history:
                all_features.update(features['meta_features'].keys())
            
            analysis['meta_features_used'] = list(all_features)
            analysis['feature_stability'] = len(all_features) / len(self.meta_features_history)
        
        return analysis
    
    def get_learning_recommendations(self) -> List[str]:
        """Get recommendations for improving meta-learning performance."""
        
        recommendations = []
        
        analysis = self.analyze_meta_learning_effectiveness()
        
        # Check confidence levels
        if analysis.get('average_confidence', 0) < 0.5:
            recommendations.append("Low confidence in model selections - consider more training data")
        
        # Check model diversity
        model_freq = analysis.get('model_selection_frequency', {})
        if len(model_freq) < 2:
            recommendations.append("Limited model diversity - consider adding more model types")
        
        # Check knowledge base size
        if len(self.model_performance_db) < 10:
            recommendations.append("Small knowledge base - more experience needed for better recommendations")
        
        # Check feature stability
        if analysis.get('feature_stability', 1.0) < 0.8:
            recommendations.append("Feature instability detected - consider feature engineering review")
        
        return recommendations
    
    def export_meta_knowledge(self, filepath: str) -> None:
        """Export meta-learning knowledge base."""
        
        knowledge_export = {
            'model_performance_db': self.model_performance_db,
            'meta_features_history': self.meta_features_history,
            'selection_history': self.selection_history,
            'config': self.config,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(knowledge_export, f, indent=2, default=str)
        
        logger.info(f"Meta-learning knowledge exported to {filepath}")
    
    def load_meta_knowledge(self, filepath: str) -> None:
        """Load meta-learning knowledge base."""
        
        with open(filepath, 'r') as f:
            knowledge_data = json.load(f)
        
        self.model_performance_db = knowledge_data.get('model_performance_db', {})
        self.meta_features_history = knowledge_data.get('meta_features_history', [])
        self.selection_history = knowledge_data.get('selection_history', [])
        
        logger.info(f"Meta-learning knowledge loaded from {filepath}")
        logger.info(f"Loaded {len(self.model_performance_db)} performance records")
    
    def reset_knowledge_base(self) -> None:
        """Reset the meta-learning knowledge base."""
        
        self.model_performance_db = {}
        self.meta_features_history = []
        self.selection_history = []
        
        logger.info("Meta-learning knowledge base reset")