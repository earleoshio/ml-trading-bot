"""
Ensemble Manager Module

Implements dynamic ensemble management for cryptocurrency trading models,
with support for adaptive weighting, model combination strategies, and
market regime-based ensemble selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from loguru import logger
from datetime import datetime
import json
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

from .model_registry import BaseModel, ModelRegistry


class EnsembleManager:
    """
    Advanced ensemble management for cryptocurrency trading models.
    
    Provides dynamic model combination with:
    - Adaptive weighting based on recent performance
    - Market regime-aware ensemble selection
    - Online learning for weight updates
    - Multiple combination strategies (average, weighted, stacked)
    - Performance monitoring and model selection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ensemble manager.
        
        Args:
            config: Configuration dictionary with ensemble parameters
        """
        self.config = config or self._default_config()
        self.models = {}
        self.weights = {}
        self.performance_history = {}
        self.ensemble_history = []
        self.meta_model = None
        
    def _default_config(self) -> Dict:
        """Default configuration for ensemble management."""
        return {
            'combination_methods': {
                'default_method': 'adaptive_weighted',
                'available_methods': ['simple_average', 'weighted_average', 
                                    'adaptive_weighted', 'stacked', 'voting']
            },
            'weight_update': {
                'update_frequency': 24,  # Hours
                'lookback_period': 168,  # Hours (1 week)
                'decay_factor': 0.95,  # Exponential decay for historical performance
                'min_weight': 0.01,  # Minimum weight for any model
                'max_weight': 0.6   # Maximum weight for any single model
            },
            'performance_metrics': {
                'regression': ['mse', 'mae', 'sharpe_ratio'],
                'classification': ['accuracy', 'precision', 'recall', 'f1']
            },
            'regime_awareness': {
                'enabled': True,
                'regime_lookback': 72,  # Hours
                'regime_weights': {
                    'bull_market': {'trend_models': 0.7, 'mean_reversion': 0.3},
                    'bear_market': {'trend_models': 0.6, 'mean_reversion': 0.4},
                    'sideways': {'trend_models': 0.3, 'mean_reversion': 0.7},
                    'high_volatility': {'robust_models': 0.8, 'sensitive_models': 0.2}
                }
            },
            'meta_learning': {
                'enabled': True,
                'meta_model_type': 'linear',  # 'linear', 'rf', 'xgboost'
                'feature_engineering': True,
                'cross_validation': True
            }
        }
    
    def add_model(self, model_name: str, model: BaseModel, 
                 initial_weight: float = None) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model_name: Name identifier for the model
            model: Model instance
            initial_weight: Initial weight (auto-calculated if None)
        """
        self.models[model_name] = model
        
        if initial_weight is None:
            # Equal weights initially
            n_models = len(self.models)
            for name in self.models.keys():
                self.weights[name] = 1.0 / n_models
        else:
            self.weights[model_name] = initial_weight
            # Renormalize weights
            total_weight = sum(self.weights.values())
            for name in self.weights.keys():
                self.weights[name] /= total_weight
        
        # Initialize performance history
        self.performance_history[model_name] = []
        
        logger.info(f"Added model '{model_name}' to ensemble with weight {self.weights[model_name]:.4f}")
    
    def remove_model(self, model_name: str) -> None:
        """Remove a model from the ensemble."""
        if model_name in self.models:
            del self.models[model_name]
            del self.weights[model_name]
            del self.performance_history[model_name]
            
            # Renormalize weights
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for name in self.weights.keys():
                    self.weights[name] /= total_weight
            
            logger.info(f"Removed model '{model_name}' from ensemble")
        else:
            logger.warning(f"Model '{model_name}' not found in ensemble")
    
    def predict(self, X: pd.DataFrame, 
               method: str = None,
               regime: str = None) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            method: Combination method to use
            regime: Current market regime (optional)
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        method = method or self.config['combination_methods']['default_method']
        
        # Get individual model predictions
        predictions = {}
        for model_name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    predictions[model_name] = pred
                except Exception as e:
                    logger.warning(f"Model '{model_name}' prediction failed: {e}")
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Combine predictions based on method
        if method == 'simple_average':
            ensemble_pred = self._simple_average(predictions)
        elif method == 'weighted_average':
            ensemble_pred = self._weighted_average(predictions)
        elif method == 'adaptive_weighted':
            ensemble_pred = self._adaptive_weighted_average(predictions, regime)
        elif method == 'stacked':
            ensemble_pred = self._stacked_prediction(X, predictions)
        elif method == 'voting':
            ensemble_pred = self._voting_prediction(predictions)
        else:
            logger.warning(f"Unknown method '{method}', using simple average")
            ensemble_pred = self._simple_average(predictions)
        
        # Record ensemble prediction
        self.ensemble_history.append({
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'regime': regime,
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_pred.tolist() if hasattr(ensemble_pred, 'tolist') else ensemble_pred
        })
        
        return ensemble_pred
    
    def _simple_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average of all predictions."""
        pred_arrays = list(predictions.values())
        return np.mean(pred_arrays, axis=0)
    
    def _weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average using current weights."""
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for model_name, pred in predictions.items():
            if model_name in self.weights:
                ensemble_pred += self.weights[model_name] * pred
        
        return ensemble_pred
    
    def _adaptive_weighted_average(self, predictions: Dict[str, np.ndarray], 
                                  regime: str = None) -> np.ndarray:
        """Adaptive weighted average considering recent performance and regime."""
        
        # Get adaptive weights
        adaptive_weights = self._calculate_adaptive_weights(predictions.keys(), regime)
        
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for model_name, pred in predictions.items():
            if model_name in adaptive_weights:
                ensemble_pred += adaptive_weights[model_name] * pred
        
        return ensemble_pred
    
    def _stacked_prediction(self, X: pd.DataFrame, 
                           predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Stacked prediction using meta-model."""
        
        if self.meta_model is None:
            logger.warning("Meta-model not trained, falling back to weighted average")
            return self._weighted_average(predictions)
        
        # Create meta-features
        meta_features = self._create_meta_features(X, predictions)
        
        # Make meta-prediction
        try:
            ensemble_pred = self.meta_model.predict(meta_features)
            return ensemble_pred
        except Exception as e:
            logger.warning(f"Meta-model prediction failed: {e}, falling back to weighted average")
            return self._weighted_average(predictions)
    
    def _voting_prediction(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Voting-based prediction (for classification)."""
        # Convert predictions to vote format (for classification)
        pred_arrays = np.array(list(predictions.values()))
        
        # For regression, use median
        if len(pred_arrays.shape) == 2:
            return np.median(pred_arrays, axis=0)
        else:
            # For classification, use majority vote
            from scipy.stats import mode
            return mode(pred_arrays, axis=0)[0]
    
    def _calculate_adaptive_weights(self, model_names: List[str], 
                                   regime: str = None) -> Dict[str, float]:
        """Calculate adaptive weights based on recent performance."""
        
        adaptive_weights = {}
        lookback_period = self.config['weight_update']['lookback_period']
        decay_factor = self.config['weight_update']['decay_factor']
        
        # Calculate performance-based weights
        performance_weights = {}
        for model_name in model_names:
            if model_name in self.performance_history:
                recent_performance = self._get_recent_performance(model_name, lookback_period)
                if recent_performance:
                    # Apply exponential decay to historical performance
                    weighted_performance = sum(
                        perf * (decay_factor ** i) 
                        for i, perf in enumerate(reversed(recent_performance))
                    )
                    performance_weights[model_name] = max(0, weighted_performance)
                else:
                    performance_weights[model_name] = self.weights.get(model_name, 0.1)
            else:
                performance_weights[model_name] = self.weights.get(model_name, 0.1)
        
        # Apply regime-specific adjustments if enabled
        if regime and self.config['regime_awareness']['enabled']:
            performance_weights = self._apply_regime_weights(performance_weights, regime)
        
        # Normalize weights
        total_weight = sum(performance_weights.values())
        if total_weight > 0:
            for model_name in model_names:
                weight = performance_weights.get(model_name, 0) / total_weight
                
                # Apply min/max constraints
                weight = max(self.config['weight_update']['min_weight'], weight)
                weight = min(self.config['weight_update']['max_weight'], weight)
                
                adaptive_weights[model_name] = weight
        else:
            # Fallback to equal weights
            for model_name in model_names:
                adaptive_weights[model_name] = 1.0 / len(model_names)
        
        # Renormalize after constraints
        total_weight = sum(adaptive_weights.values())
        for model_name in adaptive_weights:
            adaptive_weights[model_name] /= total_weight
        
        return adaptive_weights
    
    def _get_recent_performance(self, model_name: str, 
                               lookback_hours: int) -> List[float]:
        """Get recent performance scores for a model."""
        if model_name not in self.performance_history:
            return []
        
        # This is a simplified version - in practice, you'd filter by timestamp
        recent_scores = self.performance_history[model_name][-lookback_hours:]
        return [score['value'] for score in recent_scores if 'value' in score]
    
    def _apply_regime_weights(self, weights: Dict[str, float], 
                             regime: str) -> Dict[str, float]:
        """Apply regime-specific weight adjustments."""
        
        regime_config = self.config['regime_awareness']['regime_weights'].get(regime, {})
        
        if not regime_config:
            return weights
        
        # This is a simplified implementation
        # In practice, you'd have model categorization (trend_models, mean_reversion, etc.)
        adjusted_weights = weights.copy()
        
        # Apply regime-specific multipliers (simplified)
        for model_name in adjusted_weights:
            # Categorize model type based on name (simplified approach)
            if 'trend' in model_name.lower() or 'lstm' in model_name.lower():
                multiplier = regime_config.get('trend_models', 1.0)
            elif 'mean' in model_name.lower() or 'svm' in model_name.lower():
                multiplier = regime_config.get('mean_reversion', 1.0)
            elif 'robust' in model_name.lower() or 'forest' in model_name.lower():
                multiplier = regime_config.get('robust_models', 1.0)
            else:
                multiplier = 1.0
            
            adjusted_weights[model_name] *= multiplier
        
        return adjusted_weights
    
    def _create_meta_features(self, X: pd.DataFrame, 
                             predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Create meta-features for stacked ensemble."""
        
        meta_features = pd.DataFrame()
        
        # Add individual predictions as features
        for model_name, pred in predictions.items():
            meta_features[f'{model_name}_pred'] = pred
        
        if self.config['meta_learning']['feature_engineering']:
            # Add engineered features
            pred_values = list(predictions.values())
            
            # Statistical features of predictions
            meta_features['pred_mean'] = np.mean(pred_values, axis=0)
            meta_features['pred_std'] = np.std(pred_values, axis=0)
            meta_features['pred_min'] = np.min(pred_values, axis=0)
            meta_features['pred_max'] = np.max(pred_values, axis=0)
            meta_features['pred_median'] = np.median(pred_values, axis=0)
            
            # Agreement measures
            mean_pred = np.mean(pred_values, axis=0)
            meta_features['pred_agreement'] = np.mean([
                np.abs(pred - mean_pred) for pred in pred_values
            ], axis=0)
            
            # Diversity measures
            meta_features['pred_diversity'] = np.std(pred_values, axis=0)
        
        return meta_features
    
    def train_meta_model(self, X: pd.DataFrame, y: pd.Series, 
                        validation_split: float = 0.2) -> None:
        """Train the meta-model for stacked ensembling."""
        
        if not self.models:
            raise ValueError("No base models available for meta-model training")
        
        logger.info("Training meta-model for stacked ensemble...")
        
        # Split data for meta-model training
        split_idx = int(len(X) * (1 - validation_split))
        X_base, X_meta = X.iloc[:split_idx], X.iloc[split_idx:]
        y_base, y_meta = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train base models on base data
        for model_name, model in self.models.items():
            if not model.is_fitted:
                logger.info(f"Training base model '{model_name}' for meta-learning")
                model.fit(X_base, y_base)
        
        # Generate predictions on meta data
        meta_predictions = {}
        for model_name, model in self.models.items():
            if model.is_fitted:
                meta_predictions[model_name] = model.predict(X_meta)
        
        if not meta_predictions:
            raise ValueError("No base models could generate predictions for meta-training")
        
        # Create meta-features
        meta_features = self._create_meta_features(X_meta, meta_predictions)
        
        # Train meta-model
        meta_model_type = self.config['meta_learning']['meta_model_type']
        
        if meta_model_type == 'linear':
            self.meta_model = LinearRegression()
        elif meta_model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            self.meta_model = RandomForestRegressor(n_estimators=50, random_state=42)
        elif meta_model_type == 'xgboost':
            import xgboost as xgb
            self.meta_model = xgb.XGBRegressor(n_estimators=50, random_state=42)
        else:
            logger.warning(f"Unknown meta-model type '{meta_model_type}', using linear regression")
            self.meta_model = LinearRegression()
        
        self.meta_model.fit(meta_features, y_meta)
        
        logger.info("Meta-model training completed")
    
    def update_weights(self, X: pd.DataFrame, y: pd.Series, 
                      predictions: Dict[str, np.ndarray] = None) -> None:
        """Update model weights based on recent performance."""
        
        if predictions is None:
            # Generate predictions for evaluation
            predictions = {}
            for model_name, model in self.models.items():
                if model.is_fitted:
                    try:
                        predictions[model_name] = model.predict(X)
                    except Exception as e:
                        logger.warning(f"Could not generate predictions for '{model_name}': {e}")
        
        # Calculate performance for each model
        for model_name, pred in predictions.items():
            try:
                # Calculate performance metric
                if self.models[model_name].model_type == 'regression':
                    performance = -mean_squared_error(y, pred)  # Negative for maximization
                else:
                    performance = accuracy_score(y, pred)
                
                # Update performance history
                self.performance_history[model_name].append({
                    'timestamp': datetime.now().isoformat(),
                    'value': performance,
                    'metric': 'mse' if self.models[model_name].model_type == 'regression' else 'accuracy'
                })
                
            except Exception as e:
                logger.warning(f"Could not calculate performance for '{model_name}': {e}")
        
        # Recalculate weights
        self.weights = self._calculate_adaptive_weights(list(self.models.keys()))
        
        logger.info(f"Updated model weights: {self.weights}")
    
    def get_model_performance_summary(self) -> Dict:
        """Get performance summary for all models."""
        
        summary = {}
        
        for model_name, history in self.performance_history.items():
            if history:
                recent_scores = [entry['value'] for entry in history[-24:]]  # Last 24 entries
                
                summary[model_name] = {
                    'current_weight': self.weights.get(model_name, 0.0),
                    'recent_mean_performance': np.mean(recent_scores) if recent_scores else 0.0,
                    'recent_std_performance': np.std(recent_scores) if recent_scores else 0.0,
                    'total_evaluations': len(history),
                    'last_updated': history[-1]['timestamp'] if history else None
                }
        
        return summary
    
    def analyze_ensemble_diversity(self, X: pd.DataFrame) -> Dict:
        """Analyze diversity of ensemble predictions."""
        
        if not self.models:
            return {}
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            if model.is_fitted:
                try:
                    predictions[model_name] = model.predict(X)
                except:
                    continue
        
        if len(predictions) < 2:
            return {'diversity_measures': {}, 'pairwise_correlations': {}}
        
        pred_arrays = np.array(list(predictions.values()))
        model_names = list(predictions.keys())
        
        # Calculate diversity measures
        diversity_measures = {
            'mean_pairwise_correlation': 0.0,
            'prediction_variance': np.var(pred_arrays, axis=0).mean(),
            'disagreement_measure': 0.0
        }
        
        # Calculate pairwise correlations
        pairwise_correlations = {}
        total_corr = 0
        pair_count = 0
        
        for i in range(len(pred_arrays)):
            for j in range(i + 1, len(pred_arrays)):
                corr = np.corrcoef(pred_arrays[i], pred_arrays[j])[0, 1]
                if not np.isnan(corr):
                    pairwise_correlations[f"{model_names[i]}_vs_{model_names[j]}"] = corr
                    total_corr += corr
                    pair_count += 1
        
        if pair_count > 0:
            diversity_measures['mean_pairwise_correlation'] = total_corr / pair_count
        
        # Disagreement measure (average standard deviation across predictions)
        diversity_measures['disagreement_measure'] = np.mean(np.std(pred_arrays, axis=0))
        
        return {
            'diversity_measures': diversity_measures,
            'pairwise_correlations': pairwise_correlations,
            'recommendation': self._get_diversity_recommendation(diversity_measures)
        }
    
    def _get_diversity_recommendation(self, diversity_measures: Dict) -> str:
        """Provide recommendations based on ensemble diversity."""
        
        mean_corr = diversity_measures.get('mean_pairwise_correlation', 0)
        disagreement = diversity_measures.get('disagreement_measure', 0)
        
        if mean_corr > 0.8:
            return "High correlation between models - consider adding more diverse models"
        elif mean_corr < 0.3:
            return "Good diversity - ensemble should benefit from model combination"
        elif disagreement < 0.01:
            return "Low disagreement - models are very similar, diversity could be improved"
        else:
            return "Reasonable ensemble diversity"
    
    def export_ensemble_config(self, filepath: str) -> None:
        """Export ensemble configuration and weights."""
        
        config_data = {
            'models': list(self.models.keys()),
            'weights': self.weights,
            'config': self.config,
            'performance_history': self.performance_history,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        logger.info(f"Ensemble configuration exported to {filepath}")
    
    def load_ensemble_config(self, filepath: str) -> None:
        """Load ensemble configuration and weights."""
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        self.weights = config_data['weights']
        self.performance_history = config_data['performance_history']
        
        logger.info(f"Ensemble configuration loaded from {filepath}")