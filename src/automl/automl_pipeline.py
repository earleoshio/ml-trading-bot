"""
AutoML Pipeline Module

Main orchestration module for the cryptocurrency AutoML trading system.
Coordinates all components including feature engineering, model selection,
regime detection, strategy generation, and monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from loguru import logger
from datetime import datetime, timedelta
import yaml
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from .feature_engineering import (
    TechnicalIndicators, MarketMicrostructure, SentimentFeatures,
    PatternRecognition, FeatureSelector
)
from .model_selection import (
    ModelRegistry, HyperparameterOptimizer, EnsembleManager, MetaLearner
)
from .regime_detection import (
    VolatilityRegimeDetector, TrendDetector
)


class AutoMLPipeline:
    """
    Main AutoML pipeline for cryptocurrency trading.
    
    Orchestrates the entire machine learning pipeline including:
    - Data preprocessing and feature engineering
    - Model selection and hyperparameter optimization
    - Regime detection and adaptive modeling
    - Strategy generation and risk management
    - Performance monitoring and model retraining
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AutoML pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Initialize components
        self.technical_indicators = None
        self.market_microstructure = None
        self.sentiment_features = None
        self.pattern_recognition = None
        self.feature_selector = None
        
        self.model_registry = None
        self.hyperparameter_optimizer = None
        self.ensemble_manager = None
        self.meta_learner = None
        
        self.volatility_detector = None
        self.trend_detector = None
        
        # State tracking
        self.pipeline_state = {
            'initialized': False,
            'trained': False,
            'last_update': None,
            'performance_history': [],
            'current_models': {},
            'active_regimes': {}
        }
        
        self.feature_cache = {}
        self.prediction_cache = {}
        
        logger.info("AutoML Pipeline initialized")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file."""
        
        if config_path is None:
            # Use default config path
            config_path = Path(__file__).parent.parent.parent / 'config' / 'automl_config.yaml'
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using default configuration")
            return self._default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using default configuration")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration if config file is not available."""
        return {
            'feature_engineering': {'enabled': True},
            'model_selection': {'enabled': True},
            'regime_detection': {'enabled': True},
            'monitoring': {'enabled': True},
            'general': {
                'random_seed': 42,
                'n_jobs': -1
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        
        log_config = self.config.get('logging', {})
        
        # Configure loguru
        logger.remove()  # Remove default handler
        
        log_format = log_config.get(
            'format', 
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
        )
        
        if log_config.get('destinations', {}).get('console', True):
            logger.add(
                sink=lambda msg: print(msg, end=''),
                format=log_format,
                level=log_config.get('level', 'INFO')
            )
        
        if log_config.get('destinations', {}).get('file', True):
            log_file = log_config.get('file_path', 'logs/automl.log')
            logger.add(
                sink=log_file,
                format=log_format,
                level=log_config.get('level', 'INFO'),
                rotation=log_config.get('rotation', '1 week'),
                retention=log_config.get('retention', '30 days')
            )
    
    def initialize_components(self) -> None:
        """Initialize all AutoML components."""
        
        logger.info("Initializing AutoML components...")
        
        # Feature Engineering Components
        fe_config = self.config.get('feature_engineering', {})
        
        if fe_config.get('technical_indicators', {}).get('enabled', True):
            self.technical_indicators = TechnicalIndicators(
                config=fe_config.get('technical_indicators')
            )
        
        if fe_config.get('market_microstructure', {}).get('enabled', True):
            self.market_microstructure = MarketMicrostructure(
                config=fe_config.get('market_microstructure')
            )
        
        if fe_config.get('sentiment_features', {}).get('enabled', True):
            self.sentiment_features = SentimentFeatures(
                config=fe_config.get('sentiment_features')
            )
        
        if fe_config.get('pattern_recognition', {}).get('enabled', True):
            self.pattern_recognition = PatternRecognition(
                config=fe_config.get('pattern_recognition')
            )
        
        self.feature_selector = FeatureSelector(
            config=fe_config.get('feature_selection')
        )
        
        # Model Selection Components
        ms_config = self.config.get('model_selection', {})
        
        self.model_registry = ModelRegistry()
        
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            config=ms_config.get('hyperparameter_optimization')
        )
        
        self.ensemble_manager = EnsembleManager(
            config=ms_config.get('ensemble_management')
        )
        
        self.meta_learner = MetaLearner(
            config=ms_config.get('meta_learning')
        )
        
        # Regime Detection Components
        rd_config = self.config.get('regime_detection', {})
        
        if rd_config.get('volatility_regimes', {}).get('enabled', True):
            self.volatility_detector = VolatilityRegimeDetector(
                config=rd_config.get('volatility_regimes')
            )
        
        if rd_config.get('trend_detection', {}).get('enabled', True):
            self.trend_detector = TrendDetector(
                config=rd_config.get('trend_detection')
            )
        
        self.pipeline_state['initialized'] = True
        logger.info("All components initialized successfully")
    
    def fit(self, price_data: pd.DataFrame, 
           target: Optional[pd.Series] = None,
           external_data: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """
        Fit the AutoML pipeline on training data.
        
        Args:
            price_data: OHLCV price data
            target: Optional target variable (if None, will be generated)
            external_data: Optional external data sources
        """
        
        if not self.pipeline_state['initialized']:
            self.initialize_components()
        
        logger.info("Starting AutoML pipeline training...")
        
        # Generate target if not provided
        if target is None:
            target = self._generate_target(price_data)
        
        # Feature engineering
        features_df = self._engineer_features(price_data, external_data)
        
        # Feature selection
        selected_features_df = self._select_features(features_df, target)
        
        # Regime detection
        regimes = self._detect_regimes(price_data)
        
        # Model selection and training
        self._train_models(selected_features_df, target, regimes)
        
        # Update pipeline state
        self.pipeline_state['trained'] = True
        self.pipeline_state['last_update'] = datetime.now()
        
        logger.info("AutoML pipeline training completed successfully")
    
    def predict(self, price_data: pd.DataFrame,
               external_data: Optional[Dict[str, pd.DataFrame]] = None,
               return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the trained pipeline.
        
        Args:
            price_data: OHLCV price data
            external_data: Optional external data sources
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Predictions array, optionally with probabilities
        """
        
        if not self.pipeline_state['trained']:
            raise ValueError("Pipeline must be trained before making predictions")
        
        logger.info("Generating predictions...")
        
        # Feature engineering
        features_df = self._engineer_features(price_data, external_data)
        
        # Apply feature selection (use same features as training)
        if hasattr(self.feature_selector, 'selected_features_'):
            task_type = 'regression'  # Default, would be determined from training
            selected_feature_names = self.feature_selector.selected_features_.get(task_type, features_df.columns)
            features_df = features_df[selected_feature_names]
        
        # Detect current regime
        current_regime = self._detect_current_regime(price_data)
        
        # Make ensemble predictions
        predictions = self.ensemble_manager.predict(
            features_df, 
            regime=current_regime
        )
        
        if return_probabilities and hasattr(self.ensemble_manager, 'predict_proba'):
            try:
                probabilities = self.ensemble_manager.predict_proba(features_df)
                return predictions, probabilities
            except:
                logger.warning("Could not generate probabilities, returning predictions only")
        
        return predictions
    
    def _generate_target(self, price_data: pd.DataFrame) -> pd.Series:
        """Generate target variable from price data."""
        
        # Default: forward returns
        returns = np.log(price_data['close'] / price_data['close'].shift(1))
        target = returns.shift(-1)  # Next period return
        
        return target.fillna(0)
    
    def _engineer_features(self, price_data: pd.DataFrame, 
                          external_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Engineer features using all available feature engineering components."""
        
        # Start with price data
        features_df = price_data.copy()
        
        # Technical indicators
        if self.technical_indicators:
            features_df = self.technical_indicators.generate_features(features_df)
        
        # Market microstructure features
        if self.market_microstructure:
            order_book_data = external_data.get('order_book') if external_data else None
            trades_data = external_data.get('trades') if external_data else None
            
            features_df = self.market_microstructure.generate_features(
                features_df, order_book_data, trades_data
            )
        
        # Sentiment features
        if self.sentiment_features:
            social_data = external_data.get('social_media') if external_data else None
            news_data = external_data.get('news') if external_data else None
            
            features_df = self.sentiment_features.generate_features(
                features_df, social_data, news_data
            )
        
        # Pattern recognition features
        if self.pattern_recognition:
            features_df = self.pattern_recognition.generate_features(features_df)
        
        # Cache features for reuse
        self.feature_cache[datetime.now()] = features_df.copy()
        
        logger.info(f"Feature engineering completed: {len(features_df.columns)} features generated")
        return features_df
    
    def _select_features(self, features_df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select optimal features using the feature selector."""
        
        # Remove original OHLCV columns from feature selection (keep as context)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        features_for_selection = features_df[feature_columns]
        
        # Perform feature selection
        selected_features_df = self.feature_selector.select_features(
            features_for_selection, target, task_type='regression'
        )
        
        # Add back OHLCV columns for context
        result_df = pd.concat([
            features_df[['open', 'high', 'low', 'close', 'volume']],
            selected_features_df
        ], axis=1)
        
        logger.info(f"Feature selection completed: {len(selected_features_df.columns)} features selected")
        return result_df
    
    def _detect_regimes(self, price_data: pd.DataFrame) -> pd.Series:
        """Detect market regimes."""
        
        regimes = pd.Series('normal', index=price_data.index)
        
        # Volatility regimes
        if self.volatility_detector:
            vol_regimes = self.volatility_detector.detect_regimes(price_data)
            # Combine with existing regimes (simplified)
            regimes = vol_regimes
        
        # Trend regimes
        if self.trend_detector:
            trend_result = self.trend_detector.detect_trend(price_data)
            trend_direction = trend_result.get('overall_direction', 'unclear')
            
            # Create trend-based regime labels
            if trend_direction in ['uptrend', 'downtrend']:
                # Modify regime labels to include trend information
                regimes = regimes + '_' + trend_direction
        
        self.pipeline_state['active_regimes'] = {
            'current_regime': regimes.iloc[-1] if len(regimes) > 0 else 'normal',
            'regime_distribution': regimes.value_counts().to_dict()
        }
        
        return regimes
    
    def _detect_current_regime(self, price_data: pd.DataFrame) -> str:
        """Detect current market regime for prediction."""
        
        if len(price_data) < 10:
            return 'normal'
        
        # Use recent data for regime detection
        recent_data = price_data.tail(50)
        regimes = self._detect_regimes(recent_data)
        
        return regimes.iloc[-1] if len(regimes) > 0 else 'normal'
    
    def _train_models(self, features_df: pd.DataFrame, target: pd.Series, 
                     regimes: pd.Series) -> None:
        """Train models using the model selection components."""
        
        # Prepare clean data
        common_idx = features_df.index.intersection(target.index)
        X_clean = features_df.loc[common_idx]
        y_clean = target.loc[common_idx]
        
        # Remove rows with NaN values
        valid_rows = ~(X_clean.isna().any(axis=1) | y_clean.isna())
        X_clean = X_clean.loc[valid_rows]
        y_clean = y_clean.loc[valid_rows]
        
        if len(X_clean) < 50:
            logger.warning("Insufficient clean data for model training")
            return
        
        # Get model recommendations from meta-learner
        available_models = self.config.get('model_selection', {}).get(
            'model_registry', {}
        ).get('available_models', ['xgboost', 'randomforest'])
        
        recommended_model, confidence = self.meta_learner.recommend_model(
            X_clean, y_clean, available_models
        )
        
        logger.info(f"Recommended model: {recommended_model} (confidence: {confidence:.3f})")
        
        # Create and train recommended model
        model = self.model_registry.create_model(recommended_model, 'regression')
        
        # Optimize hyperparameters
        optimization_results = self.hyperparameter_optimizer.optimize_model(
            recommended_model, X_clean, y_clean, 'regression'
        )
        
        # Create model with optimized parameters
        optimized_model = self.model_registry.create_model(
            recommended_model, 'regression', 
            **optimization_results['best_params']
        )
        
        # Train the optimized model
        optimized_model.fit(X_clean, y_clean)
        
        # Register the trained model
        model_name = f"{recommended_model}_optimized"
        self.model_registry.register_model(model_name, optimized_model)
        
        # Add to ensemble
        self.ensemble_manager.add_model(model_name, optimized_model)
        
        # Train additional models for ensemble diversity
        self._train_ensemble_models(X_clean, y_clean, available_models)
        
        # Train meta-model for ensemble
        if len(self.ensemble_manager.models) > 1:
            self.ensemble_manager.train_meta_model(X_clean, y_clean)
        
        # Update meta-learning knowledge
        model_performances = {}
        for model_name, model in self.ensemble_manager.models.items():
            try:
                pred = model.predict(X_clean)
                mse = np.mean((y_clean - pred) ** 2)
                model_performances[model_name] = -mse  # Negative for maximization
            except:
                model_performances[model_name] = 0.0
        
        meta_features = self.meta_learner.extract_meta_features(X_clean, y_clean)
        self.meta_learner.update_knowledge(meta_features, model_performances)
        
        # Store current models
        self.pipeline_state['current_models'] = list(self.ensemble_manager.models.keys())
        
        logger.info(f"Model training completed. Ensemble contains {len(self.ensemble_manager.models)} models")
    
    def _train_ensemble_models(self, X: pd.DataFrame, y: pd.Series, 
                              available_models: List[str]) -> None:
        """Train additional models for ensemble diversity."""
        
        # Train a few additional models for diversity
        additional_models = [model for model in available_models[:3] 
                           if model not in self.ensemble_manager.models]
        
        for model_type in additional_models:
            try:
                # Create and train model
                model = self.model_registry.create_model(model_type, 'regression')
                model.fit(X, y)
                
                # Add to ensemble
                model_name = f"{model_type}_ensemble"
                self.model_registry.register_model(model_name, model)
                self.ensemble_manager.add_model(model_name, model)
                
                logger.info(f"Added {model_name} to ensemble")
                
            except Exception as e:
                logger.warning(f"Failed to train {model_type}: {e}")
    
    def evaluate(self, price_data: pd.DataFrame, target: Optional[pd.Series] = None,
                external_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, float]:
        """
        Evaluate the pipeline performance.
        
        Args:
            price_data: Test price data
            target: Optional target variable
            external_data: Optional external data
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        if target is None:
            target = self._generate_target(price_data)
        
        # Make predictions
        predictions = self.predict(price_data, external_data)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Align predictions and target
        min_len = min(len(predictions), len(target))
        pred_aligned = predictions[:min_len]
        target_aligned = target.iloc[:min_len]
        
        # Remove NaN values
        valid_idx = ~(pd.isna(pred_aligned) | pd.isna(target_aligned))
        pred_clean = pred_aligned[valid_idx]
        target_clean = target_aligned[valid_idx]
        
        if len(pred_clean) == 0:
            return {'error': 'No valid predictions for evaluation'}
        
        metrics = {
            'mse': mean_squared_error(target_clean, pred_clean),
            'rmse': np.sqrt(mean_squared_error(target_clean, pred_clean)),
            'mae': mean_absolute_error(target_clean, pred_clean),
            'correlation': np.corrcoef(target_clean, pred_clean)[0, 1] if len(pred_clean) > 1 else 0.0
        }
        
        # Trading-specific metrics
        if len(pred_clean) > 1 and np.std(pred_clean) > 0:
            sharpe_ratio = np.mean(pred_clean) / np.std(pred_clean) * np.sqrt(365 * 24)  # Annualized
            metrics['sharpe_ratio'] = sharpe_ratio
        
        # Store performance history
        self.pipeline_state['performance_history'].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return metrics
    
    def update_models(self, price_data: pd.DataFrame, target: Optional[pd.Series] = None,
                     external_data: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """
        Update models with new data.
        
        Args:
            price_data: New price data
            target: Optional target variable
            external_data: Optional external data
        """
        
        if target is None:
            target = self._generate_target(price_data)
        
        logger.info("Updating models with new data...")
        
        # Generate features
        features_df = self._engineer_features(price_data, external_data)
        
        # Apply existing feature selection
        if hasattr(self.feature_selector, 'selected_features_'):
            task_type = 'regression'
            selected_feature_names = self.feature_selector.selected_features_.get(task_type, features_df.columns)
            features_df = features_df[selected_feature_names]
        
        # Update ensemble weights based on recent performance
        self.ensemble_manager.update_weights(features_df, target)
        
        # Check if retraining is needed
        if self._should_retrain(price_data):
            logger.info("Retraining triggered")
            self.fit(price_data, target, external_data)
        
        self.pipeline_state['last_update'] = datetime.now()
        logger.info("Model update completed")
    
    def _should_retrain(self, price_data: pd.DataFrame) -> bool:
        """Determine if models should be retrained."""
        
        retraining_config = self.config.get('monitoring', {}).get('retraining', {})
        
        if not retraining_config.get('enabled', True):
            return False
        
        triggers = retraining_config.get('triggers', [])
        
        # Scheduled retraining
        if 'scheduled' in triggers:
            last_update = self.pipeline_state.get('last_update')
            if last_update:
                hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
                scheduled_interval = retraining_config.get('scheduled_interval', 168)
                
                if hours_since_update >= scheduled_interval:
                    logger.info("Scheduled retraining triggered")
                    return True
        
        # Performance degradation
        if 'performance_degradation' in triggers:
            if len(self.pipeline_state['performance_history']) >= 2:
                recent_performance = self.pipeline_state['performance_history'][-1]['metrics']
                previous_performance = self.pipeline_state['performance_history'][-2]['metrics']
                
                performance_metric = 'sharpe_ratio'  # or another key metric
                if performance_metric in recent_performance and performance_metric in previous_performance:
                    degradation = (previous_performance[performance_metric] - 
                                 recent_performance[performance_metric]) / abs(previous_performance[performance_metric])
                    
                    threshold = retraining_config.get('performance_threshold', 0.1)
                    if degradation > threshold:
                        logger.info("Performance degradation retraining triggered")
                        return True
        
        return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        
        status = {
            'pipeline_state': self.pipeline_state.copy(),
            'component_status': {},
            'performance_summary': {},
            'configuration': self.config
        }
        
        # Component status
        status['component_status'] = {
            'technical_indicators': self.technical_indicators is not None,
            'market_microstructure': self.market_microstructure is not None,
            'sentiment_features': self.sentiment_features is not None,
            'pattern_recognition': self.pattern_recognition is not None,
            'feature_selector': self.feature_selector is not None,
            'model_registry': self.model_registry is not None,
            'ensemble_manager': self.ensemble_manager is not None,
            'volatility_detector': self.volatility_detector is not None,
            'trend_detector': self.trend_detector is not None
        }
        
        # Performance summary
        if self.pipeline_state['performance_history']:
            latest_metrics = self.pipeline_state['performance_history'][-1]['metrics']
            status['performance_summary'] = {
                'latest_metrics': latest_metrics,
                'evaluation_count': len(self.pipeline_state['performance_history'])
            }
        
        return status
    
    def export_pipeline(self, filepath: str) -> None:
        """Export the entire pipeline configuration and state."""
        
        export_data = {
            'config': self.config,
            'pipeline_state': self.pipeline_state,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Export individual components
        if self.model_registry:
            model_registry_path = filepath.replace('.json', '_models')
            self.model_registry.save_registry(model_registry_path)
            export_data['model_registry_path'] = model_registry_path
        
        if self.meta_learner:
            meta_knowledge_path = filepath.replace('.json', '_meta_knowledge.json')
            self.meta_learner.export_meta_knowledge(meta_knowledge_path)
            export_data['meta_knowledge_path'] = meta_knowledge_path
        
        # Save main export file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Pipeline exported to {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """Load a previously exported pipeline."""
        
        with open(filepath, 'r') as f:
            export_data = json.load(f)
        
        # Load configuration and state
        self.config = export_data['config']
        self.pipeline_state = export_data['pipeline_state']
        
        # Reinitialize components with loaded config
        self.initialize_components()
        
        # Load models if available
        if 'model_registry_path' in export_data:
            self.model_registry.load_registry(export_data['model_registry_path'])
        
        # Load meta-learning knowledge
        if 'meta_knowledge_path' in export_data:
            self.meta_learner.load_meta_knowledge(export_data['meta_knowledge_path'])
        
        logger.info(f"Pipeline loaded from {filepath}")


def main():
    """Main entry point for the AutoML pipeline."""
    
    # Example usage
    pipeline = AutoMLPipeline()
    
    # This would normally be replaced with actual data loading
    print("AutoML Pipeline initialized successfully!")
    print(f"Pipeline status: {pipeline.get_pipeline_status()}")


if __name__ == "__main__":
    main()