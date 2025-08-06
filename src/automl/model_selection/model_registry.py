"""
Model Registry Module

Manages a registry of different ML models optimized for cryptocurrency trading,
including LSTM, XGBoost, RandomForest, SVM, and Transformer models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from loguru import logger
import joblib
import json
from datetime import datetime
from abc import ABC, abstractmethod

# ML model imports
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score


class BaseModel(ABC):
    """Abstract base class for all models in the registry."""
    
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type  # 'regression' or 'classification'
        self.model = None
        self.is_fitted = False
        self.training_time = 0
        self.metadata = {}
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict:
        """Get default parameters for the model."""
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classification models)."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"Model {self.name} does not support probability prediction")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if available."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names_, self.model.feature_importances_))
        return None
    
    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        model_data = {
            'name': self.name,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'metadata': self.metadata
        }
        
        # Save model separately using joblib
        joblib.dump(self.model, f"{filepath}_model.pkl")
        
        # Save metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load the model from disk."""
        # Load model
        self.model = joblib.load(f"{filepath}_model.pkl")
        
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        self.is_fitted = model_data['is_fitted']
        self.training_time = model_data['training_time']
        self.metadata = model_data['metadata']


class XGBoostModel(BaseModel):
    """XGBoost model wrapper for crypto trading."""
    
    def __init__(self, model_type: str = 'regression', **kwargs):
        super().__init__('XGBoost', model_type)
        self.params = {**self.get_default_params(), **kwargs}
        
    def get_default_params(self) -> Dict:
        """Default XGBoost parameters optimized for financial data."""
        if self.model_type == 'regression':
            return {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            return {
                'objective': 'binary:logistic',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit XGBoost model."""
        start_time = datetime.now()
        
        self.feature_names_ = X.columns.tolist()
        
        if self.model_type == 'regression':
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = xgb.XGBClassifier(**self.params)
        
        self.model.fit(X, y)
        self.is_fitted = True
        self.training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"XGBoost model fitted in {self.training_time:.2f} seconds")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class RandomForestModel(BaseModel):
    """Random Forest model wrapper for crypto trading."""
    
    def __init__(self, model_type: str = 'regression', **kwargs):
        super().__init__('RandomForest', model_type)
        self.params = {**self.get_default_params(), **kwargs}
        
    def get_default_params(self) -> Dict:
        """Default Random Forest parameters."""
        if self.model_type == 'regression':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Random Forest model."""
        start_time = datetime.now()
        
        self.feature_names_ = X.columns.tolist()
        
        if self.model_type == 'regression':
            self.model = RandomForestRegressor(**self.params)
        else:
            self.model = RandomForestClassifier(**self.params)
        
        self.model.fit(X, y)
        self.is_fitted = True
        self.training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Random Forest model fitted in {self.training_time:.2f} seconds")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class SVMModel(BaseModel):
    """Support Vector Machine model wrapper for crypto trading."""
    
    def __init__(self, model_type: str = 'regression', **kwargs):
        super().__init__('SVM', model_type)
        self.params = {**self.get_default_params(), **kwargs}
        
    def get_default_params(self) -> Dict:
        """Default SVM parameters."""
        if self.model_type == 'regression':
            return {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1,
                'gamma': 'scale'
            }
        else:
            return {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True
            }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit SVM model."""
        start_time = datetime.now()
        
        self.feature_names_ = X.columns.tolist()
        
        if self.model_type == 'regression':
            self.model = SVR(**self.params)
        else:
            self.model = SVC(**self.params)
        
        self.model.fit(X, y)
        self.is_fitted = True
        self.training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"SVM model fitted in {self.training_time:.2f} seconds")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class LSTMModel(BaseModel):
    """LSTM model wrapper for crypto trading (placeholder for now)."""
    
    def __init__(self, model_type: str = 'regression', **kwargs):
        super().__init__('LSTM', model_type)
        self.params = {**self.get_default_params(), **kwargs}
        
    def get_default_params(self) -> Dict:
        """Default LSTM parameters."""
        return {
            'sequence_length': 60,
            'hidden_units': 50,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit LSTM model (placeholder implementation)."""
        start_time = datetime.now()
        
        self.feature_names_ = X.columns.tolist()
        
        # Placeholder - would implement actual LSTM training
        logger.warning("LSTM model is not fully implemented yet")
        
        self.is_fitted = True
        self.training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"LSTM model fitted in {self.training_time:.2f} seconds")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (placeholder)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Placeholder - return zeros for now
        return np.zeros(len(X))


class TransformerModel(BaseModel):
    """Transformer model wrapper for crypto trading (placeholder for now)."""
    
    def __init__(self, model_type: str = 'regression', **kwargs):
        super().__init__('Transformer', model_type)
        self.params = {**self.get_default_params(), **kwargs}
        
    def get_default_params(self) -> Dict:
        """Default Transformer parameters."""
        return {
            'sequence_length': 60,
            'num_layers': 4,
            'num_heads': 8,
            'hidden_dim': 512,
            'dropout': 0.1,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.0001
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Transformer model (placeholder implementation)."""
        start_time = datetime.now()
        
        self.feature_names_ = X.columns.tolist()
        
        # Placeholder - would implement actual Transformer training
        logger.warning("Transformer model is not fully implemented yet")
        
        self.is_fitted = True
        self.training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Transformer model fitted in {self.training_time:.2f} seconds")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (placeholder)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Placeholder - return zeros for now
        return np.zeros(len(X))


class MLPModel(BaseModel):
    """Multi-Layer Perceptron model wrapper for crypto trading."""
    
    def __init__(self, model_type: str = 'regression', **kwargs):
        super().__init__('MLP', model_type)
        self.params = {**self.get_default_params(), **kwargs}
        
    def get_default_params(self) -> Dict:
        """Default MLP parameters."""
        if self.model_type == 'regression':
            return {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'random_state': 42
            }
        else:
            return {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'random_state': 42
            }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit MLP model."""
        start_time = datetime.now()
        
        self.feature_names_ = X.columns.tolist()
        
        if self.model_type == 'regression':
            self.model = MLPRegressor(**self.params)
        else:
            self.model = MLPClassifier(**self.params)
        
        self.model.fit(X, y)
        self.is_fitted = True
        self.training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"MLP model fitted in {self.training_time:.2f} seconds")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class ModelRegistry:
    """
    Registry for managing different ML models for cryptocurrency trading.
    
    Provides centralized access to various model types and their configurations,
    with support for model lifecycle management and performance tracking.
    """
    
    def __init__(self):
        self.models = {}
        self.model_classes = {
            'xgboost': XGBoostModel,
            'randomforest': RandomForestModel,
            'svm': SVMModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel,
            'mlp': MLPModel
        }
        self.performance_history = {}
        
    def register_model(self, model_name: str, model: BaseModel) -> None:
        """Register a model instance in the registry."""
        self.models[model_name] = model
        logger.info(f"Model '{model_name}' registered successfully")
    
    def create_model(self, model_type: str, task_type: str = 'regression', 
                    model_name: Optional[str] = None, **kwargs) -> BaseModel:
        """
        Create a new model instance.
        
        Args:
            model_type: Type of model ('xgboost', 'randomforest', etc.)
            task_type: 'regression' or 'classification'
            model_name: Optional custom name for the model
            **kwargs: Additional parameters for the model
            
        Returns:
            Created model instance
        """
        if model_type.lower() not in self.model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.model_classes[model_type.lower()]
        model = model_class(model_type=task_type, **kwargs)
        
        if model_name:
            model.name = model_name
        
        return model
    
    def get_model(self, model_name: str) -> BaseModel:
        """Get a registered model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in registry")
        return self.models[model_name]
    
    def list_models(self) -> List[str]:
        """Get list of registered model names."""
        return list(self.models.keys())
    
    def get_available_model_types(self) -> List[str]:
        """Get list of available model types."""
        return list(self.model_classes.keys())
    
    def remove_model(self, model_name: str) -> None:
        """Remove a model from the registry."""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Model '{model_name}' removed from registry")
        else:
            logger.warning(f"Model '{model_name}' not found in registry")
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        model = self.get_model(model_name)
        predictions = model.predict(X_test)
        
        if model.model_type == 'regression':
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': np.mean(np.abs(y_test - predictions))
            }
        else:
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted'),
                'recall': recall_score(y_test, predictions, average='weighted')
            }
        
        # Store performance history
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return metrics
    
    def get_model_performance(self, model_name: str) -> List[Dict]:
        """Get performance history for a model."""
        return self.performance_history.get(model_name, [])
    
    def save_registry(self, filepath: str) -> None:
        """Save all models in the registry to disk."""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        registry_info = {
            'models': list(self.models.keys()),
            'performance_history': self.performance_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save registry metadata
        with open(f"{filepath}/registry_info.json", 'w') as f:
            json.dump(registry_info, f, indent=2)
        
        # Save each model
        for model_name, model in self.models.items():
            model.save(f"{filepath}/{model_name}")
        
        logger.info(f"Model registry saved to {filepath}")
    
    def load_registry(self, filepath: str) -> None:
        """Load models from disk into the registry."""
        # Load registry metadata
        with open(f"{filepath}/registry_info.json", 'r') as f:
            registry_info = json.load(f)
        
        self.performance_history = registry_info['performance_history']
        
        # Load each model
        for model_name in registry_info['models']:
            # Determine model type from metadata
            with open(f"{filepath}/{model_name}_metadata.json", 'r') as f:
                model_data = json.load(f)
            
            # Create model instance and load
            model_type = model_name.split('_')[0].lower()  # Assumes naming convention
            if model_type in self.model_classes:
                model = self.create_model(model_type, model_data['model_type'])
                model.load(f"{filepath}/{model_name}")
                self.register_model(model_name, model)
        
        logger.info(f"Model registry loaded from {filepath}")
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, BaseModel]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        best_model_name = None
        best_score = None
        
        for model_name, history in self.performance_history.items():
            if history:
                latest_metrics = history[-1]['metrics']
                if metric in latest_metrics:
                    score = latest_metrics[metric]
                    
                    # For regression metrics (lower is better), invert comparison
                    if metric in ['mse', 'rmse', 'mae']:
                        if best_score is None or score < best_score:
                            best_score = score
                            best_model_name = model_name
                    else:  # For classification metrics (higher is better)
                        if best_score is None or score > best_score:
                            best_score = score
                            best_model_name = model_name
        
        if best_model_name:
            return best_model_name, self.get_model(best_model_name)
        else:
            raise ValueError(f"No models found with metric '{metric}'")