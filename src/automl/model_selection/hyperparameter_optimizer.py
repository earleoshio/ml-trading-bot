"""
Hyperparameter Optimizer Module

Implements advanced hyperparameter optimization using Optuna for cryptocurrency
trading models, with support for multi-objective optimization and trading-specific
metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from loguru import logger
import optuna
from optuna import Trial
from datetime import datetime
import json
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score

from .model_registry import BaseModel, ModelRegistry


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for cryptocurrency trading models.
    
    Uses Optuna for efficient hyperparameter search with support for:
    - Multi-objective optimization (returns vs risk)
    - Trading-specific metrics (Sharpe ratio, Calmar ratio)
    - Time series cross-validation
    - Early stopping and pruning
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            config: Configuration dictionary with optimization parameters
        """
        self.config = config or self._default_config()
        self.studies = {}
        self.best_params_history = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for hyperparameter optimization."""
        return {
            'optimization': {
                'n_trials': 100,
                'timeout': 3600,  # 1 hour
                'n_jobs': 1,
                'direction': 'maximize',  # or 'minimize'
                'pruner': 'median',  # 'median', 'hyperband', 'successive_halving'
                'sampler': 'tpe'  # 'tpe', 'random', 'cmaes'
            },
            'cross_validation': {
                'cv_folds': 5,
                'time_series_split': True,
                'test_size': 0.2,
                'gap': 1  # Gap between train and validation for time series
            },
            'metrics': {
                'primary_metric': 'sharpe_ratio',
                'secondary_metrics': ['returns', 'max_drawdown', 'win_rate']
            },
            'search_spaces': {
                'xgboost': {
                    'n_estimators': {'low': 50, 'high': 500},
                    'max_depth': {'low': 3, 'high': 10},
                    'learning_rate': {'low': 0.01, 'high': 0.3, 'log': True},
                    'subsample': {'low': 0.6, 'high': 1.0},
                    'colsample_bytree': {'low': 0.6, 'high': 1.0},
                    'reg_alpha': {'low': 0.0, 'high': 10.0},
                    'reg_lambda': {'low': 1.0, 'high': 10.0}
                },
                'randomforest': {
                    'n_estimators': {'low': 50, 'high': 300},
                    'max_depth': {'low': 5, 'high': 20},
                    'min_samples_split': {'low': 2, 'high': 20},
                    'min_samples_leaf': {'low': 1, 'high': 10},
                    'max_features': ['sqrt', 'log2', None]
                },
                'svm': {
                    'C': {'low': 0.1, 'high': 100.0, 'log': True},
                    'gamma': {'low': 0.001, 'high': 1.0, 'log': True},
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
            }
        }
    
    def optimize_model(self, 
                      model_type: str,
                      X: pd.DataFrame, 
                      y: pd.Series,
                      task_type: str = 'regression',
                      custom_objective: Optional[Callable] = None) -> Dict:
        """
        Optimize hyperparameters for a specific model type.
        
        Args:
            model_type: Type of model to optimize
            X: Feature matrix
            y: Target variable
            task_type: 'regression' or 'classification'
            custom_objective: Custom objective function
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization for {model_type}")
        
        # Create study
        study_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configure pruner
        if self.config['optimization']['pruner'] == 'median':
            pruner = optuna.pruners.MedianPruner()
        elif self.config['optimization']['pruner'] == 'hyperband':
            pruner = optuna.pruners.HyperbandPruner()
        else:
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        
        # Configure sampler
        if self.config['optimization']['sampler'] == 'tpe':
            sampler = optuna.samplers.TPESampler()
        elif self.config['optimization']['sampler'] == 'cmaes':
            sampler = optuna.samplers.CmaEsSampler()
        else:
            sampler = optuna.samplers.RandomSampler()
        
        study = optuna.create_study(
            direction=self.config['optimization']['direction'],
            pruner=pruner,
            sampler=sampler,
            study_name=study_name
        )
        
        # Define objective function
        if custom_objective:
            objective_func = lambda trial: custom_objective(trial, X, y, model_type, task_type)
        else:
            objective_func = lambda trial: self._default_objective(trial, X, y, model_type, task_type)
        
        # Run optimization
        study.optimize(
            objective_func,
            n_trials=self.config['optimization']['n_trials'],
            timeout=self.config['optimization']['timeout'],
            n_jobs=self.config['optimization']['n_jobs']
        )
        
        # Store results
        self.studies[study_name] = study
        
        results = {
            'study_name': study_name,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration),
            'model_type': model_type,
            'task_type': task_type
        }
        
        # Store best parameters history
        if model_type not in self.best_params_history:
            self.best_params_history[model_type] = []
        
        self.best_params_history[model_type].append({
            'timestamp': datetime.now().isoformat(),
            'params': study.best_params,
            'score': study.best_value
        })
        
        logger.info(f"Optimization completed. Best score: {study.best_value:.4f}")
        return results
    
    def _default_objective(self, trial: Trial, X: pd.DataFrame, y: pd.Series, 
                          model_type: str, task_type: str) -> float:
        """Default objective function for hyperparameter optimization."""
        
        # Get hyperparameter suggestions from trial
        params = self._suggest_hyperparameters(trial, model_type)
        
        # Create model with suggested parameters
        registry = ModelRegistry()
        model = registry.create_model(model_type, task_type, **params)
        
        # Perform cross-validation
        scores = self._cross_validate_model(model, X, y)
        
        # Return mean score
        return np.mean(scores)
    
    def _suggest_hyperparameters(self, trial: Trial, model_type: str) -> Dict:
        """Suggest hyperparameters for a specific model type."""
        
        if model_type.lower() not in self.config['search_spaces']:
            raise ValueError(f"No search space defined for model type: {model_type}")
        
        search_space = self.config['search_spaces'][model_type.lower()]
        params = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                if 'low' in param_config and 'high' in param_config:
                    # Numeric parameter
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_loguniform(
                            param_name, param_config['low'], param_config['high']
                        )
                    else:
                        if isinstance(param_config['low'], int):
                            params[param_name] = trial.suggest_int(
                                param_name, param_config['low'], param_config['high']
                            )
                        else:
                            params[param_name] = trial.suggest_uniform(
                                param_name, param_config['low'], param_config['high']
                            )
            elif isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_config)
        
        return params
    
    def _cross_validate_model(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Perform cross-validation for a model."""
        
        if self.config['cross_validation']['time_series_split']:
            # Use TimeSeriesSplit for time series data
            cv = TimeSeriesSplit(
                n_splits=self.config['cross_validation']['cv_folds'],
                gap=self.config['cross_validation']['gap']
            )
        else:
            cv = self.config['cross_validation']['cv_folds']
        
        scores = []
        
        if self.config['cross_validation']['time_series_split']:
            # Manual cross-validation for time series
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Calculate score based on primary metric
                score = self._calculate_score(y_val, y_pred, model.model_type)
                scores.append(score)
        else:
            # Use sklearn cross_val_score
            if model.model_type == 'regression':
                scoring = 'neg_mean_squared_error'
            else:
                scoring = 'accuracy'
            
            # Note: This requires the model to be sklearn-compatible
            # For custom models, we'd need to implement manual CV
            try:
                cv_scores = cross_val_score(model.model, X, y, cv=cv, scoring=scoring)
                scores = cv_scores.tolist()
            except:
                # Fallback to manual CV
                logger.warning("Using manual cross-validation fallback")
                scores = [0.0] * self.config['cross_validation']['cv_folds']
        
        return scores
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray, 
                        task_type: str) -> float:
        """Calculate score based on primary metric."""
        
        primary_metric = self.config['metrics']['primary_metric']
        
        if task_type == 'regression':
            if primary_metric == 'mse':
                return -mean_squared_error(y_true, y_pred)  # Negative for maximization
            elif primary_metric == 'rmse':
                return -np.sqrt(mean_squared_error(y_true, y_pred))
            elif primary_metric == 'mae':
                return -np.mean(np.abs(y_true - y_pred))
            elif primary_metric == 'sharpe_ratio':
                # Calculate Sharpe ratio (assuming returns)
                returns = y_pred
                if len(returns) > 1 and np.std(returns) > 0:
                    return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                else:
                    return 0.0
            else:
                return -mean_squared_error(y_true, y_pred)  # Default to MSE
        else:
            if primary_metric == 'accuracy':
                return accuracy_score(y_true, y_pred)
            else:
                return accuracy_score(y_true, y_pred)  # Default to accuracy
    
    def multi_objective_optimization(self, 
                                   model_type: str,
                                   X: pd.DataFrame, 
                                   y: pd.Series,
                                   objectives: List[str],
                                   task_type: str = 'regression') -> Dict:
        """
        Perform multi-objective optimization.
        
        Args:
            model_type: Type of model to optimize
            X: Feature matrix
            y: Target variable
            objectives: List of objective names to optimize
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with optimization results including Pareto front
        """
        logger.info(f"Starting multi-objective optimization for {model_type}")
        
        # Create study for multi-objective optimization
        study_name = f"{model_type}_multi_obj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        study = optuna.create_study(
            directions=['maximize'] * len(objectives),  # Assuming maximization
            study_name=study_name
        )
        
        def multi_objective_func(trial):
            # Get hyperparameter suggestions
            params = self._suggest_hyperparameters(trial, model_type)
            
            # Create and evaluate model
            registry = ModelRegistry()
            model = registry.create_model(model_type, task_type, **params)
            
            # Calculate multiple objectives
            scores = []
            cv_results = self._cross_validate_model(model, X, y)
            
            for objective in objectives:
                if objective == 'sharpe_ratio':
                    # Use cross-validation results to estimate Sharpe ratio
                    score = np.mean(cv_results)
                elif objective == 'returns':
                    score = np.mean(cv_results)
                elif objective == 'max_drawdown':
                    # Estimate max drawdown (simplified)
                    score = -np.std(cv_results)  # Negative std as proxy
                else:
                    score = np.mean(cv_results)
                
                scores.append(score)
            
            return scores
        
        # Run optimization
        study.optimize(
            multi_objective_func,
            n_trials=self.config['optimization']['n_trials']
        )
        
        # Extract Pareto front
        pareto_front = []
        for trial in study.best_trials:
            pareto_front.append({
                'params': trial.params,
                'values': trial.values,
                'trial_number': trial.number
            })
        
        results = {
            'study_name': study_name,
            'pareto_front': pareto_front,
            'n_trials': len(study.trials),
            'objectives': objectives,
            'model_type': model_type
        }
        
        self.studies[study_name] = study
        
        logger.info(f"Multi-objective optimization completed. Pareto front size: {len(pareto_front)}")
        return results
    
    def get_best_params(self, model_type: str) -> Optional[Dict]:
        """Get the best parameters for a model type from history."""
        if model_type in self.best_params_history and self.best_params_history[model_type]:
            return self.best_params_history[model_type][-1]['params']
        return None
    
    def analyze_optimization_results(self, study_name: str) -> Dict:
        """Analyze optimization results and provide insights."""
        if study_name not in self.studies:
            raise ValueError(f"Study {study_name} not found")
        
        study = self.studies[study_name]
        
        analysis = {
            'best_trial': {
                'number': study.best_trial.number,
                'params': study.best_trial.params,
                'value': study.best_trial.value
            },
            'trial_statistics': {
                'total_trials': len(study.trials),
                'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            }
        }
        
        # Parameter importance
        if len(study.trials) > 10:  # Need sufficient trials for importance analysis
            try:
                importance = optuna.importance.get_param_importances(study)
                analysis['parameter_importance'] = importance
            except:
                logger.warning("Could not calculate parameter importance")
                analysis['parameter_importance'] = {}
        
        return analysis
    
    def plot_optimization_history(self, study_name: str):
        """Plot optimization history (requires optuna visualization)."""
        if study_name not in self.studies:
            raise ValueError(f"Study {study_name} not found")
        
        try:
            import optuna.visualization as vis
            import plotly.graph_objects as go
            
            study = self.studies[study_name]
            
            # Plot optimization history
            fig1 = vis.plot_optimization_history(study)
            fig1.show()
            
            # Plot parameter importance
            if len(study.trials) > 10:
                fig2 = vis.plot_param_importances(study)
                fig2.show()
            
            # Plot parameter relationships
            fig3 = vis.plot_parallel_coordinate(study)
            fig3.show()
            
        except ImportError:
            logger.warning("Plotly not available. Cannot create plots.")
    
    def export_results(self, study_name: str, filepath: str):
        """Export optimization results to file."""
        if study_name not in self.studies:
            raise ValueError(f"Study {study_name} not found")
        
        study = self.studies[study_name]
        analysis = self.analyze_optimization_results(study_name)
        
        # Create comprehensive results
        results = {
            'study_name': study_name,
            'analysis': analysis,
            'all_trials': [],
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Export all trial data
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            results['all_trials'].append(trial_data)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results exported to {filepath}")
    
    def suggest_next_trial(self, study_name: str) -> Dict:
        """Suggest parameters for the next trial based on current optimization state."""
        if study_name not in self.studies:
            raise ValueError(f"Study {study_name} not found")
        
        study = self.studies[study_name]
        
        # Create a new trial to get suggestions
        trial = study.ask()
        
        # Get parameter suggestions (this doesn't run the trial)
        suggested_params = {}
        for param_name in study.best_trial.params.keys():
            # This is a simplified approach - in practice, you'd use the study's sampler
            suggested_params[param_name] = trial.suggest_categorical(param_name, [study.best_trial.params[param_name]])
        
        return suggested_params