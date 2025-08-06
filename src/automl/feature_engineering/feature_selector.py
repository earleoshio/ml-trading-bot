"""
Feature Selector Module

Implements intelligent feature selection for cryptocurrency trading,
including statistical methods, mutual information, and trading-specific selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    SelectKBest, SelectPercentile, RFE, RFECV,
    VarianceThreshold, chi2, f_regression
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Advanced feature selection for cryptocurrency trading.
    
    Provides multiple feature selection methods including:
    - Statistical selection (variance, correlation, mutual information)
    - Model-based selection (LASSO, Random Forest importance)
    - Trading-specific selection (regime-aware, stability-based)
    - Ensemble feature selection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature selector.
        
        Args:
            config: Configuration dictionary with selection parameters
        """
        self.config = config or self._default_config()
        self.selected_features_ = {}
        self.feature_importance_ = {}
        self.selection_history_ = []
        
    def _default_config(self) -> Dict:
        """Default configuration for feature selection."""
        return {
            'variance_threshold': 0.01,  # Remove features with low variance
            'correlation_threshold': 0.95,  # Remove highly correlated features
            'mutual_info_percentile': 50,  # Keep top N% of features by mutual info
            'lasso_alpha_range': np.logspace(-4, 1, 50),
            'random_forest_importance_threshold': 0.001,
            'stability_threshold': 0.7,  # Feature must be selected in 70% of CV folds
            'max_features': 100,  # Maximum number of features to select
            'regime_aware': True,  # Consider regime-specific performance
            'methods': {
                'variance_filter': True,
                'correlation_filter': True,
                'mutual_information': True,
                'lasso_selection': True,
                'random_forest': True,
                'recursive_elimination': False,  # Expensive, disabled by default
                'trading_specific': True
            }
        }
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       task_type: str = 'regression',
                       regime_labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Perform comprehensive feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'regression' or 'classification'
            regime_labels: Optional regime labels for regime-aware selection
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Starting feature selection with {len(X.columns)} features...")
        
        # Store original feature names
        original_features = X.columns.tolist()
        
        # Step 1: Variance filtering
        if self.config['methods']['variance_filter']:
            X = self._variance_filter(X)
            logger.info(f"After variance filtering: {len(X.columns)} features")
        
        # Step 2: Correlation filtering
        if self.config['methods']['correlation_filter']:
            X = self._correlation_filter(X)
            logger.info(f"After correlation filtering: {len(X.columns)} features")
        
        # Step 3: Mutual information selection
        if self.config['methods']['mutual_information']:
            X = self._mutual_information_selection(X, y, task_type)
            logger.info(f"After mutual information selection: {len(X.columns)} features")
        
        # Step 4: LASSO-based selection
        if self.config['methods']['lasso_selection']:
            X = self._lasso_selection(X, y, task_type)
            logger.info(f"After LASSO selection: {len(X.columns)} features")
        
        # Step 5: Random Forest importance
        if self.config['methods']['random_forest']:
            X = self._random_forest_selection(X, y, task_type)
            logger.info(f"After Random Forest selection: {len(X.columns)} features")
        
        # Step 6: Recursive feature elimination (optional)
        if self.config['methods']['recursive_elimination']:
            X = self._recursive_elimination(X, y, task_type)
            logger.info(f"After recursive elimination: {len(X.columns)} features")
        
        # Step 7: Trading-specific selection
        if self.config['methods']['trading_specific']:
            X = self._trading_specific_selection(X, y, regime_labels)
            logger.info(f"After trading-specific selection: {len(X.columns)} features")
        
        # Step 8: Final feature limit
        if len(X.columns) > self.config['max_features']:
            X = self._limit_features(X, y, task_type)
            logger.info(f"After feature limit: {len(X.columns)} features")
        
        # Store selection results
        self.selected_features_[task_type] = X.columns.tolist()
        
        # Log selection summary
        selection_summary = {
            'original_features': len(original_features),
            'selected_features': len(X.columns),
            'selection_ratio': len(X.columns) / len(original_features),
            'removed_features': len(original_features) - len(X.columns)
        }
        
        logger.info(f"Feature selection completed: {selection_summary}")
        self.selection_history_.append(selection_summary)
        
        return X
    
    def _variance_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance."""
        selector = VarianceThreshold(threshold=self.config['variance_threshold'])
        
        # Handle NaN values
        X_clean = X.fillna(X.mean())
        
        selected_mask = selector.fit_transform(X_clean)
        selected_features = X.columns[selector.get_support()]
        
        return X[selected_features]
    
    def _correlation_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Select features to drop
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > self.config['correlation_threshold'])]
        
        return X.drop(columns=to_drop)
    
    def _mutual_information_selection(self, X: pd.DataFrame, y: pd.Series, 
                                     task_type: str) -> pd.DataFrame:
        """Select features based on mutual information."""
        # Handle NaN values
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean() if task_type == 'regression' else y.mode()[0])
        
        # Align indices
        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]
        
        if task_type == 'regression':
            mi_scores = mutual_info_regression(X_clean, y_clean)
        else:
            mi_scores = mutual_info_classif(X_clean, y_clean)
        
        # Select top percentile
        selector = SelectPercentile(
            score_func=mutual_info_regression if task_type == 'regression' else mutual_info_classif,
            percentile=self.config['mutual_info_percentile']
        )
        
        selector.fit(X_clean, y_clean)
        selected_features = X.columns[selector.get_support()]
        
        # Store importance scores
        self.feature_importance_['mutual_info'] = dict(zip(X.columns, mi_scores))
        
        return X[selected_features]
    
    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> pd.DataFrame:
        """Select features using LASSO regularization."""
        # Handle NaN values
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean() if task_type == 'regression' else y.mode()[0])
        
        # Align indices
        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        if task_type == 'regression':
            # Use LassoCV for regression
            lasso = LassoCV(alphas=self.config['lasso_alpha_range'], cv=5, random_state=42)
            lasso.fit(X_scaled, y_clean)
            selected_mask = lasso.coef_ != 0
        else:
            # For classification, use LogisticRegressionCV with L1 penalty
            from sklearn.linear_model import LogisticRegressionCV
            lasso = LogisticRegressionCV(
                penalty='l1', solver='liblinear', 
                Cs=1/self.config['lasso_alpha_range'], cv=5, random_state=42
            )
            lasso.fit(X_scaled, y_clean)
            selected_mask = lasso.coef_[0] != 0
        
        selected_features = X.columns[selected_mask]
        
        # Store importance scores
        importance_scores = np.abs(lasso.coef_) if task_type == 'regression' else np.abs(lasso.coef_[0])
        self.feature_importance_['lasso'] = dict(zip(X.columns, importance_scores))
        
        return X[selected_features]
    
    def _random_forest_selection(self, X: pd.DataFrame, y: pd.Series, 
                                task_type: str) -> pd.DataFrame:
        """Select features using Random Forest feature importance."""
        # Handle NaN values
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean() if task_type == 'regression' else y.mode()[0])
        
        # Align indices
        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]
        
        if task_type == 'regression':
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        rf.fit(X_clean, y_clean)
        
        # Select features above importance threshold
        importance_scores = rf.feature_importances_
        selected_mask = importance_scores > self.config['random_forest_importance_threshold']
        selected_features = X.columns[selected_mask]
        
        # Store importance scores
        self.feature_importance_['random_forest'] = dict(zip(X.columns, importance_scores))
        
        return X[selected_features]
    
    def _recursive_elimination(self, X: pd.DataFrame, y: pd.Series, 
                              task_type: str) -> pd.DataFrame:
        """Select features using recursive feature elimination."""
        # Handle NaN values
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean() if task_type == 'regression' else y.mode()[0])
        
        # Align indices
        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]
        
        if task_type == 'regression':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Use RFECV for automatic feature number selection
        selector = RFECV(estimator, step=1, cv=3, scoring='r2' if task_type == 'regression' else 'accuracy')
        selector.fit(X_clean, y_clean)
        
        selected_features = X.columns[selector.support_]
        
        return X[selected_features]
    
    def _trading_specific_selection(self, X: pd.DataFrame, y: pd.Series, 
                                   regime_labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """Apply trading-specific feature selection criteria."""
        selected_features = list(X.columns)
        
        # Remove features with high noise-to-signal ratio
        noise_features = []
        for col in X.columns:
            if X[col].std() / (np.abs(X[col].mean()) + 1e-8) > 10:  # High coefficient of variation
                noise_features.append(col)
        
        selected_features = [f for f in selected_features if f not in noise_features]
        logger.info(f"Removed {len(noise_features)} noisy features")
        
        # Regime-aware selection if regime labels provided
        if regime_labels is not None and self.config['regime_aware']:
            stable_features = []
            
            for feature in selected_features:
                regime_performance = {}
                
                for regime in regime_labels.unique():
                    if pd.isna(regime):
                        continue
                    
                    regime_mask = regime_labels == regime
                    X_regime = X.loc[regime_mask, feature]
                    y_regime = y.loc[regime_mask]
                    
                    if len(X_regime) > 10:  # Minimum samples per regime
                        corr = X_regime.corr(y_regime)
                        regime_performance[regime] = corr if not pd.isna(corr) else 0
                
                # Keep features that perform well across regimes
                if regime_performance:
                    avg_performance = np.mean(list(regime_performance.values()))
                    std_performance = np.std(list(regime_performance.values()))
                    
                    # Feature is stable if it performs consistently across regimes
                    if avg_performance > 0.05 and std_performance < 0.3:
                        stable_features.append(feature)
            
            selected_features = stable_features
            logger.info(f"Selected {len(stable_features)} regime-stable features")
        
        return X[selected_features]
    
    def _limit_features(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> pd.DataFrame:
        """Limit number of features to maximum allowed."""
        if len(X.columns) <= self.config['max_features']:
            return X
        
        # Use combined importance scores to select top features
        combined_scores = self._calculate_combined_importance_scores(X.columns)
        
        # Select top features
        top_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in top_features[:self.config['max_features']]]
        
        return X[selected_features]
    
    def _calculate_combined_importance_scores(self, features: List[str]) -> Dict[str, float]:
        """Calculate combined importance scores from all methods."""
        combined_scores = {}
        
        for feature in features:
            score = 0
            count = 0
            
            # Average scores from all methods
            for method_name, scores in self.feature_importance_.items():
                if feature in scores:
                    score += scores[feature]
                    count += 1
            
            combined_scores[feature] = score / count if count > 0 else 0
        
        return combined_scores
    
    def get_feature_ranking(self, method: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature ranking based on importance scores.
        
        Args:
            method: Specific method name or None for combined ranking
            
        Returns:
            DataFrame with feature rankings
        """
        if method and method in self.feature_importance_:
            scores = self.feature_importance_[method]
        else:
            # Use combined scores
            all_features = set()
            for scores in self.feature_importance_.values():
                all_features.update(scores.keys())
            
            scores = self._calculate_combined_importance_scores(list(all_features))
        
        ranking_df = pd.DataFrame([
            {'feature': feature, 'importance': importance, 'rank': rank + 1}
            for rank, (feature, importance) in enumerate(
                sorted(scores.items(), key=lambda x: x[1], reverse=True)
            )
        ])
        
        return ranking_df
    
    def analyze_feature_stability(self, X: pd.DataFrame, y: pd.Series, 
                                 n_splits: int = 5) -> Dict[str, float]:
        """
        Analyze feature selection stability across different data splits.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of splits for stability analysis
            
        Returns:
            Dictionary with stability scores for each feature
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        feature_selections = []
        
        for train_idx, _ in kf.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            
            # Perform feature selection on this split
            selected_features = self._quick_feature_selection(X_train, y_train)
            feature_selections.append(set(selected_features))
        
        # Calculate stability scores
        stability_scores = {}
        for feature in X.columns:
            count = sum(1 for selection in feature_selections if feature in selection)
            stability_scores[feature] = count / n_splits
        
        return stability_scores
    
    def _quick_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Quick feature selection for stability analysis."""
        # Use mutual information for quick selection
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean())
        
        mi_scores = mutual_info_regression(X_clean, y_clean)
        threshold = np.percentile(mi_scores, 70)  # Top 30% of features
        
        return X.columns[mi_scores > threshold].tolist()
    
    def plot_feature_importance(self, method: Optional[str] = None, top_n: int = 20):
        """
        Plot feature importance (requires matplotlib).
        
        Args:
            method: Specific method name or None for combined
            top_n: Number of top features to show
        """
        try:
            import matplotlib.pyplot as plt
            
            ranking_df = self.get_feature_ranking(method).head(top_n)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(ranking_df)), ranking_df['importance'])
            plt.yticks(range(len(ranking_df)), ranking_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance ({method or "Combined"})')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available. Cannot plot feature importance.")
    
    def export_selection_report(self, filepath: str):
        """Export detailed feature selection report."""
        report = {
            'config': self.config,
            'selected_features': self.selected_features_,
            'feature_importance': self.feature_importance_,
            'selection_history': self.selection_history_
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Feature selection report exported to {filepath}")