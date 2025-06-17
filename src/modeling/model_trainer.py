import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import logging

class ModelTrainer:
    """
    Comprehensive model training pipeline for insurance risk analytics.
    
    Supports multiple algorithms for both regression and classification tasks
    with hyperparameter tuning and cross-validation.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def train_linear_regression(self, X_train, y_train, cv_folds=5):
        """
        Train Linear Regression model with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
        """
        self.logger.info("Training Linear Regression model...")
        
        # Linear Regression doesn't have hyperparameters to tune
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(lr_model, X_train, y_train, 
                                   cv=cv_folds, scoring='neg_mean_squared_error')
        
        self.models['linear_regression'] = lr_model
        self.cv_scores['linear_regression'] = {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'individual_scores': cv_scores
        }
        
        self.logger.info(f"Linear Regression CV RMSE: {np.sqrt(-np.mean(cv_scores)):.4f} ± {np.sqrt(np.std(cv_scores)):.4f}")
        
        return lr_model
    
    def train_decision_tree(self, X_train, y_train, problem_type='regression', 
                           tune_hyperparameters=True, cv_folds=5):
        """
        Train Decision Tree with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            problem_type: 'regression' or 'classification'
            tune_hyperparameters: Whether to perform grid search
            cv_folds: Number of cross-validation folds
        """
        self.logger.info(f"Training Decision Tree for {problem_type}...")
        
        if problem_type == 'regression':
            base_model = DecisionTreeRegressor(random_state=self.random_state)
            scoring = 'neg_mean_squared_error'
        else:
            base_model = DecisionTreeClassifier(random_state=self.random_state)
            scoring = 'accuracy'
        
        if tune_hyperparameters:
            param_grid = {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'max_features': ['auto', 'sqrt', 'log2', None]
            }
            
            grid_search = GridSearchCV(
                base_model, param_grid, cv=cv_folds, 
                scoring=scoring, n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            self.best_params['decision_tree'] = grid_search.best_params_
            self.logger.info(f"Best Decision Tree params: {grid_search.best_params_}")
        else:
            best_model = base_model
            best_model.fit(X_train, y_train)
        
        # Cross-validation with best model
        cv_scores = cross_val_score(best_model, X_train, y_train, 
                                   cv=cv_folds, scoring=scoring)
        
        self.models['decision_tree'] = best_model
        self.cv_scores['decision_tree'] = {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'individual_scores': cv_scores
        }
        
        if problem_type == 'regression':
            self.logger.info(f"Decision Tree CV RMSE: {np.sqrt(-np.mean(cv_scores)):.4f} ± {np.sqrt(np.std(cv_scores)):.4f}")
        else:
            self.logger.info(f"Decision Tree CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return best_model
    
    def train_random_forest(self, X_train, y_train, problem_type='regression', 
                           tune_hyperparameters=True, cv_folds=5):
        """
        Train Random Forest with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            problem_type: 'regression' or 'classification'
            tune_hyperparameters: Whether to perform grid search
            cv_folds: Number of cross-validation folds
        """
        self.logger.info(f"Training Random Forest for {problem_type}...")
        
        if problem_type == 'regression':
            base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            scoring = 'neg_mean_squared_error'
        else:
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            scoring = 'accuracy'
        
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt']
            }
            
            grid_search = GridSearchCV(
                base_model, param_grid, cv=cv_folds, 
                scoring=scoring, n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            self.best_params['random_forest'] = grid_search.best_params_
            self.logger.info(f"Best Random Forest params: {grid_search.best_params_}")
        else:
            # Use reasonable default parameters
            if problem_type == 'regression':
                best_model = RandomForestRegressor(
                    n_estimators=200, max_depth=10, min_samples_split=5,
                    random_state=self.random_state, n_jobs=-1
                )
            else:
                best_model = RandomForestClassifier(
                    n_estimators=200, max_depth=10, min_samples_split=5,
                    random_state=self.random_state, n_jobs=-1
                )
            best_model.fit(X_train, y_train)
        
        # Cross-validation with best model
        cv_scores = cross_val_score(best_model, X_train, y_train, 
                                   cv=cv_folds, scoring=scoring)
        
        self.models['random_forest'] = best_model
        self.cv_scores['random_forest'] = {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'individual_scores': cv_scores
        }
        
        if problem_type == 'regression':
            self.logger.info(f"Random Forest CV RMSE: {np.sqrt(-np.mean(cv_scores)):.4f} ± {np.sqrt(np.std(cv_scores)):.4f}")
        else:
            self.logger.info(f"Random Forest CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return best_model
    
    def train_xgboost(self, X_train, y_train, problem_type='regression', 
                     tune_hyperparameters=True, cv_folds=5):
        """
        Train XGBoost with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            problem_type: 'regression' or 'classification'
            tune_hyperparameters: Whether to perform grid search
            cv_folds: Number of cross-validation folds
        """
        self.logger.info(f"Training XGBoost for {problem_type}...")
        
        if problem_type == 'regression':
            base_model = xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1)
            scoring = 'neg_mean_squared_error'
        else:
            base_model = xgb.XGBClassifier(random_state=self.random_state, n_jobs=-1)
            scoring = 'accuracy'
        
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            grid_search = GridSearchCV(
                base_model, param_grid, cv=cv_folds, 
                scoring=scoring, n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            self.best_params['xgboost'] = grid_search.best_params_
            self.logger.info(f"Best XGBoost params: {grid_search.best_params_}")
        else:
            # Use reasonable default parameters
            if problem_type == 'regression':
                best_model = xgb.XGBRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=5,
                    random_state=self.random_state, n_jobs=-1
                )
            else:
                best_model = xgb.XGBClassifier(
                    n_estimators=200, learning_rate=0.1, max_depth=5,
                    random_state=self.random_state, n_jobs=-1
                )
            best_model.fit(X_train, y_train)
        
        # Cross-validation with best model
        cv_scores = cross_val_score(best_model, X_train, y_train, 
                                   cv=cv_folds, scoring=scoring)
        
        self.models['xgboost'] = best_model
        self.cv_scores['xgboost'] = {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'individual_scores': cv_scores
        }
        
        if problem_type == 'regression':
            self.logger.info(f"XGBoost CV RMSE: {np.sqrt(-np.mean(cv_scores)):.4f} ± {np.sqrt(np.std(cv_scores)):.4f}")
        else:
            self.logger.info(f"XGBoost CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return best_model
    
    def train_all_models(self, X_train, y_train, problem_type='regression', 
                        tune_hyperparameters=False, cv_folds=5):
        """
        Train all available models for comparison.
        
        Args:
            X_train: Training features
            y_train: Training target
            problem_type: 'regression' or 'classification'
            tune_hyperparameters: Whether to perform grid search
            cv_folds: Number of cross-validation folds
        """
        self.logger.info(f"Training all models for {problem_type} task...")
        
        models_trained = {}
        
        # Train each model
        if problem_type == 'regression':
            models_trained['linear_regression'] = self.train_linear_regression(
                X_train, y_train, cv_folds
            )
        
        models_trained['decision_tree'] = self.train_decision_tree(
            X_train, y_train, problem_type, tune_hyperparameters, cv_folds
        )
        
        models_trained['random_forest'] = self.train_random_forest(
            X_train, y_train, problem_type, tune_hyperparameters, cv_folds
        )
        
        models_trained['xgboost'] = self.train_xgboost(
            X_train, y_train, problem_type, tune_hyperparameters, cv_folds
        )
        
        return models_trained
    
    def get_feature_importance(self, model_name):
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            self.logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return None
    
    def save_models(self, filepath_prefix):
        """
        Save all trained models.
        
        Args:
            filepath_prefix: Prefix for model file paths
        """
        for model_name, model in self.models.items():
            filepath = f"{filepath_prefix}_{model_name}.joblib"
            joblib.dump(model, filepath)
            self.logger.info(f"Model {model_name} saved to {filepath}")
        
        # Save training metadata
        metadata = {
            'best_params': self.best_params,
            'cv_scores': self.cv_scores
        }
        joblib.dump(metadata, f"{filepath_prefix}_metadata.joblib")