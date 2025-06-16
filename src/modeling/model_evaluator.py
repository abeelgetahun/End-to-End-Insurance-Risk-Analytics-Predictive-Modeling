import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class ModelEvaluator:
    """
    Comprehensive model evaluation for insurance risk analytics.
    
    Provides detailed performance metrics, visualizations, and 
    business-relevant evaluation criteria.
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.evaluation_results = {}
        
    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def evaluate_regression_model(self, model, model_name, X_test, y_test, 
                                y_train=None, X_train=None):
        """
        Comprehensive evaluation for regression models.
        
        Args:
            model: Trained model
            model_name: Name identifier for the model
            X_test, y_test: Test data
            y_train, X_train: Training data (optional, for overfitting check)
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating regression model: {model_name}")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Business-relevant metrics for insurance
        # Loss ratio prediction accuracy
        actual_loss_ratio = np.sum(y_test) / np.sum(y_test)  # This is 1, but for comparison
        predicted_loss_ratio = np.sum(y_pred) / np.sum(y_test)
        loss_ratio_error = abs(actual_loss_ratio - predicted_loss_ratio)
        
        evaluation_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'loss_ratio_error': loss_ratio_error,
            'mean_prediction': np.mean(y_pred),
            'std_prediction': np.std(y_pred),
            'mean_actual': np.mean(y_test),
            'std_actual': np.std(y_test)
        }
        
        # Check for overfitting if training data provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            
            evaluation_metrics['train_r2'] = train_r2
            evaluation_metrics['train_rmse'] = train_rmse
            evaluation_metrics['overfitting_score'] = train_r2 - r2  # Positive indicates overfitting
        
        self.evaluation_results[model_name] = evaluation_metrics
        
        # Log key metrics
        self.logger.info(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
        
        return evaluation_metrics
    
    def evaluate_classification_model(self, model, model_name, X_test, y_test, 
                                    y_train=None, X_train=None):
        """
        Comprehensive evaluation for classification models.
        
        Args:
            model: Trained model
            model_name: Name identifier for the model
            X_test, y_test: Test data
            y_train, X_train: Training data (optional, for overfitting check)
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating classification model: {model_name}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        evaluation_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            evaluation_metrics['roc_auc'] = roc_auc
        
        # Business-relevant metrics for insurance
        # False positive rate (incorrectly predicting claims)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        evaluation_metrics['false_positive_rate'] = fpr
        evaluation_metrics['false_negative_rate'] = fnr
        
        # Check for overfitting if training data provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            evaluation_metrics['train_accuracy'] = train_accuracy
            evaluation_metrics['overfitting_score'] = train_accuracy - accuracy
        
        self.evaluation_results[model_name] = evaluation_metrics
        
        # Log key metrics
        self.logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}")
        
        return evaluation_metrics
    
    def compare_models(self, metric='rmse', problem_type='regression'):
        """
        Compare all evaluated models based on a specific metric.
        
        Args:
            metric: Metric to compare ('rmse', 'r2_score', 'accuracy', etc.)
            problem_type: 'regression' or 'classification'
        
        Returns:
            DataFrame with model comparison
        """
        if not self.evaluation_results:
            self.logger.warning("No models have been evaluated yet.")
            return None
        
        comparison_data = []
        
        for model_name, metrics in self.evaluation_results.items():
            if metric in metrics:
                comparison_data.append({
                    'model': model_name,
                    'metric': metric,
                    'value': metrics[metric]
                })
        
        if not comparison_data:
            self.logger.warning(f"Metric '{metric}' not found in evaluation results.")
            return None
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort based on metric type (lower is better for error metrics, higher for accuracy)
        ascending = metric in ['mse', 'rmse', 'mae', 'mape', 'loss_ratio_error', 'false_positive_rate', 'false_negative_rate']
        comparison_df = comparison_df.sort_values('value', ascending=ascending)
        
        self.logger.info(f"Model comparison based on {metric}:")
        for _, row in comparison_df.iterrows():
            self.logger.info(f"{row['model']}: {row['value']:.4f}")
        
        return comparison_df
    
    def plot_regression_results(self, model, model_name, X_test, y_test, save_path=None):
        """
        Create comprehensive plots for regression model evaluation.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test, y_test: Test data
            save_path: Path to save plots
        """
        y_pred = model.predict(X_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Regression Evaluation', fontsize=16)
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        
        # Add R² to the plot
        r2 = r2_score(y_test, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residuals plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        
        # 3. Residuals histogram
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        
        # 4. Q-Q plot for residuals normality
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/{model_name}_regression_evaluation.png", dpi=300, bbox_inches='tight')
            self.logger.info(f"Regression plots saved to {save_path}/{model_name}_regression_evaluation.png")
        
        plt.show()
    
    def plot_classification_results(self, model, model_name, X_test, y_test, save_path=None):
        """
        Create comprehensive plots for classification model evaluation.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test, y_test: Test data
            save_path: Path to save plots
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Classification Evaluation', fontsize=16)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_title('Confusion Matrix')
        
        # 2. ROC Curve (if probabilities available)
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend(loc="lower right")
        
        # 3. Prediction Distribution
        if y_pred_proba is not None:
            axes[1, 0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='No Claim', color='blue')
            axes[1, 0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Claim', color='red')
            axes[1, 0].set_xlabel('Predicted Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Predicted Probabilities')
            axes[1, 0].legend()
        
        # 4. Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            # Show top 10 features
            top_features = np.argsort(feature_importance)[-10:]
            
            axes[1, 1].barh(range(len(top_features)), feature_importance[top_features])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels([f'Feature_{i}' for i in top_features])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importances')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/{model_name}_classification_evaluation.png", dpi=300, bbox_inches='tight')
            self.logger.info(f"Classification plots saved to {save_path}/{model_name}_classification_evaluation.png")
        
        plt.show()
    
    def generate_evaluation_report(self, save_path=None):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report
        
        Returns:
            Dictionary with formatted report
        """
        if not self.evaluation_results:
            self.logger.warning("No evaluation results available.")
            return None
        
        report = {
            'summary': {},
            'detailed_results': self.evaluation_results,
            'recommendations': []
        }
        
        # Create summary statistics
        for model_name, metrics in self.evaluation_results.items():
            report['summary'][model_name] = {
                'primary_metric': metrics.get('rmse', metrics.get('accuracy', 'N/A')),
                'secondary_metric': metrics.get('r2_score', metrics.get('f1_score', 'N/A'))
            }
        
        # Generate recommendations based on results
        if 'rmse' in list(self.evaluation_results.values())[0]:
            # Regression recommendations
            best_model = min(self.evaluation_results.items(), 
                           key=lambda x: x[1].get('rmse', float('inf')))
            report['recommendations'].append(
                f"Best performing model for claim severity prediction: {best_model[0]} "
                f"(RMSE: {best_model[1]['rmse']:.4f})"
            )
        
        if 'accuracy' in list(self.evaluation_results.values())[0]:
            # Classification recommendations
            best_model = max(self.evaluation_results.items(), 
                           key=lambda x: x[1].get('accuracy', 0))
            report['recommendations'].append(
                f"Best performing model for claim probability: {best_model[0]} "
                f"(Accuracy: {best_model[1]['accuracy']:.4f})"
            )
        
        if save_path:
            import json
            with open(f"{save_path}/evaluation_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Evaluation report saved to {save_path}/evaluation_report.json")
        
        return report