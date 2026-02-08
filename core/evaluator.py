from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import logging

class Evaluator:
    """
    Evaluates model performance and computes standardized metrics.
    
    This class provides a consistent interface for evaluating any model,
    regardless of its underlying implementation.
    """
    
    def __init__(self, pos_label: int = 1):
        """
        Initialize the Evaluator.
        
        Args:
            pos_label: The label of the positive class (default: 1 for binary classification)
        """
        self.pos_label = pos_label
    
    def evaluate(
        self,
        model: 'BaseModel',
        X: np.ndarray,
        y_true: np.ndarray,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Evaluate a model on given data.
        
        Args:
            model: Trained model implementing BaseModel interface
            X: Features to evaluate on
            y_true: True labels
            dataset_name: Name of the dataset (for logging/display purposes)
            
        Returns:
            Dictionary containing all evaluation metrics and predictions
        """
        # Get predictions
        y_pred = model.predict(X)
        
        # Try to get probabilities
        try:
            y_proba = model.predict_proba(X)
            has_proba = True
        except (NotImplementedError, AttributeError):
            y_proba = None
            has_proba = False
        
        # Compute basic metrics
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0),
            'recall': recall_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0),
            'f1': f1_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0),
        }
        
        # Store raw predictions for plotting
        metrics['y_true'] = y_true
        metrics['y_pred'] = y_pred
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['tn'] = cm[0, 0] if cm.shape == (2, 2) else None
        metrics['fp'] = cm[0, 1] if cm.shape == (2, 2) else None
        metrics['fn'] = cm[1, 0] if cm.shape == (2, 2) else None
        metrics['tp'] = cm[1, 1] if cm.shape == (2, 2) else None
        
        # Compute ROC and PR curves if probabilities are available
        if has_proba:
            # For binary classification, use probability of positive class
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_score = y_proba[:, 1]
            else:
                y_score = y_proba
            
            metrics['y_proba'] = y_proba
            metrics['y_score'] = y_score
            
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_score, pos_label=self.pos_label)
            metrics['roc_curve'] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds,
                'auc': auc(fpr, tpr)
            }
            
            # Precision-Recall curve
            precision, recall, pr_thresholds = precision_recall_curve(
                y_true, y_score, pos_label=self.pos_label
            )
            metrics['pr_curve'] = {
                'precision': precision,
                'recall': recall,
                'thresholds': pr_thresholds
            }
        else:
            metrics['y_proba'] = None
            metrics['y_score'] = None
            metrics['roc_curve'] = None
            metrics['pr_curve'] = None
        
        return metrics
    
    def evaluate_all_splits(
        self,
        model: 'BaseModel',
        data_dict: Dict[str, tuple]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model on multiple data splits.
        
        Args:
            model: Trained model implementing BaseModel interface
            data_dict: Dictionary with keys like 'train', 'val', 'test',
                      each containing (X, y) tuple
                      
        Returns:
            Dictionary mapping split names to their evaluation results
        """
        results = {}
        for split_name, (X, y) in data_dict.items():
            results[split_name] = self.evaluate(model, X, y, dataset_name=split_name)
        return results
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics from evaluate()
        """
        dataset = metrics.get('dataset', 'unknown')
        logging.info(f"{'='*30}")
        logging.info(f"{dataset.upper()} SET RESULTS")
        logging.info(f"{'='*30}")
        logging.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall:    {metrics['recall']:.4f}")
        logging.info(f"F1-score:  {metrics['f1']:.4f}")
        
        if metrics.get('roc_curve') is not None:
            logging.info(f"ROC AUC:   {metrics['roc_curve']['auc']:.4f}")
        
        # Print confusion matrix
        if metrics.get('tp') is not None:
            logging.info(f"Confusion Matrix:")
            logging.info(f"  TN: {metrics['tn']}, FP: {metrics['fp']}")
            logging.info(f"  FN: {metrics['fn']}, TP: {metrics['tp']}")
        logging.info(f"{'='*30}")
    
    def print_all_metrics(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Print metrics for all evaluated splits.
        
        Args:
            results: Dictionary of results from evaluate_all_splits()
        """
        for split_name, metrics in results.items():
            self.print_metrics(metrics)
    
    def compare_metrics(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> None:
        """
        Compare metrics between two models.
        
        Args:
            results1: Evaluation results for first model
            results2: Evaluation results for second model
            model1_name: Name of first model (for display)
            model2_name: Name of second model (for display)
        """
        logging.info(f"{'='*30}")
        logging.info(f"MODEL COMPARISON: {model1_name} vs {model2_name}")
        logging.info(f"{'='*30}")
        logging.info(f"{'Metric':<15} {model1_name:<20} {model2_name:<20}")
        logging.info(f"{'-'*30}")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            val1 = results1[metric]
            val2 = results2[metric]
            diff = val2 - val1
            diff_symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            logging.info(f"{metric.capitalize():<15} {val1:.4f}               {val2:.4f}  {diff_symbol} ({diff:+.4f})")
        
        if results1.get('roc_curve') and results2.get('roc_curve'):
            auc1 = results1['roc_curve']['auc']
            auc2 = results2['roc_curve']['auc']
            diff = auc2 - auc1
            diff_symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            logging.info(f"{'ROC AUC':<15} {auc1:.4f}               {auc2:.4f}  {diff_symbol} ({diff:+.4f})")
        
        logging.info(f"{'='*30}")
    
    def get_summary(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Get a summary of key metrics.
        
        Args:
            metrics: Dictionary of metrics from evaluate()
            
        Returns:
            Dictionary with key performance metrics only
        """
        summary = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }
        
        if metrics.get('roc_curve') is not None:
            summary['roc_auc'] = metrics['roc_curve']['auc']
        
        return summary