from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import os


class Visualizer:
    """
    Generates visualizations for model evaluation results.
    
    This class provides a consistent interface for creating plots
    from standardized evaluation metrics.
    """
    
    def __init__(
        self,
        save_dir: str = "./plots",
        save_plots: bool = True,
        show_plots: bool = False,
        dpi: int = 100,
        figsize: tuple = (8, 6)
    ):
        """
        Initialize the Visualizer.
        
        Args:
            save_dir: Directory where plots should be saved
            save_plots: Whether to save plots to disk
            show_plots: Whether to display plots interactively
            dpi: Resolution of saved plots
            figsize: Default figure size (width, height) in inches
        """
        self.save_dir = save_dir
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.dpi = dpi
        self.figsize = figsize
        
        # Create save directory if it doesn't exist
        if self.save_plots:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model",
        class_names: List[str] = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            metrics: Metrics dictionary from Evaluator
            model_name: Name of the model (for title and filename)
            class_names: Names of classes (default: ["Non-Fraudulent", "Fraudulent"])
        """
        if class_names is None:
            class_names = ["Non-Fraudulent", "Fraudulent"]
        
        cm = metrics['confusion_matrix']
        dataset = metrics.get('dataset', 'test')
        
        fig, ax = plt.subplots(figsize=self.figsize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name} ({dataset})')
        
        self._save_or_show(fig, f'confusion_matrix_{model_name}_{dataset}')
    
    def plot_roc_curve(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model"
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            metrics: Metrics dictionary from Evaluator
            model_name: Name of the model (for title and filename)
        """
        if metrics.get('roc_curve') is None:
            print(f"Warning: No ROC curve data available for {model_name}")
            return
        
        roc_data = metrics['roc_curve']
        dataset = metrics.get('dataset', 'test')
        
        fig, ax = plt.subplots(figsize=self.figsize)
        roc_display = RocCurveDisplay(
            fpr=roc_data['fpr'],
            tpr=roc_data['tpr'],
            roc_auc=roc_data['auc'],
            estimator_name=model_name
        )
        roc_display.plot(ax=ax)
        ax.set_title(f'ROC Curve - {model_name} ({dataset})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.legend()
        
        self._save_or_show(fig, f'roc_curve_{model_name}_{dataset}')
    
    def plot_precision_recall_curve(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model"
    ) -> None:
        """
        Plot Precision-Recall curve.
        
        Args:
            metrics: Metrics dictionary from Evaluator
            model_name: Name of the model (for title and filename)
        """
        if metrics.get('pr_curve') is None:
            print(f"Warning: No PR curve data available for {model_name}")
            return
        
        pr_data = metrics['pr_curve']
        dataset = metrics.get('dataset', 'test')
        
        fig, ax = plt.subplots(figsize=self.figsize)
        pr_display = PrecisionRecallDisplay(
            precision=pr_data['precision'],
            recall=pr_data['recall']
        )
        pr_display.plot(ax=ax)
        ax.set_title(f'Precision-Recall Curve - {model_name} ({dataset})')
        
        self._save_or_show(fig, f'pr_curve_{model_name}_{dataset}')
    
    def plot_all(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model",
        class_names: List[str] = None
    ) -> None:
        """
        Generate all available plots for a model.
        
        Args:
            metrics: Metrics dictionary from Evaluator
            model_name: Name of the model (for titles and filenames)
            class_names: Names of classes for confusion matrix
        """
        print(f"Generating plots for {model_name}...")
        
        self.plot_confusion_matrix(metrics, model_name, class_names)
        self.plot_roc_curve(metrics, model_name)
        self.plot_precision_recall_curve(metrics, model_name)
        
        print(f"Plots generated for {model_name}")
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        model_name: str = "Model",
        metrics: List[str] = None
    ) -> None:
        """
        Plot training history (for neural network models).
        
        Args:
            history: Training history dictionary with keys like 'loss', 'val_loss', etc.
            model_name: Name of the model (for title and filename)
            metrics: List of metric names to plot (default: ['loss', 'accuracy'])
        """
        if metrics is None:
            metrics = ['loss', 'accuracy']
        
        available_metrics = [m for m in metrics if m in history or f'val_{m}' in history]
        
        if not available_metrics:
            print(f"Warning: No training history data available for {model_name}")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(self.figsize[0] * n_metrics, self.figsize[1]))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available_metrics):
            if metric in history:
                ax.plot(history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history:
                ax.plot(history[f'val_{metric}'], label=f'Validation {metric}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} over Epochs - {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_or_show(fig, f'training_history_{model_name}')
    
    def compare_models_barplot(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        metric: str = 'f1',
        title: str = None
    ) -> None:
        """
        Create a bar plot comparing multiple models on a specific metric.
        
        Args:
            results_dict: Dictionary mapping model names to their evaluation results
            metric: Metric to compare (default: 'f1')
            title: Custom title (default: auto-generated)
        """
        model_names = list(results_dict.keys())
        values = [results_dict[name][metric] for name in model_names]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(model_names, values, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom')
        
        ax.set_ylabel(metric.upper())
        ax.set_title(title or f'Model Comparison - {metric.upper()}')
        ax.set_ylim([0, 1.0])
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        self._save_or_show(fig, f'model_comparison_{metric}')
    
    def compare_models_radar(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        metrics: List[str] = None
    ) -> None:
        """
        Create a radar chart comparing multiple models across multiple metrics.
        
        Args:
            results_dict: Dictionary mapping model names to their evaluation results
            metrics: List of metrics to include (default: ['accuracy', 'precision', 'recall', 'f1'])
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for model_name, results in results_dict.items():
            values = [results[m] for m in metrics]
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        self._save_or_show(fig, 'model_comparison_radar')
    
    def _save_or_show(self, fig: plt.Figure, filename: str) -> None:
        """
        Save and/or show a figure based on configuration.
        
        Args:
            fig: Matplotlib figure to save/show
            filename: Base filename (without extension)
        """
        if self.save_plots:
            filepath = os.path.join(self.save_dir, f'{filename}.png')
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    def set_style(self, style: str = 'default') -> None:
        """
        Set matplotlib style for plots.
        
        Args:
            style: Matplotlib style name (e.g., 'seaborn', 'ggplot', 'default')
        """
        plt.style.use(style)
    
    def __repr__(self) -> str:
        """String representation of the Visualizer."""
        return (f"Visualizer(save_dir='{self.save_dir}', "
                f"save_plots={self.save_plots}, show_plots={self.show_plots})")