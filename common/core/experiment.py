from typing import Dict, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_model import BaseModel
from core.data_loader import DataLoader
from core.evaluator import Evaluator
from core.visualizer import Visualizer
import logging


class Experiment:
    """
    Orchestrates the complete ML experiment pipeline.
    
    This class coordinates data loading, model training, evaluation,
    and visualization in a consistent, reproducible way.
    """
    
    def __init__(
        self,
        name: str,
        model: BaseModel,
        data_loader: DataLoader,
        evaluator: Evaluator,
        visualizer: Visualizer
    ):
        """
        Initialize an experiment.
        
        Args:
            name: Name of the experiment (for logging and file naming)
            model: Model instance implementing BaseModel
            data_loader: DataLoader instance with configured vectorizer
            evaluator: Evaluator instance for computing metrics
            visualizer: Visualizer instance for creating plots
        """
        self.name = name
        self.model = model
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.visualizer = visualizer
        
        # Store results
        self.training_history = None
        self.results = {}
    
    def run(
        self,
        data_file: str,
        evaluate_on_train: bool = False,
        evaluate_on_val: bool = True,
        evaluate_on_test: bool = True,
        generate_plots: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run the complete experiment pipeline.
        
        Args:
            data_file: Path to the dataset CSV file
            evaluate_on_train: Whether to evaluate on training set
            evaluate_on_val: Whether to evaluate on validation set
            evaluate_on_test: Whether to evaluate on test set
            generate_plots: Whether to generate visualization plots
            
        Returns:
            Dictionary of evaluation results for each dataset
        """
        logging.info(f"{'='*30}")
        logging.info(f"EXPERIMENT: {self.name}")
        logging.info(f"{'='*30}")
        
        # Step 1: Load and prepare data
        logging.info("Step 1: Loading and preparing data...")
        self.data_loader.load_data(data_file).split_data().preprocess()
        
        # Step 2: Get data for training
        logging.info("Step 2: Preparing training data...")
        X_train, y_train = self.data_loader.get_train_data()
        X_val, y_val = self.data_loader.get_val_data()
        
        # Step 3: Train model (build happens automatically inside train())
        logging.info("Step 3: Training model...")
        self.training_history = self.model.train(X_train, y_train, X_val, y_val)
        
        # Step 4: Evaluate model
        logging.info("Step 4: Evaluating model...")
        
        if evaluate_on_train:
            logging.info("Evaluating on training set...")
            train_results = self.evaluator.evaluate(
                self.model, X_train, y_train, dataset_name="train"
            )
            self.results['train'] = train_results
            self.evaluator.print_metrics(train_results)
        
        if evaluate_on_val:
            logging.info("Evaluating on validation set...")
            val_results = self.evaluator.evaluate(
                self.model, X_val, y_val, dataset_name="validation"
            )
            self.results['validation'] = val_results
            self.evaluator.print_metrics(val_results)
        
        if evaluate_on_test:
            logging.info("Evaluating on test set...")
            X_test, y_test = self.data_loader.get_test_data()
            test_results = self.evaluator.evaluate(
                self.model, X_test, y_test, dataset_name="test"
            )
            self.results['test'] = test_results
            self.evaluator.print_metrics(test_results)
        
        # Step 5: Generate visualizations
        if generate_plots:
            logging.info("Step 5: Generating visualizations...")
            for dataset_name, metrics in self.results.items():
                self.visualizer.plot_all(metrics, model_name=f"{self.name}_{dataset_name}")
            
            # Plot training history if available
            if self.training_history and len(self.training_history) > 1:
                self.visualizer.plot_training_history(
                    self.training_history,
                    model_name=self.name
                )
        
        logging.info(f"{'='*30}")
        logging.info(f"EXPERIMENT COMPLETE: {self.name}")
        logging.info(f"{'='*30}")
        
        return self.results
    
    def run_training_only(self, data_file: str) -> Dict[str, Any]:
        """
        Run only the data loading and training steps.
        
        Args:
            data_file: Path to the dataset CSV file
            
        Returns:
            Training history dictionary
        """
        logging.info(f"{'='*30}")
        logging.info(f"TRAINING ONLY: {self.name}")
        logging.info(f"{'='*30}")
        
        # Load and prepare data
        self.data_loader.load_data(data_file).split_data().preprocess()
        
        # Build and train model (build happens inside train() for models that need data dimensions)
        X_train, y_train = self.data_loader.get_train_data()
        X_val, y_val = self.data_loader.get_val_data()
        
        self.training_history = self.model.train(X_train, y_train, X_val, y_val)
        
        logging.info(f"Training complete for {self.name}")
        return self.training_history
    
    def run_evaluation_only(
        self,
        dataset_name: str = "test",
        generate_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Run only evaluation on a specific dataset.
        
        Assumes model is already trained and data is loaded.
        
        Args:
            dataset_name: Which dataset to evaluate on ('train', 'validation', or 'test')
            generate_plots: Whether to generate visualization plots
            
        Returns:
            Evaluation results dictionary
        """
        if not self.model.is_trained:
            raise RuntimeError("Model must be trained before evaluation. Call run_training_only() first.")
        
        logging.info(f"{'='*30}")
        logging.info(f"EVALUATION ONLY: {self.name} on {dataset_name}")
        logging.info(f"{'='*30}")
        
        # Get appropriate data
        if dataset_name == "train":
            X, y = self.data_loader.get_train_data()
        elif dataset_name == "validation" or dataset_name == "val":
            X, y = self.data_loader.get_val_data()
        elif dataset_name == "test":
            X, y = self.data_loader.get_test_data()
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        
        # Evaluate
        results = self.evaluator.evaluate(self.model, X, y, dataset_name=dataset_name)
        self.results[dataset_name] = results
        
        self.evaluator.print_metrics(results)
        
        # Generate plots
        if generate_plots:
            self.visualizer.plot_all(results, model_name=f"{self.name}_{dataset_name}")
        
        logging.info(f"Evaluation complete for {self.name} on {dataset_name}")
        return results
    
    def evaluate_on_external_dataset(
        self,
        data_file: str,
        dataset_name: str = "external",
        generate_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on an external dataset.
        
        Args:
            data_file: Path to external dataset CSV file
            dataset_name: Name for this dataset (for logging)
            generate_plots: Whether to generate visualization plots
            
        Returns:
            Evaluation results dictionary
        """
        if not self.model.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        
        logging.info(f"{'='*30}")
        logging.info(f"EXTERNAL EVALUATION: {self.name} on {dataset_name}")
        logging.info(f"{'='*30}")
        
        # Load and transform external data using fitted vectorizer
        X_external, y_external = self.data_loader.load_and_transform_external(data_file)
        
        # Evaluate
        results = self.evaluator.evaluate(
            self.model, X_external, y_external, dataset_name=dataset_name
        )
        self.results[dataset_name] = results
        
        self.evaluator.print_metrics(results)
        
        # Generate plots
        if generate_plots:
            self.visualizer.plot_all(results, model_name=f"{self.name}_{dataset_name}")
        
        logging.info(f"External evaluation complete for {self.name}")
        return results
    
    def get_results_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get a summary of results across all evaluated datasets.
        
        Returns:
            Dictionary mapping dataset names to their key metrics
        """
        summary = {}
        for dataset_name, metrics in self.results.items():
            summary[dataset_name] = self.evaluator.get_summary(metrics)
        return summary
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path where the model should be saved
        """
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model.load_model(filepath)
    
    def __repr__(self) -> str:
        """String representation of the experiment."""
        return (f"Experiment(name='{self.name}', "
                f"model={self.model.__class__.__name__}, "
                f"trained={self.model.is_trained})")