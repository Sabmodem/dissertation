from typing import Dict, Any, Optional
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_model import BaseModel

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class HuggingFaceModel(BaseModel):
    """
    Wrapper for HuggingFace transformer models.
    
    Supports any HuggingFace model for sequence classification
    (BERT, RoBERTa, DistilBERT, GPT-2, FinBERT, etc.)
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_labels: int = 2,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        eval_strategy: str = 'epoch',
        save_strategy: str = 'epoch',
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = 'f1',
        output_dir: str = './results',
        logging_dir: str = './logs',
        logging_steps: int = 10,
        save_total_limit: int = 2,
        **kwargs
    ):
        """
        Initialize HuggingFace model configuration.
        
        Args:
            model_name: Name of pretrained model (e.g., 'bert-base-uncased', 'gpt2', 'yiyanghkust/finbert-pretrain')
            num_labels: Number of output labels (2 for binary classification)
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for regularization
            eval_strategy: Evaluation strategy ('no', 'steps', 'epoch')
            save_strategy: Model saving strategy ('no', 'steps', 'epoch')
            load_best_model_at_end: Whether to load best model at end
            metric_for_best_model: Metric to determine best model
            output_dir: Directory for model outputs
            logging_dir: Directory for logs
            logging_steps: Log every N steps
            save_total_limit: Maximum number of checkpoints to keep
            **kwargs: Additional configuration
        """
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            output_dir=output_dir,
            logging_dir=logging_dir,
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
            **kwargs
        )
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.eval_strategy = eval_strategy
        self.save_strategy = save_strategy
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.save_total_limit = save_total_limit
        
        self.tokenizer = None
        self.trainer = None
        self.training_results = None
    
    def build_model(self) -> None:
        """
        Build the HuggingFace transformer model.
        """
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Handle models without pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Resize embeddings if we added tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"Model loaded successfully: {self.model_name}")
        print(f"  Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the HuggingFace model.
        
        Note: X_train and X_val should be tokenized input IDs (from TransformerTokenizer).
        
        Args:
            X_train: Training input IDs (tokenized text)
            y_train: Training labels
            X_val: Validation input IDs (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training history and metrics
        """
        if self.model is None:
            self.build_model()
        
        print(f"Training {self.model_name} for {self.num_epochs} epochs...")
        
        # Create datasets
        train_dataset = HFDataset(X_train, y_train)
        val_dataset = HFDataset(X_val, y_val) if X_val is not None else None
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            eval_strategy=self.eval_strategy,
            save_strategy=self.save_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            save_total_limit=self.save_total_limit,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        
        # Train
        train_result = self.trainer.train()
        
        self.is_trained = True
        self.training_results = train_result
        
        # Get final metrics
        print(f"\nTraining completed!")
        print(f"Training loss: {train_result.training_loss:.4f}")
        
        # Evaluate on validation set if provided
        if val_dataset is not None:
            val_results = self.trainer.evaluate(eval_dataset=val_dataset)
            print(f"\nValidation Results:")
            for key, value in val_results.items():
                print(f"  {key}: {value:.4f}")
            
            return {
                'train': train_result.metrics,
                'validation': val_results
            }
        
        return {'train': train_result.metrics}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Tokenized input IDs
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        
        # Create dataset
        dataset = HFDataset(X, np.zeros(len(X)))  # Dummy labels
        
        # Get predictions
        predictions = self.trainer.predict(dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        
        return pred_labels
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Tokenized input IDs
            
        Returns:
            Probability estimates. Shape: (n_samples, num_labels)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        
        # Create dataset
        dataset = HFDataset(X, np.zeros(len(X)))  # Dummy labels
        
        # Get predictions (logits)
        predictions = self.trainer.predict(dataset)
        logits = predictions.predictions
        
        # Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        
        return probabilities
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained HuggingFace model.
        
        Args:
            filepath: Directory path where the model should be saved
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving.")
        
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        print(f"HuggingFace model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained HuggingFace model from disk.
        
        Args:
            filepath: Directory path to the saved model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(filepath)
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)
        self.is_trained = True
        print(f"HuggingFace model loaded from {filepath}")
    
    def _compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation (used by Trainer).
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of computed metrics
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"HuggingFaceModel(model={self.model_name}, "
                f"trained={self.is_trained})")


class HFDataset(Dataset):
    """
    PyTorch Dataset for HuggingFace models.
    
    Wraps tokenized inputs and labels for use with HuggingFace Trainer.
    """
    
    def __init__(self, input_ids: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            input_ids: Tokenized input IDs
            labels: Labels for the data
        """
        self.input_ids = input_ids
        self.labels = labels
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with input_ids and labels as tensors
        """
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }