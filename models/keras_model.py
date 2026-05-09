from typing import Dict, Any, Optional, List
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_model import BaseModel
import logging

# TensorFlow/Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


class KerasModel(BaseModel):
    """
    Wrapper for Keras/TensorFlow neural network models.
    
    Supports configurable architectures with various activation functions,
    dropout, and training callbacks.
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        layers: List[int] = [256, 128, 64],
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        output_activation: str = 'sigmoid',
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'binary_crossentropy',
        metrics: List[str] = ['accuracy'],
        epochs: int = 20,
        batch_size: int = 32,
        early_stopping_patience: int = 3,
        verbose: int = 1,
        **kwargs
    ):
        """
        Initialize Keras model configuration.
        
        Args:
            input_dim: Input dimension (number of features). Can be set later.
            layers: List of hidden layer sizes
            activation: Activation function for hidden layers ('relu', 'tanh', 'sigmoid')
            dropout_rate: Dropout rate (0.0 = no dropout)
            output_activation: Activation for output layer (default: 'sigmoid' for binary)
            optimizer: Optimizer name ('adam', 'rmsprop', 'sgd')
            learning_rate: Learning rate for optimizer
            loss: Loss function (default: 'binary_crossentropy')
            metrics: List of metrics to track during training
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping (0 = disabled)
            verbose: Verbosity level during training (0=silent, 1=progress bar, 2=one line per epoch)
            **kwargs: Additional configuration
        """
        super().__init__(
            input_dim=input_dim,
            layers=layers,
            activation=activation,
            dropout_rate=dropout_rate,
            output_activation=output_activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            metrics=metrics,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose,
            **kwargs
        )
        
        self.input_dim = input_dim
        self.layers = layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        
        self.history = None
    
    def build_model(self) -> None:
        """
        Build the Keras neural network architecture.
        
        Raises:
            RuntimeError: If input_dim is not set
        """
        if self.input_dim is None:
            raise RuntimeError(
                "input_dim must be set before building model. "
                "Either pass it to __init__ or call set_input_dim() first."
            )
        
        self.model = Sequential()
        
        # Input layer
        self.model.add(Input(shape=(self.input_dim,)))
        
        # Hidden layers
        for layer_size in self.layers:
            self.model.add(Dense(layer_size, activation=self.activation))
            
            # Add dropout if specified
            if self.dropout_rate > 0:
                self.model.add(Dropout(self.dropout_rate))
        
        # Output layer (binary classification)
        self.model.add(Dense(1, activation=self.output_activation))
        
        # Get optimizer
        optimizer = self._get_optimizer()
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=self.metrics
        )
        
        logging.info(f"Model built successfully:")
        logging.info(f"  Architecture: {self.input_dim} -> {' -> '.join(map(str, self.layers))} -> 1")
        logging.info(f"  Activation: {self.activation}, Dropout: {self.dropout_rate}")
        logging.info(f"  Optimizer: {self.optimizer_name}, LR: {self.learning_rate}")
    
    def set_input_dim(self, input_dim: int) -> 'KerasModel':
        """
        Set the input dimension (useful when it's not known at initialization).
        
        Args:
            input_dim: Number of input features
            
        Returns:
            self (for method chaining)
        """
        self.input_dim = input_dim
        return self
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the Keras model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (used for early stopping and monitoring)
            y_val: Validation labels
            
        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            # Auto-detect input dimension and build if not already built
            if self.input_dim is None:
                self.input_dim = X_train.shape[1]
            self.build_model()
        
        logging.info(f"Training Keras model for up to {self.epochs} epochs...")
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping if validation data provided and patience > 0
        if X_val is not None and y_val is not None and self.early_stopping_patience > 0:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        self.is_trained = True
        self.history = history.history
        
        # Print final metrics
        final_epoch = len(history.history['loss'])
        logging.info(f"Training completed after {final_epoch} epochs")
        logging.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        if 'val_loss' in history.history:
            logging.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class labels (0 or 1)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        
        # Get probabilities and threshold at 0.5
        probabilities = self.model.predict(X, verbose=0)
        predictions = (probabilities > 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability estimates. Shape: (n_samples, 2) for binary classification
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        
        # Get probability of positive class
        proba_positive = self.model.predict(X, verbose=0).flatten()
        
        # Create probability matrix [prob_negative, prob_positive]
        proba_negative = 1 - proba_positive
        probabilities = np.vstack([proba_negative, proba_positive]).T
        
        return probabilities
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained Keras model.
        
        Args:
            filepath: Path where the model should be saved (should end with .keras or .keras)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving.")
        
        self.model.save(filepath)
        logging.info(f"Keras model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained Keras model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        from tensorflow.keras.models import load_model as keras_load_model
        
        self.model = keras_load_model(filepath)
        self.is_trained = True
        logging.info(f"Keras model loaded from {filepath}")
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """
        Get the training history.
        
        Returns:
            Dictionary with keys like 'loss', 'val_loss', 'accuracy', etc.
        """
        return self.history
    
    def _get_optimizer(self):
        """
        Get the configured optimizer instance.
        
        Returns:
            Keras optimizer instance
        """
        optimizer_map = {
            'adam': Adam,
            'rmsprop': RMSprop,
            'sgd': SGD
        }
        
        optimizer_class = optimizer_map.get(self.optimizer_name.lower())
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        return optimizer_class(learning_rate=self.learning_rate)
    
    def summary(self) -> None:
        """
        Print a summary of the model architecture.
        """
        if self.model is None:
            logging.info("Model not built yet. Call build_model() first.")
        else:
            self.model.summary()
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"KerasModel(layers={self.layers}, activation={self.activation}, "
                f"trained={self.is_trained})")