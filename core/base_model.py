from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    This class defines the interface that all models must implement,
    regardless of their underlying framework (sklearn, Keras, PyTorch, etc.)
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the model with hyperparameters.
        
        Args:
            **kwargs: Model-specific hyperparameters
        """
        self.model = None
        self.is_trained = False
        self.config = kwargs
    
    @abstractmethod
    def build_model(self) -> None:
        """
        Build/initialize the model architecture.
        
        This method should create the model object but not train it.
        For sklearn models, this might just instantiate the classifier.
        For neural networks, this defines the architecture.
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dict containing training history/metrics (format varies by model)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the given data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class labels (0 or 1)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the given data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability estimates for each class. Shape: (n_samples, n_classes)
            For binary classification: [:, 1] gives probability of positive class
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Dictionary of model hyperparameters
        """
        return self.config
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where the model should be saved
            
        Note:
            Subclasses can override this for framework-specific saving
        """
        raise NotImplementedError("Model saving not implemented for this model type")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Note:
            Subclasses can override this for framework-specific loading
        """
        raise NotImplementedError("Model loading not implemented for this model type")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(config={self.config})"