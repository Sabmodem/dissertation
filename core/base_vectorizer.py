from abc import ABC, abstractmethod
from typing import Any, List, Union
import numpy as np
import pandas as pd


class BaseVectorizer(ABC):
    """
    Abstract base class for all text vectorization strategies.
    
    This class defines a common interface for different vectorization methods
    (TF-IDF, transformer tokenizers, etc.), allowing them to be used interchangeably.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the vectorizer with configuration parameters.
        
        Args:
            **kwargs: Vectorizer-specific configuration
        """
        self.vectorizer = None
        self.is_fitted = False
        self.config = kwargs
    
    @abstractmethod
    def fit(self, texts: Union[List[str], pd.Series]) -> 'BaseVectorizer':
        """
        Fit the vectorizer on the provided texts.
        
        This learns the vocabulary, token mappings, or any other
        necessary information from the training data.
        
        Args:
            texts: List or Series of text documents
            
        Returns:
            self (for method chaining)
        """
        pass
    
    @abstractmethod
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Transform texts into numerical representations.
        
        Args:
            texts: List or Series of text documents
            
        Returns:
            Numpy array of vectorized texts. Shape depends on vectorization method.
            For TF-IDF: (n_samples, n_features)
            For transformers: (n_samples, max_length) or (n_samples, max_length, embedding_dim)
        """
        pass
    
    def fit_transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Fit the vectorizer and transform texts in one step.
        
        Args:
            texts: List or Series of text documents
            
        Returns:
            Numpy array of vectorized texts
        """
        self.fit(texts)
        return self.transform(texts)
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary) learned by the vectorizer.
        
        Returns:
            List of feature names. For TF-IDF, these are words.
            For transformers, might return token IDs or empty list.
        """
        pass
    
    def get_config(self) -> dict:
        """
        Get the vectorizer configuration.
        
        Returns:
            Dictionary of vectorizer parameters
        """
        return self.config
    
    def get_output_dim(self) -> int:
        """
        Get the output dimensionality of the vectorizer.
        
        Returns:
            Number of features in the output
            
        Raises:
            RuntimeError: If vectorizer hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before getting output dimension")
        return self._get_output_dim()
    
    @abstractmethod
    def _get_output_dim(self) -> int:
        """
        Internal method to get output dimension.
        Subclasses implement this based on their vectorization method.
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the vectorizer."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_status}, config={self.config})"