from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_vectorizer import BaseVectorizer


class TfidfVectorizer(BaseVectorizer):
    """
    TF-IDF vectorizer wrapper that implements BaseVectorizer interface.
    
    Wraps sklearn's TfidfVectorizer to provide consistent interface
    with other vectorization methods.
    """
    
    def __init__(self, max_features: int = None, **kwargs):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            **kwargs: Additional arguments passed to sklearn's TfidfVectorizer
        """
        super().__init__(max_features=max_features, **kwargs)
        self.max_features = max_features
        self.vectorizer = SklearnTfidfVectorizer(max_features=max_features, **kwargs)
    
    def fit(self, texts: Union[List[str], pd.Series]) -> 'TfidfVectorizer':
        """
        Fit the TF-IDF vectorizer on the provided texts.
        
        Args:
            texts: List or Series of text documents
            
        Returns:
            self (for method chaining)
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Transform texts into TF-IDF feature vectors.
        
        Args:
            texts: List or Series of text documents
            
        Returns:
            Numpy array of TF-IDF features, shape: (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transform. Call fit() first.")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Convert sparse matrix to dense array
        tfidf_matrix = self.vectorizer.transform(texts)
        return tfidf_matrix.toarray()
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary) learned by the vectorizer.
        
        Returns:
            List of feature names (words in vocabulary)
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before getting feature names.")
        
        # Handle different sklearn versions
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            return self.vectorizer.get_feature_names_out().tolist()
        else:
            return self.vectorizer.get_feature_names()
    
    def _get_output_dim(self) -> int:
        """
        Get the output dimensionality (number of features).
        
        Returns:
            Number of features in the TF-IDF representation
        """
        return len(self.vectorizer.vocabulary_)
    
    def get_vocabulary(self) -> dict:
        """
        Get the vocabulary mapping from terms to feature indices.
        
        Returns:
            Dictionary mapping terms to their feature indices
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before getting vocabulary.")
        
        return self.vectorizer.vocabulary_
    
    def get_idf_values(self) -> np.ndarray:
        """
        Get the IDF (Inverse Document Frequency) values.
        
        Returns:
            Array of IDF values for each feature
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before getting IDF values.")
        
        return self.vectorizer.idf_