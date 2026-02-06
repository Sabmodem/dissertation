from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    """
    Handles loading, splitting, and preprocessing of fraud detection data.
    
    This class encapsulates all data preparation logic, using a vectorizer
    strategy for text preprocessing.
    """
    
    def __init__(
        self,
        vectorizer: 'BaseVectorizer',
        test_size: float = 0.2,
        val_size: float = 0.25,
        random_state: int = 42,
        text_column: str = 'Fillings',
        label_column: str = 'Fraud'
    ):
        """
        Initialize the DataLoader.
        
        Args:
            vectorizer: Instance of BaseVectorizer for text preprocessing
            test_size: Proportion of dataset for test set (0.0 to 1.0)
            val_size: Proportion of remaining data for validation (0.0 to 1.0)
            random_state: Random seed for reproducibility
            text_column: Name of the column containing text data
            label_column: Name of the column containing labels
        """
        self.vectorizer = vectorizer
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.text_column = text_column
        self.label_column = label_column
        
        self.label_encoder = LabelEncoder()
        self.is_loaded = False
        
        # Store original data
        self.df: Optional[pd.DataFrame] = None
        
        # Store split datasets
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        
        # Store vectorized data
        self.X_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
    
    def load_data(self, filepath: str) -> 'DataLoader':
        """
        Load data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            self (for method chaining)
        """
        self.df = pd.read_csv(filepath)
        self.is_loaded = True
        print(f"Loaded dataset with {len(self.df)} samples")
        return self
    
    def split_data(self) -> 'DataLoader':
        """
        Split data into train, validation, and test sets.
        
        Returns:
            self (for method chaining)
            
        Raises:
            RuntimeError: If data hasn't been loaded yet
        """
        if not self.is_loaded:
            raise RuntimeError("Data must be loaded before splitting. Call load_data() first.")
        
        # First split: separate test set
        train_val, self.test_df = train_test_split(
            self.df,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Second split: separate validation from training
        self.train_df, self.val_df = train_test_split(
            train_val,
            test_size=self.val_size,
            random_state=self.random_state
        )
        
        print(f"Split data - Train: {len(self.train_df)}, "
              f"Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        return self
    
    def preprocess(self) -> 'DataLoader':
        """
        Preprocess the data: encode labels and vectorize text.
        
        Returns:
            self (for method chaining)
            
        Raises:
            RuntimeError: If data hasn't been split yet
        """
        if self.train_df is None:
            raise RuntimeError("Data must be split before preprocessing. Call split_data() first.")
        
        # Encode labels
        self.y_train = self.label_encoder.fit_transform(
            self.train_df[self.label_column]
        )
        self.y_val = self.label_encoder.transform(
            self.val_df[self.label_column]
        )
        self.y_test = self.label_encoder.transform(
            self.test_df[self.label_column]
        )
        
        # Vectorize text
        print("Vectorizing text data...")
        self.X_train = self.vectorizer.fit_transform(
            self.train_df[self.text_column]
        )
        self.X_val = self.vectorizer.transform(
            self.val_df[self.text_column]
        )
        self.X_test = self.vectorizer.transform(
            self.test_df[self.text_column]
        )
        
        print(f"Preprocessing complete. Feature dimension: {self.X_train.shape[1]}")
        
        return self
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data.
        
        Returns:
            Tuple of (X_train, y_train)
        """
        self._check_preprocessed()
        return self.X_train, self.y_train
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get validation data.
        
        Returns:
            Tuple of (X_val, y_val)
        """
        self._check_preprocessed()
        return self.X_val, self.y_val
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get test data.
        
        Returns:
            Tuple of (X_test, y_test)
        """
        self._check_preprocessed()
        return self.X_test, self.y_test
    
    def get_all_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get all data splits at once.
        
        Returns:
            Dictionary with keys 'train', 'val', 'test', each containing (X, y) tuple
        """
        self._check_preprocessed()
        return {
            'train': (self.X_train, self.y_train),
            'val': (self.X_val, self.y_val),
            'test': (self.X_test, self.y_test)
        }
    
    def get_label_mapping(self) -> Dict[int, str]:
        """
        Get the mapping from encoded labels to original labels.
        
        Returns:
            Dictionary mapping integers to original label strings
        """
        if not hasattr(self.label_encoder, 'classes_'):
            raise RuntimeError("Labels haven't been encoded yet. Call preprocess() first.")
        
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded and processed data.
        
        Returns:
            Dictionary containing dataset statistics and configuration
        """
        info = {
            'is_loaded': self.is_loaded,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state,
            'text_column': self.text_column,
            'label_column': self.label_column
        }
        
        if self.is_loaded:
            info['total_samples'] = len(self.df)
        
        if self.train_df is not None:
            info['train_samples'] = len(self.train_df)
            info['val_samples'] = len(self.val_df)
            info['test_samples'] = len(self.test_df)
        
        if self.X_train is not None:
            info['feature_dim'] = self.X_train.shape[1]
            info['label_mapping'] = self.get_label_mapping()
        
        return info
    
    def _check_preprocessed(self) -> None:
        """
        Check if data has been preprocessed.
        
        Raises:
            RuntimeError: If data hasn't been preprocessed yet
        """
        if self.X_train is None:
            raise RuntimeError("Data must be preprocessed. Call preprocess() first.")
    
    def __repr__(self) -> str:
        """String representation of the DataLoader."""
        status = "loaded" if self.is_loaded else "not loaded"
        preprocessed = "preprocessed" if self.X_train is not None else "not preprocessed"
        return f"DataLoader({status}, {preprocessed}, vectorizer={self.vectorizer.__class__.__name__})"
    
    def load_and_transform_external(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load external dataset and transform using already-fitted vectorizer.
        Used for evaluating on a different dataset than training.
        """
        if not self.vectorizer.is_fitted:
            raise RuntimeError("Vectorizer must be fitted first")
        
        df_external = pd.read_csv(filepath)
        X = self.vectorizer.transform(df_external[self.text_column])
        y = self.label_encoder.transform(df_external[self.label_column])
        return X, y    