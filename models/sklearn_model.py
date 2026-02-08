from typing import Dict, Any, Optional
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_model import BaseModel
import logging


class SklearnModel(BaseModel):
    """
    Wrapper for scikit-learn models that implements BaseModel interface.
    
    This class provides a unified interface for any sklearn classifier,
    making it compatible with the evaluation and visualization pipeline.
    """
    
    def __init__(self, estimator, **kwargs):
        """
        Initialize the sklearn model wrapper.
        
        Args:
            estimator: An sklearn classifier instance (e.g., LogisticRegression())
            **kwargs: Additional configuration (stored but not used for sklearn models)
        """
        super().__init__(**kwargs)
        self.estimator = estimator
        self.model = estimator
    
    def build_model(self) -> None:
        """
        Build/initialize the model.
        
        For sklearn models, the estimator is already initialized in __init__,
        so this method doesn't need to do anything.
        """
        # Sklearn models are already "built" when instantiated
        pass
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the sklearn model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used by most sklearn models)
            y_val: Validation labels (not used by most sklearn models)
            
        Returns:
            Dictionary with training info (empty for basic sklearn models)
        """
        logging.info(f"Training {self.estimator.__class__.__name__}...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training accuracy
        train_score = self.model.score(X_train, y_train)
        
        history = {
            'train_score': train_score
        }
        
        # If validation data provided, calculate validation score
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            history['val_score'] = val_score
            logging.info(f"Training complete. Train score: {train_score:.4f}, Val score: {val_score:.4f}")
        else:
            logging.info(f"Training complete. Train score: {train_score:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability estimates for each class
            
        Raises:
            AttributeError: If the underlying model doesn't support predict_proba
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            # For models like SVM without probability=True
            if hasattr(self.model, 'decision_function'):
                # Use decision function and normalize
                decision = self.model.decision_function(X)
                # Convert to probabilities using sigmoid for binary classification
                from scipy.special import expit
                proba_positive = expit(decision)
                proba_negative = 1 - proba_positive
                return np.vstack([proba_negative, proba_positive]).T
            else:
                raise AttributeError(
                    f"{self.model.__class__.__name__} doesn't support probability predictions. "
                    "For SVM, use SVC(probability=True)"
                )
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model using joblib.
        
        Args:
            filepath: Path where the model should be saved
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving.")
        
        import joblib
        joblib.dump(self.model, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        import joblib
        self.model = joblib.load(filepath)
        self.estimator = self.model
        self.is_trained = True
        logging.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Array of feature importances, or None if not available
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first.")
        
        # For linear models (LogisticRegression, LinearSVC, etc.)
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        
        # For tree-based models (RandomForest, XGBoost, etc.)
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        
        return None
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"SklearnModel({self.estimator.__class__.__name__}, trained={self.is_trained})"