"""
Core abstractions for fraud detection ML pipeline.

This module provides the base classes and utilities for building
modular, reusable machine learning experiments.
"""

from .base_model import BaseModel
from .base_vectorizer import BaseVectorizer
from .data_loader import DataLoader
from .evaluator import Evaluator
from .visualizer import Visualizer
from .experiment import Experiment

__all__ = [
    'BaseModel',
    'BaseVectorizer',
    'DataLoader',
    'Evaluator',
    'Visualizer',
    'Experiment'
]

__version__ = '1.0.0'