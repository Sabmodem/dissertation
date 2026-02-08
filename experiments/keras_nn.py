"""
Keras Neural Network Experiment using the modular framework.

This script demonstrates how to train a neural network using
the Keras model wrapper.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from vectorizers.tfidf_vectorizer import TfidfVectorizer
from models.keras_model import KerasModel
import config as config
import utils

utils.setup_logging()

def main():
    """Run the Keras Neural Network experiment."""
    # Create Keras model with tanh activation like in original script
    model = KerasModel(
        layers=[256, 128, 64],
        activation='tanh',
        dropout_rate=0.0,
        optimizer='rmsprop',
        learning_rate=0.001,
        epochs=20,
        batch_size=32,
        early_stopping_patience=3,
        verbose=1
    )
    vectorizer = TfidfVectorizer(max_features=config.MAX_FEATURES)
    return utils.make_experiment(model_name='KerasNeuralNetwork', model=model, vectorizer=vectorizer)

if __name__ == '__main__':
    main()