"""
Example: Compare multiple models using the modular framework.

This script demonstrates how easy it is to run experiments with
different models and compare their performance.
"""

from sklearn.ensemble import RandomForestClassifier
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from vectorizers.tfidf_vectorizer import TfidfVectorizer
from models.sklearn_model import SklearnModel
import config as config
import os
import utils

utils.setup_logging()

def main():
    """Run and compare multiple models."""
    sklearn_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    model = SklearnModel(estimator=sklearn_estimator)
    vectorizer=TfidfVectorizer(max_features=config.MAX_FEATURES)
    return utils.make_experiment(model_name='RandomForest', model=model, vectorizer=vectorizer)


if __name__ == '__main__':
    main()