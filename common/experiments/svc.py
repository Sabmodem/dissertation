"""
Example: Compare multiple models using the modular framework.

This script demonstrates how easy it is to run experiments with
different models and compare their performance.
"""

from sklearn.svm import SVC
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from vectorizers.tfidf_vectorizer import TfidfVectorizer
from models.sklearn_model import SklearnModel
import structuring.common.config as config
import os
from structuring.common.utils import make_experiment

def main():
    """Run and compare multiple models."""
    sklearn_estimator = SVC(kernel='linear')
    model = SklearnModel(estimator=sklearn_estimator)
    vectorizer=TfidfVectorizer(max_features=config.MAX_FEATURES)
    return make_experiment(model_name='RandomForest', model=model, vectorizer=vectorizer)


if __name__ == '__main__':
    main()