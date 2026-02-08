"""
Example: Compare multiple models using the modular framework.

This script demonstrates how easy it is to run experiments with
different models and compare their performance.
"""

from xgboost import XGBClassifier
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from vectorizers.tfidf_vectorizer import TfidfVectorizer
from models.sklearn_model import SklearnModel
import config as config
import os
# from utils import make_experiment
import utils

utils.setup_logging()

def main():
    """Run and compare multiple models."""
    sklearn_estimator = XGBClassifier(
        reg_alpha=0.1,           # L1 regularization term on weight (increase for more regularization)
        reg_lambda=1,            # L2 regularization term on weight
        max_depth=3,             # Maximum depth of a tree (increase to make model more complex)
        min_child_weight=1,      # Minimum sum of instance weight (hessian) needed in a child
        learning_rate=0.3,       # Step size shrinkage used in update to prevents overfitting
    )
    model = SklearnModel(estimator=sklearn_estimator)
    vectorizer=TfidfVectorizer(max_features=config.MAX_FEATURES)
    return utils.make_experiment(model_name='XGBClassifier', model=model, vectorizer=vectorizer)



if __name__ == '__main__':
    main()