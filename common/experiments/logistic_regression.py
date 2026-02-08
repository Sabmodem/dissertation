"""
Logistic Regression Experiment using the new modular framework.

This script demonstrates how to use the core abstractions to run
a complete ML experiment with minimal code.
"""

from sklearn.linear_model import LogisticRegression
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from vectorizers.tfidf_vectorizer import TfidfVectorizer
from models.sklearn_model import SklearnModel
import config as config
import utils

utils.setup_logging()

def main():
    """Run the Logistic Regression experiment."""      
    vectorizer=TfidfVectorizer(max_features=config.MAX_FEATURES)
    sklearn_estimator = LogisticRegression(max_iter=1000, random_state=42)
    model = SklearnModel(estimator=sklearn_estimator)
    return utils.make_experiment(model_name='LogisticRegression', model=model, vectorizer=vectorizer)

if __name__ == "__main__":
    main()