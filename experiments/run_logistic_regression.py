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

from core.data_loader import DataLoader
from core.evaluator import Evaluator
from core.visualizer import Visualizer
from vectorizers.tfidf_vectorizer import TfidfVectorizer
from models.sklearn_model import SklearnModel
from core.experiment import Experiment


def main():
    """Run the Logistic Regression experiment."""
    
    # Configuration
    DATA_FILE = "config.DATA_FILE"  # Replace with actual path or use config
    EXPERIMENT_NAME = "LogisticRegression"
    
    # If using config file:
    try:
        import config
        DATA_FILE = config.DATA_FILE
    except ImportError:
        print("Warning: config.py not found. Using default path.")
        DATA_FILE = "fraud_data.csv"
    
    print("="*70)
    print("LOGISTIC REGRESSION FRAUD DETECTION EXPERIMENT")
    print("="*70)
    
    # Step 1: Initialize components
    print("\n1. Initializing components...")
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(max_features=None)  # Use all features
    print(f"   ✓ Vectorizer: {vectorizer}")
    
    # Create data loader
    data_loader = DataLoader(
        vectorizer=vectorizer,
        test_size=0.2,
        val_size=0.25,  # 25% of remaining 80% = 20% of total
        random_state=42
    )
    print(f"   ✓ Data Loader: {data_loader}")
    
    # Create model
    sklearn_estimator = LogisticRegression(max_iter=1000, random_state=42)
    model = SklearnModel(estimator=sklearn_estimator)
    print(f"   ✓ Model: {model}")
    
    # Create evaluator
    evaluator = Evaluator(pos_label=1)  # Assuming 'yes' fraud is encoded as 1
    print(f"   ✓ Evaluator: {evaluator}")
    
    # Create visualizer
    visualizer = Visualizer(
        save_dir="../plots/logistic_regression",
        save_plots=True,
        show_plots=False,  # Set to True if you want to see plots interactively
        dpi=100
    )
    print(f"   ✓ Visualizer: {visualizer}")
    
    # Step 2: Create experiment
    print("\n2. Creating experiment...")
    experiment = Experiment(
        name=EXPERIMENT_NAME,
        model=model,
        data_loader=data_loader,
        evaluator=evaluator,
        visualizer=visualizer
    )
    print(f"   ✓ Experiment: {experiment}")
    
    # Step 3: Run the experiment
    print("\n3. Running complete experiment pipeline...")
    results = experiment.run(
        data_file=DATA_FILE,
        evaluate_on_train=False,  # Set to True if you want to see training set performance
        evaluate_on_val=True,
        evaluate_on_test=True,
        generate_plots=True
    )
    
    # Step 4: Display summary
    print("\n4. Results Summary:")
    print("="*70)
    summary = experiment.get_results_summary()
    
    for dataset_name, metrics in summary.items():
        print(f"\n{dataset_name.upper()} SET:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:12s}: {value:.4f}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nPlots saved to: {visualizer.save_dir}")
    print(f"Total datasets evaluated: {len(results)}")
    
    # Optional: Save the model
    # experiment.save_model("./models/logistic_regression.pkl")
    
    return experiment, results


if __name__ == "__main__":
    experiment, results = main()
    
    # You can now access the experiment and results for further analysis
    # For example:
    # - experiment.model.get_feature_importance()
    # - results['test']['confusion_matrix']
    # - experiment.evaluate_on_external_dataset('another_dataset.csv')