"""
Example: Compare multiple models using the modular framework.

This script demonstrates how easy it is to run experiments with
different models and compare their performance.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core import Experiment, DataLoader, Evaluator, Visualizer
from vectorizers.tfidf_vectorizer import TfidfVectorizer
from models.sklearn_model import SklearnModel


def run_model_comparison():
    """Run and compare multiple models."""
    
    # Configuration
    try:
        import config
        DATA_FILE = config.DATA_FILE
    except ImportError:
        DATA_FILE = "fraud_data.csv"
    
    print("="*70)
    print("MULTI-MODEL COMPARISON EXPERIMENT")
    print("="*70)
    
    # Define models to compare
    models_config = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42)
    }
    
    # Shared components
    vectorizer = TfidfVectorizer(max_features=5000)
    evaluator = Evaluator(pos_label=1)
    visualizer = Visualizer(
        save_dir="../plots/comparison",
        save_plots=True,
        show_plots=False
    )
    
    # Store results for comparison
    all_results = {}
    test_results_for_comparison = {}
    
    # Run experiment for each model
    for model_name, sklearn_estimator in models_config.items():
        print(f"\n{'='*70}")
        print(f"Running experiment: {model_name}")
        print(f"{'='*70}")
        
        # Create fresh data loader for each model (to ensure independence)
        data_loader = DataLoader(
            vectorizer=TfidfVectorizer(max_features=5000),  # Fresh vectorizer
            test_size=0.2,
            val_size=0.25,
            random_state=42
        )
        
        # Create model wrapper
        model = SklearnModel(estimator=sklearn_estimator)
        
        # Create and run experiment
        experiment = Experiment(
            name=model_name,
            model=model,
            data_loader=data_loader,
            evaluator=evaluator,
            visualizer=visualizer
        )
        
        results = experiment.run(
            data_file=DATA_FILE,
            evaluate_on_train=False,
            evaluate_on_val=True,
            evaluate_on_test=True,
            generate_plots=True
        )
        
        all_results[model_name] = results
        test_results_for_comparison[model_name] = results['test']
    
    # Compare models
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    # Print comparison table
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    for model_name, results in test_results_for_comparison.items():
        print(f"{model_name:<20} "
              f"{results['accuracy']:<12.4f} "
              f"{results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} "
              f"{results['f1']:<12.4f}")
    
    # Generate comparison visualizations
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    # Bar chart comparison
    visualizer.compare_models_barplot(
        test_results_for_comparison,
        metric='f1',
        title='F1-Score Comparison Across Models'
    )
    
    visualizer.compare_models_barplot(
        test_results_for_comparison,
        metric='accuracy',
        title='Accuracy Comparison Across Models'
    )
    
    # Radar chart comparison
    visualizer.compare_models_radar(
        test_results_for_comparison,
        metrics=['accuracy', 'precision', 'recall', 'f1']
    )
    
    # Detailed pairwise comparisons
    model_names = list(test_results_for_comparison.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            print(f"\nComparing {model1} vs {model2}:")
            evaluator.compare_metrics(
                test_results_for_comparison[model1],
                test_results_for_comparison[model2],
                model1_name=model1,
                model2_name=model2
            )
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETED!")
    print(f"{'='*70}")
    print(f"\nAll plots saved to: {visualizer.save_dir}")
    
    return all_results


if __name__ == "__main__":
    results = run_model_comparison()