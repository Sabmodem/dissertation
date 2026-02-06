"""
Keras Neural Network Experiment using the modular framework.

This script demonstrates how to train a neural network using
the Keras model wrapper.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core import DataLoader, Evaluator, Visualizer, Experiment
from vectorizers.tfidf_vectorizer import TfidfVectorizer
from models.keras_model import KerasModel

def main():
    """Run the Keras Neural Network experiment."""
    
    # Configuration
    try:
        import config
        DATA_FILE = config.DATA_FILE
    except ImportError:
        DATA_FILE = "fraud_data.csv"
    
    print("="*70)
    print("KERAS NEURAL NETWORK FRAUD DETECTION EXPERIMENT")
    print("="*70)
    
    # Initialize components
    print("\n1. Initializing components...")
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    print(f"   ✓ Vectorizer: {vectorizer}")
    
    # Create data loader
    data_loader = DataLoader(
        vectorizer=vectorizer,
        test_size=0.2,
        val_size=0.25,
        random_state=42
    )
    print(f"   ✓ Data Loader: {data_loader}")
    
    # Create Keras model with tanh activation (like your original script)
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
    print(f"   ✓ Model: {model}")
    
    # Create evaluator
    evaluator = Evaluator(pos_label=1)
    print(f"   ✓ Evaluator: {evaluator}")
    
    # Create visualizer
    visualizer = Visualizer(
        save_dir="./plots/keras_nn",
        save_plots=True,
        show_plots=False
    )
    print(f"   ✓ Visualizer: {visualizer}")
    
    # Create experiment
    print("\n2. Creating experiment...")
    experiment = Experiment(
        name="KerasNN_Tanh",
        model=model,
        data_loader=data_loader,
        evaluator=evaluator,
        visualizer=visualizer
    )
    
    # Run experiment
    print("\n3. Running experiment...")
    results = experiment.run(
        data_file=DATA_FILE,
        evaluate_on_train=False,
        evaluate_on_val=True,
        evaluate_on_test=True,
        generate_plots=True
    )
    
    # Display summary
    print("\n4. Results Summary:")
    print("="*70)
    summary = experiment.get_results_summary()
    
    for dataset_name, metrics in summary.items():
        print(f"\n{dataset_name.upper()} SET:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:12s}: {value:.4f}")
    
    # Plot training history
    print("\n5. Generating training history plot...")
    history = model.get_training_history()
    if history:
        visualizer.plot_training_history(
            history,
            model_name="KerasNN_Tanh",
            metrics=['loss', 'accuracy']
        )
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED!")
    print("="*70)
    print(f"\nPlots saved to: {visualizer.save_dir}")
    
    return experiment, results


def compare_activations():
    """Compare different activation functions."""
    
    try:
        import config
        DATA_FILE = config.DATA_FILE
    except ImportError:
        DATA_FILE = "fraud_data.csv"
    
    print("="*70)
    print("COMPARING ACTIVATION FUNCTIONS")
    print("="*70)
    
    activations = ['relu', 'tanh', 'sigmoid']
    all_results = {}
    
    for activation in activations:
        print(f"\n{'='*70}")
        print(f"Testing {activation.upper()} activation")
        print(f"{'='*70}")
        
        # Create fresh components for each model
        vectorizer = TfidfVectorizer(max_features=5000)
        data_loader = DataLoader(vectorizer, test_size=0.2, val_size=0.25)
        
        model = KerasModel(
            layers=[256, 128, 64],
            activation=activation,
            optimizer='rmsprop',
            epochs=20,
            batch_size=32,
            early_stopping_patience=3,
            verbose=0  # Quiet mode for comparison
        )
        
        evaluator = Evaluator()
        visualizer = Visualizer(
            save_dir=f"./plots/keras_comparison",
            save_plots=True,
            show_plots=False
        )
        
        experiment = Experiment(
            f"KerasNN_{activation.capitalize()}",
            model, data_loader, evaluator, visualizer
        )
        
        results = experiment.run(DATA_FILE, evaluate_on_val=False)
        all_results[f"Keras_{activation}"] = results['test']
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Activation':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        activation = model_name.split('_')[1]
        print(f"{activation:<15} "
              f"{results['accuracy']:<12.4f} "
              f"{results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} "
              f"{results['f1']:<12.4f}")
    
    # Generate comparison plots
    visualizer = Visualizer(save_dir="./plots/keras_comparison")
    visualizer.compare_models_barplot(all_results, metric='f1')
    visualizer.compare_models_radar(all_results)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    # Run single experiment
    experiment, results = main()
    
    # Uncomment to compare activations
    # compare_activations()