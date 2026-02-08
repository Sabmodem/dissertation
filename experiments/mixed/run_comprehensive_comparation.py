"""
Comprehensive Model Comparison Script.

This script compares all available model types:
1. Classical ML: Logistic Regression, Random Forest, SVM
2. Neural Networks: Keras models with different architectures
3. Transformers: GPT-2, FinBERT (if available)

Demonstrates the power of the modular framework by running
multiple experiments with minimal code.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from core import DataLoader, Evaluator, Visualizer
from vectorizers.tfidf_vectorizer import TfidfVectorizer
from vectorizers.transformer_tokenizer import TransformerTokenizer
from models.sklearn_model import SklearnModel
from models.keras_model import KerasModel
from models.hf_model import HuggingFaceModel
from core.experiment import Experiment
import logging


def run_comprehensive_comparison(
    data_file: str,
    include_transformers: bool = False  # Set to True if you want to include transformers
):
    """
    Run comprehensive comparison of all model types.
    
    Args:
        data_file: Path to the dataset
        include_transformers: Whether to include transformer models (slower)
    """
    
    logging.info("="*30)
    logging.info("COMPREHENSIVE MODEL COMPARISON")
    logging.info("Comparing: Classical ML + Neural Networks" + 
          (" + Transformers" if include_transformers else ""))
    logging.info("="*30)
    
    all_results = {}
    
    # =========================================================================
    # PART 1: CLASSICAL ML MODELS (with TF-IDF)
    # =========================================================================
    
    logging.info("" + "="*30)
    logging.info("PART 1: CLASSICAL MACHINE LEARNING MODELS")
    logging.info("="*30)
    
    classical_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42)
    }
    
    for model_name, sklearn_estimator in classical_models.items():
        logging.info(f"{'-'*30}")
        logging.info(f"Training: {model_name}")
        logging.info(f"{'-'*30}")
        
        # Create fresh components
        vectorizer = TfidfVectorizer(max_features=5000)
        data_loader = DataLoader(vectorizer, test_size=0.2, val_size=0.25, random_state=42)
        model = SklearnModel(estimator=sklearn_estimator)
        evaluator = Evaluator()
        visualizer = Visualizer(
            save_dir="./plots/comprehensive_comparison",
            save_plots=True,
            show_plots=False
        )
        
        # Run experiment
        experiment = Experiment(model_name, model, data_loader, evaluator, visualizer)
        try:
            results = experiment.run(
                data_file,
                evaluate_on_train=False,
                evaluate_on_val=False,  # Only test for comparison
                evaluate_on_test=True,
                generate_plots=False  # Generate plots later
            )
            all_results[model_name] = results['test']
            logging.info(f"✓ {model_name} completed successfully")
        except Exception as e:
            logging.info(f"✗ {model_name} failed: {e}")
    
    # =========================================================================
    # PART 2: NEURAL NETWORKS (with TF-IDF)
    # =========================================================================
    
    logging.info("" + "="*30)
    logging.info("PART 2: NEURAL NETWORKS")
    logging.info("="*30)
    
    nn_configs = {
        "NN_ReLU": {
            "layers": [256, 128, 64],
            "activation": "relu",
            "epochs": 15,
            "verbose": 0
        },
        "NN_Tanh": {
            "layers": [256, 128, 64],
            "activation": "tanh",
            "epochs": 15,
            "verbose": 0
        },
        "NN_Deep": {
            "layers": [512, 256, 128, 64],
            "activation": "relu",
            "dropout_rate": 0.3,
            "epochs": 20,
            "verbose": 0
        }
    }
    
    for model_name, config in nn_configs.items():
        logging.info(f"{'-'*30}")
        logging.info(f"Training: {model_name}")
        logging.info(f"{'-'*30}")
        
        # Create components
        vectorizer = TfidfVectorizer(max_features=5000)
        data_loader = DataLoader(vectorizer, test_size=0.2, val_size=0.25, random_state=42)
        model = KerasModel(
            **config,
            optimizer='adam',
            batch_size=32,
            early_stopping_patience=3
        )
        evaluator = Evaluator()
        visualizer = Visualizer(
            save_dir="./plots/comprehensive_comparison",
            save_plots=True,
            show_plots=False
        )
        
        # Run experiment
        experiment = Experiment(model_name, model, data_loader, evaluator, visualizer)
        try:
            results = experiment.run(
                data_file,
                evaluate_on_train=False,
                evaluate_on_val=False,
                evaluate_on_test=True,
                generate_plots=False
            )
            all_results[model_name] = results['test']
            logging.info(f"✓ {model_name} completed successfully")
        except Exception as e:
            logging.info(f"✗ {model_name} failed: {e}")
    
    # =========================================================================
    # PART 3: TRANSFORMERS (Optional - slower)
    # =========================================================================
    
    if include_transformers:
        logging.info("" + "="*30)
        logging.info("PART 3: TRANSFORMER MODELS")
        logging.info("="*30)
        
        transformer_configs = {
            "GPT-2": {
                "model_name": "gpt2",
                "epochs": 3,
                "batch_size": 4
            }
        }
        
        for model_display_name, config in transformer_configs.items():
            logging.info(f"{'-'*30}")
            logging.info(f"Training: {model_display_name}")
            logging.info(f"{'-'*30}")
            
            try:
                # Create components
                tokenizer = TransformerTokenizer(
                    model_name=config["model_name"],
                    max_length=256  # Shorter for speed
                )
                data_loader = DataLoader(tokenizer, test_size=0.2, val_size=0.25, random_state=42)
                model = HuggingFaceModel(
                    model_name=config["model_name"],
                    num_epochs=config["epochs"],
                    batch_size=config["batch_size"],
                    output_dir=f'./results/{model_display_name.lower().replace("-", "_")}',
                    logging_dir=f'./logs/{model_display_name.lower().replace("-", "_")}'
                )
                evaluator = Evaluator()
                visualizer = Visualizer(
                    save_dir="./plots/comprehensive_comparison",
                    save_plots=True,
                    show_plots=False
                )
                
                # Run experiment
                experiment = Experiment(model_display_name, model, data_loader, evaluator, visualizer)
                results = experiment.run(
                    data_file,
                    evaluate_on_train=False,
                    evaluate_on_val=False,
                    evaluate_on_test=True,
                    generate_plots=False
                )
                all_results[model_display_name] = results['test']
                logging.info(f"✓ {model_display_name} completed successfully")
            except Exception as e:
                logging.info(f"✗ {model_display_name} failed: {e}")
    
    # =========================================================================
    # FINAL COMPARISON AND VISUALIZATION
    # =========================================================================
    
    logging.info("" + "="*30)
    logging.info("FINAL RESULTS")
    logging.info("="*30)
    
    if not all_results:
        logging.info("No results to compare!")
        return
    
    # Print comparison table
    logging.info(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    logging.info("-" * 80)
    
    for model_name, results in sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True):
        logging.info(f"{model_name:<25} "
              f"{results['accuracy']:<12.4f} "
              f"{results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} "
              f"{results['f1']:<12.4f}")
    
    # Find best model
    best_model = max(all_results.items(), key=lambda x: x[1]['f1'])
    logging.info(f"{'='*30}")
    logging.info(f"🏆 BEST MODEL: {best_model[0]} (F1-Score: {best_model[1]['f1']:.4f})")
    logging.info(f"{'='*30}")
    
    # Generate comparison visualizations
    logging.info("Generating comparison visualizations...")
    visualizer = Visualizer(save_dir="./plots/comprehensive_comparison")
    
    visualizer.compare_models_barplot(all_results, metric='accuracy', title='Accuracy Comparison')
    visualizer.compare_models_barplot(all_results, metric='f1', title='F1-Score Comparison')
    visualizer.compare_models_radar(all_results)
    
    # Generate individual plots for top 3 models
    top_3_models = sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
    logging.info(f"Generating detailed plots for top 3 models...")
    for model_name, results in top_3_models:
        visualizer.plot_all(results, model_name=model_name)
    
    logging.info("" + "="*30)
    logging.info("COMPREHENSIVE COMPARISON COMPLETED!")
    logging.info("="*30)
    logging.info(f"Total models compared: {len(all_results)}")
    logging.info(f"Plots saved to: ./plots/comprehensive_comparison/")
    logging.info(f"Summary:")
    logging.info(f"  - Classical ML models: {sum(1 for k in all_results if k in classical_models)}")
    logging.info(f"  - Neural Networks: {sum(1 for k in all_results if 'NN_' in k)}")
    if include_transformers:
        logging.info(f"  - Transformers: {sum(1 for k in all_results if k not in classical_models and 'NN_' not in k)}")
    
    return all_results


if __name__ == "__main__":
    # Get data file
    try:
        import config as config
        DATA_FILE = config.DATA_FILE
    except ImportError:
        DATA_FILE = "fraud_data.csv"
        logging.info(f"Warning: config.py not found, using default: {DATA_FILE}")
    
    # Run comprehensive comparison
    # Set include_transformers=True to include transformer models (slower)
    results = run_comprehensive_comparison(
        DATA_FILE,
        include_transformers=False  # Change to True to include transformers
    )