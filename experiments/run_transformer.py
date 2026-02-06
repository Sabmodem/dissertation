"""
Transformer Model Experiment using the modular framework.

This script demonstrates how to train transformer models (GPT-2, FinBERT, etc.)
using the HuggingFace model wrapper and TransformerTokenizer.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core import DataLoader, Experiment, Evaluator, Visualizer
from vectorizers.transformer_tokenizer import TransformerTokenizer
from models.hf_model import HuggingFaceModel

def run_gpt2_experiment():
    """Run GPT-2 model experiment."""
    
    try:
        import config
        DATA_FILE = config.DATA_FILE
    except ImportError:
        DATA_FILE = "fraud_data.csv"
    
    print("="*70)
    print("GPT-2 FRAUD DETECTION EXPERIMENT")
    print("="*70)
    
    # Initialize components
    print("\n1. Initializing components...")
    
    # Create tokenizer for GPT-2
    tokenizer = TransformerTokenizer(
        model_name='gpt2',
        max_length=512,
        padding='max_length',
        truncation=True
    )
    print(f"   ✓ Tokenizer: {tokenizer}")
    
    # Create data loader
    data_loader = DataLoader(
        vectorizer=tokenizer,
        test_size=0.2,
        val_size=0.25,
        random_state=42
    )
    print(f"   ✓ Data Loader: {data_loader}")
    
    # Create HuggingFace GPT-2 model
    model = HuggingFaceModel(
        model_name='gpt2',
        num_labels=2,
        num_epochs=4,
        batch_size=8,
        learning_rate=5e-5,
        warmup_steps=100,
        output_dir='./results/gpt2',
        logging_dir='./logs/gpt2'
    )
    print(f"   ✓ Model: {model}")
    
    # Create evaluator
    evaluator = Evaluator(pos_label=1)
    print(f"   ✓ Evaluator: {evaluator}")
    
    # Create visualizer
    visualizer = Visualizer(
        save_dir="./plots/gpt2",
        save_plots=True,
        show_plots=False
    )
    print(f"   ✓ Visualizer: {visualizer}")
    
    # Create experiment
    print("\n2. Creating experiment...")
    experiment = Experiment(
        name="GPT2",
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
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED!")
    print("="*70)
    
    return experiment, results


def run_finbert_experiment():
    """Run FinBERT model experiment (specialized for financial text)."""
    
    try:
        import config
        DATA_FILE = config.DATA_FILE
    except ImportError:
        DATA_FILE = "fraud_data.csv"
    
    print("="*70)
    print("FINBERT FRAUD DETECTION EXPERIMENT")
    print("="*70)
    
    # Initialize components
    print("\n1. Initializing components...")
    
    # Create tokenizer for FinBERT
    tokenizer = TransformerTokenizer(
        model_name='yiyanghkust/finbert-pretrain',
        max_length=512,
        padding='max_length',
        truncation=True
    )
    print(f"   ✓ Tokenizer: {tokenizer}")
    
    # Create data loader
    data_loader = DataLoader(
        vectorizer=tokenizer,
        test_size=0.2,
        val_size=0.25,
        random_state=42
    )
    print(f"   ✓ Data Loader: {data_loader}")
    
    # Create HuggingFace FinBERT model
    model = HuggingFaceModel(
        model_name='yiyanghkust/finbert-pretrain',
        num_labels=2,
        num_epochs=3,
        batch_size=8,
        learning_rate=5e-5,
        warmup_steps=100,
        output_dir='./results/finbert',
        logging_dir='./logs/finbert'
    )
    print(f"   ✓ Model: {model}")
    
    # Create evaluator and visualizer
    evaluator = Evaluator(pos_label=1)
    visualizer = Visualizer(
        save_dir="./plots/finbert",
        save_plots=True,
        show_plots=False
    )
    
    # Create and run experiment
    experiment = Experiment(
        name="FinBERT",
        model=model,
        data_loader=data_loader,
        evaluator=evaluator,
        visualizer=visualizer
    )
    
    print("\n2. Running experiment...")
    results = experiment.run(
        data_file=DATA_FILE,
        evaluate_on_train=False,
        evaluate_on_val=True,
        evaluate_on_test=True,
        generate_plots=True
    )
    
    # Display summary
    print("\n3. Results Summary:")
    print("="*70)
    summary = experiment.get_results_summary()
    
    for dataset_name, metrics in summary.items():
        print(f"\n{dataset_name.upper()} SET:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:12s}: {value:.4f}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED!")
    print("="*70)
    
    return experiment, results


def compare_transformers():
    """Compare different transformer models."""
    
    try:
        import config
        DATA_FILE = config.DATA_FILE
    except ImportError:
        DATA_FILE = "fraud_data.csv"
    
    print("="*70)
    print("COMPARING TRANSFORMER MODELS")
    print("="*70)
    
    # Models to compare
    transformer_configs = {
        "GPT-2": {
            "model_name": "gpt2",
            "epochs": 4
        },
        "DistilBERT": {
            "model_name": "distilbert-base-uncased",
            "epochs": 3
        }
    }
    
    all_results = {}
    
    for model_display_name, config_dict in transformer_configs.items():
        print(f"\n{'='*70}")
        print(f"Testing {model_display_name}")
        print(f"{'='*70}")
        
        # Create components
        tokenizer = TransformerTokenizer(
            model_name=config_dict["model_name"],
            max_length=512
        )
        
        data_loader = DataLoader(tokenizer, test_size=0.2, val_size=0.25)
        
        model = HuggingFaceModel(
            model_name=config_dict["model_name"],
            num_epochs=config_dict["epochs"],
            batch_size=8,
            output_dir=f'./results/{model_display_name.lower().replace("-", "_")}',
            logging_dir=f'./logs/{model_display_name.lower().replace("-", "_")}'
        )
        
        evaluator = Evaluator()
        visualizer = Visualizer(
            save_dir="./plots/transformer_comparison",
            save_plots=True,
            show_plots=False
        )
        
        experiment = Experiment(
            model_display_name,
            model, data_loader, evaluator, visualizer
        )
        
        try:
            results = experiment.run(DATA_FILE, evaluate_on_val=False)
            all_results[model_display_name] = results['test']
        except Exception as e:
            print(f"Error training {model_display_name}: {e}")
            continue
    
    # Compare results
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 70)
        
        for model_name, results in all_results.items():
            print(f"{model_name:<20} "
                  f"{results['accuracy']:<12.4f} "
                  f"{results['precision']:<12.4f} "
                  f"{results['recall']:<12.4f} "
                  f"{results['f1']:<12.4f}")
        
        # Generate comparison plots
        visualizer = Visualizer(save_dir="./plots/transformer_comparison")
        visualizer.compare_models_barplot(all_results, metric='f1')
        visualizer.compare_models_radar(all_results)
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETED!")
        print("="*70)


if __name__ == "__main__":
    # Choose which experiment to run:
    
    # Option 1: Run GPT-2
    experiment, results = run_gpt2_experiment()
    
    # Option 2: Run FinBERT (uncomment to use)
    # experiment, results = run_finbert_experiment()
    
    # Option 3: Compare multiple transformers (uncomment to use)
    # compare_transformers()