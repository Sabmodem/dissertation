"""
Transformer Model Experiment using the modular framework.

This script demonstrates how to train transformer models (GPT-2, FinBERT, etc.)
using the HuggingFace model wrapper and TransformerTokenizer.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from vectorizers.transformer_tokenizer import TransformerTokenizer
from models.hf_model import HuggingFaceModel
from structuring.common.utils import make_experiment

def main():
    """Run GPT-2 model experiment."""
    tokenizer = TransformerTokenizer(
        model_name='gpt2',
        max_length=512,
        padding='max_length',
        truncation=True
    )
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
    return make_experiment('GPT2', model, tokenizer)