from typing import List, Union, Optional
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_vectorizer import BaseVectorizer

from transformers import AutoTokenizer
import torch
import logging


class TransformerTokenizer(BaseVectorizer):
    """
    Transformer tokenizer wrapper that implements BaseVectorizer interface.
    
    Wraps HuggingFace tokenizers to provide consistent interface with
    other vectorization methods.
    """
    
    def __init__(
        self,
        model_name: str = 'gpt2',
        max_length: int = 512,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs
    ):
        """
        Initialize transformer tokenizer.
        
        Args:
            model_name: Name of the pretrained model/tokenizer (e.g., 'gpt2', 'bert-base-uncased')
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length', 'longest', or False)
            truncation: Whether to truncate sequences longer than max_length
            return_tensors: Format of returned tensors ('pt' for PyTorch, 'np' for NumPy, None for lists)
            add_special_tokens: Whether to add special tokens (CLS, SEP, etc.)
            **kwargs: Additional arguments passed to the tokenizer
        """
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            **kwargs
        )
        
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.add_special_tokens = add_special_tokens
        self.tokenizer_kwargs = kwargs
        
        # Initialize tokenizer
        logging.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle special tokens (some models don't have pad tokens)
        if self.tokenizer.pad_token is None:
            logging.info(f"Adding pad token to {model_name} tokenizer")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.vocab_size = len(self.tokenizer)
    
    def fit(self, texts: Union[List[str], pd.Series]) -> 'TransformerTokenizer':
        """
        Fit the tokenizer (no-op for pretrained tokenizers).
        
        Transformer tokenizers are pretrained and don't need fitting,
        but we implement this for interface compatibility.
        
        Args:
            texts: List or Series of text documents (not used)
            
        Returns:
            self (for method chaining)
        """
        # Pretrained tokenizers don't need fitting
        self.is_fitted = True
        logging.info(f"Tokenizer ready (pretrained, no fitting needed)")
        return self
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """
        Transform texts into token IDs.
        
        Args:
            texts: List or Series of text documents
            
        Returns:
            Numpy array of token IDs, shape: (n_samples, max_length)
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before transform. Call fit() first.")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            add_special_tokens=self.add_special_tokens,
            **self.tokenizer_kwargs
        )
        
        # Get input_ids
        input_ids = encoded['input_ids']
        
        # Convert to numpy array if needed
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.numpy()
        elif not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)
        
        return input_ids
    
    def transform_with_attention(
        self,
        texts: Union[List[str], pd.Series]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform texts and return both input IDs and attention masks.
        
        Useful for models that need attention masks (most transformers).
        
        Args:
            texts: List or Series of text documents
            
        Returns:
            Tuple of (input_ids, attention_masks), both as numpy arrays
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before transform.")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            add_special_tokens=self.add_special_tokens,
            **self.tokenizer_kwargs
        )
        
        # Get input_ids and attention_mask
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Convert to numpy arrays if needed
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.numpy()
            attention_mask = attention_mask.numpy()
        elif not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)
            attention_mask = np.array(attention_mask)
        
        return input_ids, attention_mask
    
    def get_feature_names(self) -> List[str]:
        """
        Get token IDs as feature names.
        
        For transformers, features are token positions, not interpretable words.
        
        Returns:
            List of feature names (token position indices)
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before getting feature names.")
        
        # Return position indices as strings
        return [f"token_pos_{i}" for i in range(self.max_length)]
    
    def _get_output_dim(self) -> int:
        """
        Get the output dimensionality (sequence length).
        
        Returns:
            Maximum sequence length
        """
        return self.max_length
    
    def get_vocab(self) -> dict:
        """
        Get the vocabulary mapping from tokens to IDs.
        
        Returns:
            Dictionary mapping tokens to their IDs
        """
        return self.tokenizer.get_vocab()
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size.
        
        Returns:
            Size of the vocabulary
        """
        return self.vocab_size
    
    def decode(self, token_ids: Union[List[int], np.ndarray]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def batch_decode(self, token_ids_batch: Union[List[List[int]], np.ndarray]) -> List[str]:
        """
        Decode a batch of token IDs back to text.
        
        Args:
            token_ids_batch: Batch of token IDs to decode
            
        Returns:
            List of decoded text strings
        """
        if isinstance(token_ids_batch, np.ndarray):
            token_ids_batch = token_ids_batch.tolist()
        
        return self.tokenizer.batch_decode(token_ids_batch, skip_special_tokens=True)
    
    def resize_token_embeddings(self, new_size: int) -> None:
        """
        Resize token embeddings (updates vocab_size).
        
        Useful when adding new special tokens.
        
        Args:
            new_size: New vocabulary size
        """
        self.vocab_size = new_size
        logging.info(f"Vocabulary size updated to {new_size}")
    
    def __repr__(self) -> str:
        """String representation of the tokenizer."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return (f"TransformerTokenizer(model={self.model_name}, "
                f"max_length={self.max_length}, {fitted_status})")