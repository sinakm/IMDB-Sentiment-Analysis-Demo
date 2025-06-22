"""
Dataset classes for sentiment analysis.

This module implements PyTorch Dataset classes for all three model approaches:
1. LSTM datasets (pure and pre-trained embeddings)
2. Verbalizer dataset for transformer-based approach
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from .preprocessing import TextPreprocessor, text_to_indices, create_attention_mask


class IMDBDataset(Dataset):
    """Base IMDB dataset class."""
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 max_length: int = 512):
        """
        Initialize IMDB dataset.
        
        Args:
            texts: List of review texts
            labels: List of labels (0=negative, 1=positive)
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        assert len(texts) == len(labels), "Texts and labels must have same length"
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def get_text_lengths(self) -> List[int]:
        """Get text lengths in words."""
        return [len(text.split()) for text in self.texts]
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get label distribution."""
        from collections import Counter
        return dict(Counter(self.labels))


class LSTMDataset(IMDBDataset):
    """
    Dataset for LSTM models (both pure and pre-trained embeddings).
    
    This dataset converts texts to token indices using a vocabulary.
    """
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 vocab: Dict[str, int],
                 max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize LSTM dataset.
        
        Args:
            texts: List of review texts
            labels: List of labels
            vocab: Vocabulary mapping
            max_length: Maximum sequence length
            preprocessor: Text preprocessor
        """
        super().__init__(texts, labels, max_length)
        
        self.vocab = vocab
        self.preprocessor = preprocessor or TextPreprocessor(
            lowercase=True, 
            remove_punctuation=False
        )
        
        # Pre-process and convert texts to indices
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Pre-process all texts and convert to indices."""
        print("Preprocessing texts for LSTM dataset...")
        
        # Convert texts to token indices
        self.token_ids, self.lengths = text_to_indices(
            self.texts, 
            self.vocab, 
            self.max_length, 
            self.preprocessor
        )
        
        # Create attention masks
        self.attention_masks = create_attention_mask(self.lengths, self.max_length)
        
        # Convert labels to tensor
        self.label_tensor = torch.tensor(self.labels, dtype=torch.long)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary with model inputs
        """
        return {
            'input_ids': self.token_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'lengths': self.lengths[idx],
            'labels': self.label_tensor[idx]
        }
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_token_statistics(self) -> Dict[str, float]:
        """Get token-level statistics."""
        all_lengths = self.lengths.numpy()
        return {
            'avg_tokens': float(np.mean(all_lengths)),
            'median_tokens': float(np.median(all_lengths)),
            'min_tokens': int(np.min(all_lengths)),
            'max_tokens': int(np.max(all_lengths)),
            'std_tokens': float(np.std(all_lengths))
        }


class VerbalizerDataset(IMDBDataset):
    """
    Simple dataset for verbalizer approach.
    
    This dataset appends a single verbalizer template to texts and uses ModernBERT tokenization.
    The model will extract the last token embedding for classification.
    """
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 tokenizer_name: str = 'answerdotai/ModernBERT-base',
                 max_length: int = 512):
        """
        Initialize simple verbalizer dataset.
        
        Args:
            texts: List of review texts
            labels: List of labels
            tokenizer_name: Transformer tokenizer name (ModernBERT)
            max_length: Maximum sequence length
        """
        super().__init__(texts, labels, max_length)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Simple verbalizer template (always the same)
        self.verbalizer_template = " The sentiment of this statement is positive."
        
        # Pre-process data
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Pre-process all texts with verbalizer template."""
        print("Preprocessing texts for simple verbalizer dataset...")
        
        # Add template to all texts
        verbalized_texts = [text + self.verbalizer_template for text in self.texts]
        
        # Tokenize all texts
        encoding = self.tokenizer(
            verbalized_texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        self.input_ids = encoding['input_ids']
        self.attention_masks = encoding['attention_mask']
        
        # Convert labels to tensor
        self.label_tensor = torch.tensor(self.labels, dtype=torch.float)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary with model inputs (compatible with new verbalizer model)
        """
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.label_tensor[idx]
        }
    
    def get_verbalizer_statistics(self) -> Dict[str, Union[int, float]]:
        """Get verbalizer-specific statistics."""
        # Calculate sequence lengths (non-padding tokens)
        lengths = self.attention_masks.sum(dim=1).numpy()
        
        return {
            'avg_sequence_length': float(np.mean(lengths)),
            'min_sequence_length': int(np.min(lengths)),
            'max_sequence_length': int(np.max(lengths)),
            'std_sequence_length': float(np.std(lengths)),
            'verbalizer_template': self.verbalizer_template
        }


class MultiModelDataset:
    """
    Wrapper class that creates datasets for all three model approaches.
    
    This is useful for training and comparing all models on the same data splits.
    """
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 vocab: Dict[str, int],
                 tokenizer_name: str = 'bert-base-uncased',
                 max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize multi-model dataset wrapper.
        
        Args:
            texts: List of review texts
            labels: List of labels
            vocab: Vocabulary for LSTM models
            tokenizer_name: Transformer tokenizer name
            max_length: Maximum sequence length
            preprocessor: Text preprocessor for LSTM models
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.preprocessor = preprocessor
        
        # Create datasets for all models
        self._create_datasets()
    
    def _create_datasets(self):
        """Create datasets for all model types."""
        print("Creating datasets for all model types...")
        
        # LSTM dataset (for both pure and pre-trained models)
        self.lstm_dataset = LSTMDataset(
            texts=self.texts,
            labels=self.labels,
            vocab=self.vocab,
            max_length=self.max_length,
            preprocessor=self.preprocessor
        )
        
        # Verbalizer dataset (simplified)
        self.verbalizer_dataset = VerbalizerDataset(
            texts=self.texts,
            labels=self.labels,
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length
        )
        
        # For backward compatibility, use same dataset
        self.verbalizer_both_dataset = self.verbalizer_dataset
    
    def get_lstm_dataset(self) -> LSTMDataset:
        """Get LSTM dataset."""
        return self.lstm_dataset
    
    def get_verbalizer_dataset(self, use_both_templates: bool = False) -> VerbalizerDataset:
        """Get verbalizer dataset."""
        if use_both_templates:
            return self.verbalizer_both_dataset
        else:
            return self.verbalizer_dataset
    
    def get_all_datasets(self) -> Dict[str, Dataset]:
        """Get all datasets."""
        return {
            'lstm': self.lstm_dataset,
            'verbalizer': self.verbalizer_dataset,
            'verbalizer_both': self.verbalizer_both_dataset
        }
    
    def get_dataset_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all datasets."""
        return {
            'lstm': {
                **self.lstm_dataset.get_token_statistics(),
                **self.lstm_dataset.get_label_distribution()
            },
            'verbalizer': {
                **self.verbalizer_dataset.get_verbalizer_statistics(),
                **self.verbalizer_dataset.get_label_distribution()
            },
            'verbalizer_both': {
                **self.verbalizer_both_dataset.get_verbalizer_statistics(),
                **self.verbalizer_both_dataset.get_label_distribution()
            }
        }


class DatasetFactory:
    """Factory class for creating datasets."""
    
    @staticmethod
    def create_lstm_datasets(train_texts: List[str],
                           train_labels: List[int],
                           val_texts: List[str],
                           val_labels: List[int],
                           test_texts: List[str],
                           test_labels: List[int],
                           vocab: Dict[str, int],
                           max_length: int = 512,
                           preprocessor: Optional[TextPreprocessor] = None) -> Tuple[LSTMDataset, LSTMDataset, LSTMDataset]:
        """
        Create LSTM datasets for train/val/test splits.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = LSTMDataset(train_texts, train_labels, vocab, max_length, preprocessor)
        val_dataset = LSTMDataset(val_texts, val_labels, vocab, max_length, preprocessor)
        test_dataset = LSTMDataset(test_texts, test_labels, vocab, max_length, preprocessor)
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def create_verbalizer_datasets(train_texts: List[str],
                                 train_labels: List[int],
                                 val_texts: List[str],
                                 val_labels: List[int],
                                 test_texts: List[str],
                                 test_labels: List[int],
                                 tokenizer_name: str = 'answerdotai/ModernBERT-base',
                                 max_length: int = 512,
                                 template_type: str = 'positive') -> Tuple[VerbalizerDataset, VerbalizerDataset, VerbalizerDataset]:
        """
        Create verbalizer datasets for train/val/test splits.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = VerbalizerDataset(train_texts, train_labels, tokenizer_name, max_length)
        val_dataset = VerbalizerDataset(val_texts, val_labels, tokenizer_name, max_length)
        test_dataset = VerbalizerDataset(test_texts, test_labels, tokenizer_name, max_length)
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def create_all_datasets(train_texts: List[str],
                          train_labels: List[int],
                          val_texts: List[str],
                          val_labels: List[int],
                          test_texts: List[str],
                          test_labels: List[int],
                          vocab: Dict[str, int],
                          tokenizer_name: str = 'bert-base-uncased',
                          max_length: int = 512,
                          preprocessor: Optional[TextPreprocessor] = None) -> Dict[str, Tuple[Dataset, Dataset, Dataset]]:
        """
        Create all datasets for all model types.
        
        Returns:
            Dictionary with datasets for each model type
        """
        # LSTM datasets
        lstm_datasets = DatasetFactory.create_lstm_datasets(
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
            vocab, max_length, preprocessor
        )
        
        # Verbalizer datasets
        verbalizer_datasets = DatasetFactory.create_verbalizer_datasets(
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
            tokenizer_name, max_length
        )
        
        return {
            'lstm': lstm_datasets,
            'verbalizer': verbalizer_datasets
        }
