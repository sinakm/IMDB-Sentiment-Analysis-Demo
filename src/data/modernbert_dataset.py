"""
Dataset class for ModernBERT-based sentiment analysis.

This module provides a PyTorch Dataset class that handles tokenization
and preprocessing specifically for ModernBERT models.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer
import numpy as np


class ModernBERTSentimentDataset(Dataset):
    """
    Dataset class for sentiment analysis using ModernBERT tokenization.
    
    This dataset handles:
    - Text tokenization using ModernBERT tokenizer
    - Proper padding and truncation
    - Label encoding for binary classification
    - Attention mask generation
    """
    
    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 tokenizer_name: str = "answerdotai/ModernBERT-base",
                 max_length: int = 512,
                 return_token_type_ids: bool = False):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels (0 or 1 for binary classification)
            tokenizer_name: Name of the ModernBERT tokenizer
            max_length: Maximum sequence length
            return_token_type_ids: Whether to return token type IDs
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.return_token_type_ids = return_token_type_ids
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure we have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Validate inputs
        assert len(texts) == len(labels), "Number of texts and labels must match"
        
        # Convert labels to tensor
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
            - input_ids: Token IDs [max_length]
            - attention_mask: Attention mask [max_length]
            - labels: Label tensor [1]
            - token_type_ids: Token type IDs [max_length] (if return_token_type_ids=True)
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=self.return_token_type_ids,
            return_tensors='pt'
        )
        
        # Prepare output dictionary
        output = {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }
        
        # Add token type IDs if requested
        if self.return_token_type_ids and 'token_type_ids' in encoding:
            output['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        
        return output
    
    def get_sample_text(self, idx: int) -> str:
        """Get the original text for a given index."""
        return self.texts[idx]
    
    def get_sample_label(self, idx: int) -> int:
        """Get the label for a given index."""
        return int(self.labels[idx].item())
    
    def get_tokenizer(self):
        """Get the tokenizer instance."""
        return self.tokenizer
    
    def decode_tokens(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_dataset_stats(self) -> Dict[str, float]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        # Text length statistics (in tokens)
        token_lengths = []
        for text in self.texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            token_lengths.append(len(tokens))
        
        # Label distribution
        positive_count = (self.labels == 1).sum().item()
        negative_count = (self.labels == 0).sum().item()
        
        return {
            'num_samples': len(self.texts),
            'positive_samples': positive_count,
            'negative_samples': negative_count,
            'positive_ratio': positive_count / len(self.texts),
            'avg_token_length': np.mean(token_lengths),
            'median_token_length': np.median(token_lengths),
            'min_token_length': np.min(token_lengths),
            'max_token_length': np.max(token_lengths),
            'std_token_length': np.std(token_lengths),
            'truncated_samples': sum(1 for length in token_lengths if length > self.max_length)
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training.
        
        Returns:
            Class weights tensor [2] for negative and positive classes
        """
        positive_count = (self.labels == 1).sum().item()
        negative_count = (self.labels == 0).sum().item()
        total_count = len(self.labels)
        
        # Calculate inverse frequency weights
        pos_weight = total_count / (2 * positive_count) if positive_count > 0 else 1.0
        neg_weight = total_count / (2 * negative_count) if negative_count > 0 else 1.0
        
        return torch.tensor([neg_weight, pos_weight], dtype=torch.float32)
    
    def create_subset(self, indices: List[int]) -> 'ModernBERTSentimentDataset':
        """
        Create a subset of the dataset.
        
        Args:
            indices: List of indices to include in the subset
            
        Returns:
            New dataset instance with the subset
        """
        subset_texts = [self.texts[i] for i in indices]
        subset_labels = [int(self.labels[i].item()) for i in indices]
        
        return ModernBERTSentimentDataset(
            texts=subset_texts,
            labels=subset_labels,
            tokenizer_name=self.tokenizer.name_or_path,
            max_length=self.max_length,
            return_token_type_ids=self.return_token_type_ids
        )


class ModernBERTDataCollator:
    """
    Data collator for ModernBERT datasets.
    
    This handles dynamic padding and batching for efficient training.
    """
    
    def __init__(self, 
                 tokenizer,
                 padding: bool = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 return_token_type_ids: bool = False):
        """
        Initialize the data collator.
        
        Args:
            tokenizer: ModernBERT tokenizer
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            pad_to_multiple_of: Pad to multiple of this value
            return_token_type_ids: Whether to return token type IDs
        """
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_token_type_ids = return_token_type_ids
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched features dictionary
        """
        # Extract components
        input_ids = [f['input_ids'] for f in features]
        attention_masks = [f['attention_mask'] for f in features]
        labels = torch.stack([f['labels'] for f in features])
        
        # Stack input_ids and attention_masks (they should already be padded)
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }
        
        # Add token type IDs if present
        if self.return_token_type_ids and 'token_type_ids' in features[0]:
            token_type_ids = torch.stack([f['token_type_ids'] for f in features])
            batch['token_type_ids'] = token_type_ids
        
        return batch


def create_modernbert_datasets(train_texts: List[str],
                              train_labels: List[int],
                              val_texts: List[str],
                              val_labels: List[int],
                              test_texts: List[str],
                              test_labels: List[int],
                              tokenizer_name: str = "answerdotai/ModernBERT-base",
                              max_length: int = 512,
                              return_token_type_ids: bool = False) -> Tuple[ModernBERTSentimentDataset, 
                                                                          ModernBERTSentimentDataset, 
                                                                          ModernBERTSentimentDataset]:
    """
    Create train, validation, and test datasets for ModernBERT.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        test_texts: Test texts
        test_labels: Test labels
        tokenizer_name: ModernBERT tokenizer name
        max_length: Maximum sequence length
        return_token_type_ids: Whether to return token type IDs
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = ModernBERTSentimentDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        return_token_type_ids=return_token_type_ids
    )
    
    val_dataset = ModernBERTSentimentDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        return_token_type_ids=return_token_type_ids
    )
    
    test_dataset = ModernBERTSentimentDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        return_token_type_ids=return_token_type_ids
    )
    
    return train_dataset, val_dataset, test_dataset


def analyze_tokenization(texts: List[str], 
                        tokenizer_name: str = "answerdotai/ModernBERT-base",
                        max_length: int = 512) -> Dict[str, any]:
    """
    Analyze tokenization statistics for a list of texts.
    
    Args:
        texts: List of texts to analyze
        tokenizer_name: ModernBERT tokenizer name
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenization statistics
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    token_lengths = []
    truncated_count = 0
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_lengths.append(len(tokens))
        if len(tokens) > max_length:
            truncated_count += 1
    
    return {
        'total_texts': len(texts),
        'avg_token_length': np.mean(token_lengths),
        'median_token_length': np.median(token_lengths),
        'min_token_length': np.min(token_lengths),
        'max_token_length': np.max(token_lengths),
        'std_token_length': np.std(token_lengths),
        'truncated_count': truncated_count,
        'truncation_rate': truncated_count / len(texts),
        'vocab_size': len(tokenizer),
        'special_tokens': {
            'pad_token': tokenizer.pad_token,
            'cls_token': tokenizer.cls_token,
            'sep_token': tokenizer.sep_token,
            'unk_token': tokenizer.unk_token
        }
    }
