"""
Dataset class for verbalizer embeddings.

This module provides a simple dataset class that works with pre-computed
embeddings for the verbalizer classifier.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any


class VerbalizerDataset(Dataset):
    """
    Dataset for verbalizer embeddings.
    
    This dataset works with pre-computed embeddings rather than raw text.
    """
    
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Initialize verbalizer dataset.
        
        Args:
            embeddings: Pre-computed embeddings [num_samples, embedding_dim]
            labels: Labels [num_samples]
        """
        self.embeddings = embeddings
        self.labels = labels
        
        assert len(embeddings) == len(labels), "Embeddings and labels must have same length"
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary with input_ids (embeddings) and labels for trainer compatibility
        """
        return {
            'input_ids': self.embeddings[idx],  # Rename for trainer compatibility
            'labels': self.labels[idx]
        }
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Get embedding statistics.
        
        Returns:
            Dictionary with embedding statistics
        """
        return {
            'num_samples': len(self.embeddings),
            'embedding_dim': self.embeddings.shape[1],
            'embedding_mean': float(self.embeddings.mean()),
            'embedding_std': float(self.embeddings.std()),
            'positive_samples': int((self.labels == 1).sum()),
            'negative_samples': int((self.labels == 0).sum())
        }
