"""
Simple Verbalizer Classifier with pre-computed embeddings.

This module implements a sentiment analysis model using a simple classifier
trained on pre-computed ModernBERT embeddings with verbalizer template.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class VerbalizerClassifier(nn.Module):
    """
    Simple verbalizer classifier that works on pre-computed embeddings.
    
    This model implements:
    - Simple feedforward classifier
    - Dropout for regularization
    - Binary classification head
    - Works on pre-extracted ModernBERT embeddings
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 hidden_dim: int = 128,
                 dropout: float = 0.3):
        """
        Initialize the Verbalizer Classifier.
        
        Args:
            embedding_dim: Dimension of input embeddings (768 for ModernBERT)
            hidden_dim: Hidden dimension of classifier
            dropout: Dropout probability
        """
        super(VerbalizerClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        
        # Improved classifier architecture - deeper and more robust
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),  # 256 neurons
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),     # 128 neurons
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout * 0.7),  # Slightly less dropout in deeper layers
            
            nn.Linear(hidden_dim, hidden_dim // 2),    # 64 neurons
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden_dim // 2, 1)              # Output layer
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None,
                embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Pre-computed embeddings [batch_size, embedding_dim] (for trainer compatibility)
            attention_mask: Not used, kept for API compatibility
            lengths: Not used, kept for trainer compatibility
            embeddings: Alternative parameter name for embeddings (for direct usage)
            
        Returns:
            Classification logits [batch_size, 1]
        """
        # Use input_ids (trainer compatibility) or embeddings (direct usage)
        if embeddings is not None:
            input_embeddings = embeddings
        else:
            input_embeddings = input_ids
        
        # Simple forward pass through classifier
        logits = self.classifier(input_embeddings)
        return logits, None  # Return tuple for trainer compatibility (logits, attention_weights)
    
    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            embeddings: Pre-computed embeddings [batch_size, embedding_dim]
            
        Returns:
            Probabilities [batch_size]
        """
        with torch.no_grad():
            logits, _ = self.forward(embeddings=embeddings)  # Unpack tuple
            probabilities = torch.sigmoid(logits.squeeze())
            return probabilities
    
    def predict(self, 
               embeddings: torch.Tensor,
               threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions.
        
        Args:
            embeddings: Pre-computed embeddings [batch_size, embedding_dim]
            threshold: Classification threshold
            
        Returns:
            Binary predictions [batch_size]
        """
        probabilities = self.predict_proba(embeddings)
        return (probabilities > threshold).long()
    
    def get_model_info(self) -> dict:
        """
        Get model information for logging/debugging.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'VerbalizerClassifier',
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_prob,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'description': 'Simple classifier on pre-computed ModernBERT embeddings'
        }
