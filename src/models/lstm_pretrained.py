"""
LSTM Classifier with pre-trained embeddings.

This module implements a sentiment analysis model using LSTM with pre-trained
GloVe embeddings for better semantic understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class LSTMWithPretrainedEmbeddings(nn.Module):
    """
    LSTM classifier that uses pre-trained embeddings (e.g., GloVe).
    
    This model implements:
    - Pre-trained embedding layer (GloVe, Word2Vec, etc.)
    - Bidirectional LSTM for sequence modeling
    - Attention mechanism for weighted pooling
    - Dropout for regularization
    - Binary classification head
    """
    
    def __init__(self, 
                 pretrained_embeddings: torch.Tensor,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 freeze_embeddings: bool = False,
                 fine_tune_embeddings: bool = True):
        """
        Initialize the LSTM with Pre-trained Embeddings.
        
        Args:
            pretrained_embeddings: Pre-trained embedding matrix [vocab_size, embed_dim]
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
            freeze_embeddings: Whether to freeze embedding weights
            fine_tune_embeddings: Whether to allow fine-tuning of embeddings
        """
        super(LSTMWithPretrainedEmbeddings, self).__init__()
        
        vocab_size, embed_dim = pretrained_embeddings.shape
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.freeze_embeddings = freeze_embeddings
        
        # Pre-trained embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Load pre-trained weights
        self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Set padding embedding to zero
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
        
        # Freeze embeddings if specified
        if freeze_embeddings or not fine_tune_embeddings:
            self.embedding.weight.requires_grad = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate final LSTM output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(lstm_output_dim, 1)
        
        # Initialize weights (except embeddings)
        self._init_weights()
    
    @classmethod
    def from_glove(cls, 
                   glove_embeddings: torch.Tensor,
                   **kwargs):
        """
        Create model from GloVe embeddings.
        
        Args:
            glove_embeddings: GloVe embedding matrix
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Initialized model
        """
        return cls(pretrained_embeddings=glove_embeddings, **kwargs)
    
    def _init_weights(self):
        """Initialize model weights (excluding embeddings)."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        # Initialize attention weights
        if self.use_attention:
            nn.init.xavier_uniform_(self.attention.weight)
            self.attention.bias.data.fill_(0)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (optional)
            lengths: Actual sequence lengths [batch_size] (optional)
            
        Returns:
            Tuple of (logits, attention_weights)
            - logits: Classification logits [batch_size, 1]
            - attention_weights: Attention weights [batch_size, seq_len, 1] (if using attention)
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding lookup with pre-trained embeddings
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Apply dropout to embeddings
        embedded = self.dropout(embedded)
        
        # Pack sequences if lengths are provided (for efficiency)
        if lengths is not None:
            # Sort by length for packing
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sort_idx]
            
            # Pack sequences
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            
            # LSTM forward pass
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # Unpack sequences
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            
            # Restore original order
            _, unsort_idx = sort_idx.sort()
            lstm_output = lstm_output[unsort_idx]
            
        else:
            # Standard LSTM forward pass
            lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # lstm_output: [batch_size, seq_len, lstm_output_dim]
        
        # Apply attention mechanism or use final hidden state
        if self.use_attention:
            # Attention-based pooling
            attention_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attention_weights = attention_weights.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, float('-inf')
                )
            
            # Softmax over sequence dimension
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Weighted sum
            attended_output = torch.sum(attention_weights * lstm_output, dim=1)  # [batch_size, lstm_output_dim]
            
        else:
            # Use final hidden state
            if self.bidirectional:
                # Concatenate forward and backward final hidden states
                attended_output = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                attended_output = hidden[-1]
            
            attention_weights = None
        
        # Apply dropout
        attended_output = self.dropout(attended_output)
        
        # Classification
        logits = self.classifier(attended_output)  # [batch_size, 1]
        
        return logits, attention_weights
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for input tokens.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, seq_len, embed_dim]
        """
        return self.embedding(input_ids)
    
    def get_attention_weights(self, 
                            input_ids: torch.Tensor, 
                            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (optional)
            
        Returns:
            Attention weights [batch_size, seq_len]
        """
        if not self.use_attention:
            raise ValueError("Model was not initialized with attention mechanism")
        
        with torch.no_grad():
            _, attention_weights = self.forward(input_ids, attention_mask)
            return attention_weights.squeeze(-1)  # Remove last dimension
    
    def predict_proba(self, 
                     input_ids: torch.Tensor, 
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (optional)
            
        Returns:
            Probabilities [batch_size]
        """
        with torch.no_grad():
            logits, _ = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits.squeeze())
            return probabilities
    
    def predict(self, 
               input_ids: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None,
               threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (optional)
            threshold: Classification threshold
            
        Returns:
            Binary predictions [batch_size]
        """
        probabilities = self.predict_proba(input_ids, attention_mask)
        return (probabilities > threshold).long()
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding weights for fine-tuning."""
        self.embedding.weight.requires_grad = True
        self.freeze_embeddings = False
    
    def freeze_embeddings_layer(self):
        """Freeze embedding weights."""
        self.embedding.weight.requires_grad = False
        self.freeze_embeddings = True
    
    def get_embedding_similarity(self, word_ids: torch.Tensor) -> torch.Tensor:
        """
        Get cosine similarity between word embeddings.
        
        Args:
            word_ids: Word IDs to compare [num_words]
            
        Returns:
            Similarity matrix [num_words, num_words]
        """
        with torch.no_grad():
            embeddings = self.embedding(word_ids)  # [num_words, embed_dim]
            
            # Normalize embeddings
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = torch.mm(embeddings_norm, embeddings_norm.t())
            
            return similarity
    
    def get_model_info(self) -> dict:
        """
        Get model information for logging/debugging.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding_params = self.embedding.weight.numel()
        
        return {
            'model_type': 'LSTMWithPretrainedEmbeddings',
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'use_attention': self.use_attention,
            'freeze_embeddings': self.freeze_embeddings,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_parameters': embedding_params,
            'embedding_trainable': self.embedding.weight.requires_grad
        }


class EmbeddingAdapter(nn.Module):
    """
    Adapter layer for pre-trained embeddings.
    
    This can be used to adapt pre-trained embeddings to the specific task
    while keeping the original embeddings frozen.
    """
    
    def __init__(self, embed_dim: int, adapter_dim: int = 64):
        """
        Initialize embedding adapter.
        
        Args:
            embed_dim: Dimension of input embeddings
            adapter_dim: Dimension of adapter bottleneck
        """
        super(EmbeddingAdapter, self).__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, embed_dim)
        )
        
        # Initialize with small weights
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embed_dim]
            
        Returns:
            Adapted embeddings [batch_size, seq_len, embed_dim]
        """
        # Residual connection
        return embeddings + self.adapter(embeddings)
