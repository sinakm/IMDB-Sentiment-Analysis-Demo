"""
Pure LSTM Classifier with learned embeddings.

This module implements a sentiment analysis model using LSTM with embeddings
learned from scratch. It includes attention mechanism for better sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PureLSTMClassifier(nn.Module):
    """
    Pure LSTM classifier that learns embeddings from scratch.
    
    This model implements:
    - Embedding layer trained from scratch
    - Bidirectional LSTM for sequence modeling
    - Attention mechanism for weighted pooling
    - Dropout for regularization
    - Binary classification head
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 200,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True,
                 use_attention: bool = True):
        """
        Initialize the Pure LSTM Classifier.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of embedding vectors
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
        """
        super(PureLSTMClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Embedding layer - learned from scratch
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.embedding.weight)
        # Set padding embedding to zero
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
        
        # LSTM layer
        lstm_input_dim = embed_dim
        lstm_hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate final LSTM output dimension
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(lstm_output_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
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
        
        # Embedding lookup
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
            attention_weights = self.attention(lstm_output)  # [batch_size, actual_seq_len, 1]
            
            # Apply attention mask if provided and sizes match
            if attention_mask is not None:
                # Ensure attention mask matches the actual sequence length
                actual_seq_len = lstm_output.size(1)
                if attention_mask.size(1) != actual_seq_len:
                    # Truncate or pad attention mask to match
                    if attention_mask.size(1) > actual_seq_len:
                        attention_mask = attention_mask[:, :actual_seq_len]
                    else:
                        # Pad with zeros
                        pad_size = actual_seq_len - attention_mask.size(1)
                        attention_mask = F.pad(attention_mask, (0, pad_size), value=0)
                
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
    
    def get_model_info(self) -> dict:
        """
        Get model information for logging/debugging.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'PureLSTMClassifier',
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'use_attention': self.use_attention,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
