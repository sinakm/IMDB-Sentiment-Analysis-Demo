"""
ModernBERT-based Sentiment Classifier.

This module implements a sentiment analysis model using ModernBERT
from Hugging Face for state-of-the-art performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np


class ModernBERTClassifier(nn.Module):
    """
    Sentiment classifier using ModernBERT as the backbone.
    
    This model implements:
    - ModernBERT encoder for contextual embeddings
    - Optional pooling strategies (CLS, mean, max, attention)
    - Dropout for regularization
    - Binary classification head
    - Optional adapter layers for efficient fine-tuning
    """
    
    def __init__(self, 
                 model_name: str = "answerdotai/ModernBERT-base",
                 num_classes: int = 1,
                 dropout: float = 0.1,
                 pooling_strategy: str = "cls",
                 use_adapter: bool = False,
                 adapter_dim: int = 64,
                 freeze_encoder: bool = False,
                 max_length: int = 512):
        """
        Initialize the ModernBERT classifier.
        
        Args:
            model_name: ModernBERT model name from Hugging Face
            num_classes: Number of output classes (1 for binary classification)
            dropout: Dropout probability
            pooling_strategy: How to pool sequence representations ("cls", "mean", "max", "attention")
            use_adapter: Whether to use adapter layers
            adapter_dim: Dimension of adapter layers
            freeze_encoder: Whether to freeze the encoder weights
            max_length: Maximum sequence length
        """
        super(ModernBERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.pooling_strategy = pooling_strategy
        self.use_adapter = use_adapter
        self.freeze_encoder = freeze_encoder
        self.max_length = max_length
        
        # Load ModernBERT configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from config
        self.hidden_size = self.config.hidden_size
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Adapter layers (optional)
        if use_adapter:
            self.adapter = AdapterLayer(self.hidden_size, adapter_dim)
        
        # Pooling layer for attention-based pooling
        if pooling_strategy == "attention":
            self.attention_pooling = AttentionPooling(self.hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize classification head
        self._init_classifier_weights()
    
    @classmethod
    def from_pretrained(cls, 
                       model_name: str = "answerdotai/ModernBERT-base",
                       **kwargs):
        """
        Create model from pre-trained ModernBERT.
        
        Args:
            model_name: ModernBERT model name
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Initialized model
        """
        return cls(model_name=model_name, **kwargs)
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - logits: Classification logits [batch_size, num_classes]
            - pooled_output: Pooled sequence representation [batch_size, hidden_size]
            - attention_weights: Attention weights (if return_attention=True)
        """
        # ModernBERT forward pass
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=return_attention,
            return_dict=True
        )
        
        # Get sequence output
        sequence_output = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply adapter if specified
        if self.use_adapter:
            sequence_output = self.adapter(sequence_output)
        
        # Pool sequence representations
        pooled_output = self._pool_sequence(sequence_output, attention_mask)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Prepare output
        output = {
            'logits': logits,
            'pooled_output': pooled_output
        }
        
        if return_attention and encoder_outputs.attentions is not None:
            output['attention_weights'] = encoder_outputs.attentions
        
        return output
    
    def _pool_sequence(self, 
                      sequence_output: torch.Tensor, 
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool sequence representations using specified strategy.
        
        Args:
            sequence_output: Sequence output [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token representation
            return sequence_output[:, 0]  # [batch_size, hidden_size]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling over non-padded tokens
            if attention_mask is not None:
                # Mask out padded tokens
                masked_output = sequence_output * attention_mask.unsqueeze(-1)
                # Sum and divide by actual length
                sum_output = masked_output.sum(dim=1)
                lengths = attention_mask.sum(dim=1, keepdim=True).float()
                return sum_output / lengths
            else:
                return sequence_output.mean(dim=1)
        
        elif self.pooling_strategy == "max":
            # Max pooling over non-padded tokens
            if attention_mask is not None:
                # Set padded positions to very negative values
                masked_output = sequence_output.clone()
                masked_output[attention_mask == 0] = float('-inf')
                return masked_output.max(dim=1)[0]
            else:
                return sequence_output.max(dim=1)[0]
        
        elif self.pooling_strategy == "attention":
            # Attention-based pooling
            return self.attention_pooling(sequence_output, attention_mask)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def predict_proba(self, 
                     input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Probabilities [batch_size] (for binary classification)
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            
            if self.num_classes == 1:
                # Binary classification
                probabilities = torch.sigmoid(logits.squeeze())
            else:
                # Multi-class classification
                probabilities = F.softmax(logits, dim=-1)
            
            return probabilities
    
    def predict(self, 
               input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            threshold: Classification threshold (for binary classification)
            
        Returns:
            Predictions [batch_size]
        """
        probabilities = self.predict_proba(input_ids, attention_mask)
        
        if self.num_classes == 1:
            # Binary classification
            return (probabilities > threshold).long()
        else:
            # Multi-class classification
            return probabilities.argmax(dim=-1)
    
    def get_embeddings(self, 
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get contextualized embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            return encoder_outputs.last_hidden_state
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
    
    def freeze_encoder_layers(self):
        """Freeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for logging/debugging.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        
        return {
            'model_type': 'ModernBERTClassifier',
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'pooling_strategy': self.pooling_strategy,
            'use_adapter': self.use_adapter,
            'freeze_encoder': self.freeze_encoder,
            'max_length': self.max_length,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_parameters': encoder_params,
            'encoder_trainable': not self.freeze_encoder
        }


class AdapterLayer(nn.Module):
    """
    Adapter layer for efficient fine-tuning.
    
    This implements the adapter architecture from "Parameter-Efficient Transfer Learning for NLP"
    """
    
    def __init__(self, hidden_size: int, adapter_dim: int = 64):
        """
        Initialize adapter layer.
        
        Args:
            hidden_size: Hidden size of the model
            adapter_dim: Dimension of the adapter bottleneck
        """
        super(AdapterLayer, self).__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_size)
        )
        
        # Initialize with small weights for stability
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Adapted hidden states [batch_size, seq_len, hidden_size]
        """
        # Residual connection
        return hidden_states + self.adapter(hidden_states)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer.
    
    This computes attention weights over the sequence and performs weighted pooling.
    """
    
    def __init__(self, hidden_size: int):
        """
        Initialize attention pooling.
        
        Args:
            hidden_size: Hidden size of the input
        """
        super(AttentionPooling, self).__init__()
        
        self.attention = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
    
    def forward(self, 
                sequence_output: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through attention pooling.
        
        Args:
            sequence_output: Sequence output [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        # Compute attention weights
        attention_weights = self.attention(sequence_output)  # [batch_size, seq_len, 1]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask.unsqueeze(-1) == 0, float('-inf')
            )
        
        # Softmax over sequence dimension
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        pooled_output = torch.sum(attention_weights * sequence_output, dim=1)
        
        return pooled_output


class ModernBERTTokenizer:
    """
    Wrapper for ModernBERT tokenizer with additional utilities.
    """
    
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", max_length: int = 512):
        """
        Initialize tokenizer.
        
        Args:
            model_name: ModernBERT model name
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.model_name = model_name
    
    def encode_texts(self, 
                    texts: list,
                    add_special_tokens: bool = True,
                    padding: str = "max_length",
                    truncation: bool = True,
                    return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Encode texts for model input.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens
            padding: Padding strategy
            truncation: Whether to truncate long sequences
            return_tensors: Format of returned tensors
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
    
    def decode_tokens(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> list:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad_token_id': self.tokenizer.pad_token_id,
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token_id': self.tokenizer.sep_token_id,
            'unk_token_id': self.tokenizer.unk_token_id,
            'mask_token_id': getattr(self.tokenizer, 'mask_token_id', None)
        }
