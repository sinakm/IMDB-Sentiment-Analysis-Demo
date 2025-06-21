"""
Verbalizer/PET Classifier using transformer embeddings.

This module implements the innovative verbalizer approach where we append
"This statement is positive" to each review and extract the contextualized
embedding of the "positive" token for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Tuple, List, Dict
import numpy as np


class VerbalizerClassifier(nn.Module):
    """
    Verbalizer-based classifier using Pattern-Exploiting Training (PET) approach.
    
    This innovative model:
    1. Appends verbalizer templates like "This statement is positive" to reviews
    2. Uses a pre-trained transformer to get contextualized embeddings
    3. Extracts the embedding of the verbalizer token ("positive")
    4. Uses this contextualized embedding for classification
    
    The key insight is that the transformer will contextualize the word "positive"
    based on the entire review content, providing rich semantic information.
    """
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 num_classes: int = 1,
                 dropout: float = 0.3,
                 freeze_transformer: bool = False,
                 use_pooling: bool = False,
                 pooling_strategy: str = 'mean'):
        """
        Initialize the Verbalizer Classifier.
        
        Args:
            model_name: Pre-trained transformer model name
            num_classes: Number of output classes (1 for binary classification)
            dropout: Dropout probability
            freeze_transformer: Whether to freeze transformer weights
            use_pooling: Whether to use pooling over multiple verbalizer tokens
            pooling_strategy: Pooling strategy ('mean', 'max', 'attention')
        """
        super(VerbalizerClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_transformer = freeze_transformer
        self.use_pooling = use_pooling
        self.pooling_strategy = pooling_strategy
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze transformer if specified
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Get model dimensions
        self.hidden_size = self.transformer.config.hidden_size
        
        # Verbalizer tokens
        self.positive_token = "positive"
        self.negative_token = "negative"
        
        # Get token IDs for verbalizers
        self.positive_token_id = self.tokenizer.convert_tokens_to_ids(self.positive_token)
        self.negative_token_id = self.tokenizer.convert_tokens_to_ids(self.negative_token)
        
        # Templates for verbalizer
        self.templates = {
            'positive': " This statement is positive",
            'negative': " This statement is negative",
            'neutral': " This statement is neutral"
        }
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        if use_pooling and pooling_strategy == 'attention':
            # Attention-based pooling over verbalizer embeddings
            self.attention_pooling = nn.Linear(self.hidden_size, 1)
            self.classifier = nn.Linear(self.hidden_size, num_classes)
        else:
            # Simple classification on verbalizer token embedding
            self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)
        
        if hasattr(self, 'attention_pooling'):
            nn.init.xavier_uniform_(self.attention_pooling.weight)
            self.attention_pooling.bias.data.fill_(0)
    
    def create_verbalizer_input(self, 
                              texts: List[str], 
                              template_type: str = 'positive',
                              max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Create input with verbalizer template.
        
        Args:
            texts: List of input texts
            template_type: Type of template ('positive', 'negative', 'neutral')
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs and verbalizer positions
        """
        template = self.templates[template_type]
        
        # Add template to each text
        verbalized_texts = [text + template for text in texts]
        
        # Tokenize
        encoding = self.tokenizer(
            verbalized_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Find verbalizer token positions
        verbalizer_positions = []
        target_token_id = self.positive_token_id if template_type == 'positive' else self.negative_token_id
        
        for i, input_ids in enumerate(encoding['input_ids']):
            # Find the position of the verbalizer token
            positions = (input_ids == target_token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                # Use the last occurrence (most likely the one from our template)
                verbalizer_positions.append(positions[-1].item())
            else:
                # Fallback: use position before [SEP] or [PAD]
                sep_positions = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                if len(sep_positions) > 0:
                    verbalizer_positions.append(sep_positions[0].item() - 1)
                else:
                    verbalizer_positions.append(max_length - 2)  # Before [PAD]
        
        encoding['verbalizer_positions'] = torch.tensor(verbalizer_positions)
        return encoding
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                verbalizer_positions: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Forward pass of the verbalizer model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            verbalizer_positions: Positions of verbalizer tokens [batch_size]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        batch_size = input_ids.size(0)
        
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Extract verbalizer token embeddings
        verbalizer_embeddings = []
        for i in range(batch_size):
            pos = verbalizer_positions[i].item()
            verbalizer_embeddings.append(last_hidden_state[i, pos])
        
        verbalizer_embeddings = torch.stack(verbalizer_embeddings)  # [batch_size, hidden_size]
        
        # Apply dropout
        verbalizer_embeddings = self.dropout(verbalizer_embeddings)
        
        # Classification
        logits = self.classifier(verbalizer_embeddings)
        
        return logits
    
    def forward_with_both_templates(self, 
                                  texts: List[str],
                                  max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using both positive and negative templates for comparison.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (positive_logits, negative_logits)
        """
        # Create inputs with positive template
        pos_inputs = self.create_verbalizer_input(texts, 'positive', max_length)
        pos_logits = self.forward(**pos_inputs)
        
        # Create inputs with negative template
        neg_inputs = self.create_verbalizer_input(texts, 'negative', max_length)
        neg_logits = self.forward(**neg_inputs)
        
        return pos_logits, neg_logits
    
    def predict_with_verbalizer_analysis(self, 
                                       texts: List[str],
                                       max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Predict with detailed verbalizer analysis.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with predictions and analysis
        """
        with torch.no_grad():
            # Get predictions with both templates
            pos_logits, neg_logits = self.forward_with_both_templates(texts, max_length)
            
            # Convert to probabilities
            pos_probs = torch.sigmoid(pos_logits.squeeze())
            neg_probs = torch.sigmoid(neg_logits.squeeze())
            
            # Calculate confidence difference
            confidence_diff = pos_probs - (1 - neg_probs)  # Should be close to 0 for consistent predictions
            
            # Final prediction (using positive template)
            predictions = (pos_probs > 0.5).long()
            
            return {
                'predictions': predictions,
                'positive_probs': pos_probs,
                'negative_probs': neg_probs,
                'confidence_difference': confidence_diff,
                'consistency_score': 1.0 - torch.abs(confidence_diff)  # Higher = more consistent
            }
    
    def get_verbalizer_embeddings(self, 
                                texts: List[str],
                                template_type: str = 'positive',
                                max_length: int = 512) -> torch.Tensor:
        """
        Get the contextualized verbalizer token embeddings.
        
        Args:
            texts: List of input texts
            template_type: Type of template to use
            max_length: Maximum sequence length
            
        Returns:
            Verbalizer embeddings [batch_size, hidden_size]
        """
        with torch.no_grad():
            inputs = self.create_verbalizer_input(texts, template_type, max_length)
            
            # Get transformer outputs
            outputs = self.transformer(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True
            )
            
            last_hidden_state = outputs.last_hidden_state
            batch_size = last_hidden_state.size(0)
            
            # Extract verbalizer embeddings
            verbalizer_embeddings = []
            for i in range(batch_size):
                pos = inputs['verbalizer_positions'][i].item()
                verbalizer_embeddings.append(last_hidden_state[i, pos])
            
            return torch.stack(verbalizer_embeddings)
    
    def analyze_verbalizer_attention(self, 
                                   texts: List[str],
                                   template_type: str = 'positive',
                                   max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Analyze attention patterns around verbalizer tokens.
        
        Args:
            texts: List of input texts
            template_type: Type of template to use
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with attention analysis
        """
        with torch.no_grad():
            inputs = self.create_verbalizer_input(texts, template_type, max_length)
            
            # Get transformer outputs with attention
            outputs = self.transformer(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True,
                return_dict=True
            )
            
            # Get attention weights from last layer
            last_layer_attention = outputs.attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
            
            # Average over heads
            avg_attention = last_layer_attention.mean(dim=1)  # [batch_size, seq_len, seq_len]
            
            # Extract attention to/from verbalizer positions
            batch_size = avg_attention.size(0)
            verbalizer_attention = []
            
            for i in range(batch_size):
                pos = inputs['verbalizer_positions'][i].item()
                # Attention from verbalizer token to all other tokens
                attention_from_verbalizer = avg_attention[i, pos, :]
                # Attention to verbalizer token from all other tokens
                attention_to_verbalizer = avg_attention[i, :, pos]
                
                verbalizer_attention.append({
                    'from_verbalizer': attention_from_verbalizer,
                    'to_verbalizer': attention_to_verbalizer
                })
            
            return {
                'verbalizer_attention': verbalizer_attention,
                'full_attention': avg_attention,
                'verbalizer_positions': inputs['verbalizer_positions']
            }
    
    def predict_proba(self, 
                     texts: List[str],
                     template_type: str = 'positive',
                     max_length: int = 512) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            texts: List of input texts
            template_type: Template type to use
            max_length: Maximum sequence length
            
        Returns:
            Probabilities [batch_size]
        """
        with torch.no_grad():
            inputs = self.create_verbalizer_input(texts, template_type, max_length)
            logits = self.forward(**inputs)
            probabilities = torch.sigmoid(logits.squeeze())
            return probabilities
    
    def predict(self, 
               texts: List[str],
               template_type: str = 'positive',
               max_length: int = 512,
               threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions.
        
        Args:
            texts: List of input texts
            template_type: Template type to use
            max_length: Maximum sequence length
            threshold: Classification threshold
            
        Returns:
            Binary predictions [batch_size]
        """
        probabilities = self.predict_proba(texts, template_type, max_length)
        return (probabilities > threshold).long()
    
    def get_model_info(self) -> dict:
        """
        Get model information for logging/debugging.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        
        return {
            'model_type': 'VerbalizerClassifier',
            'base_model': self.model_name,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'freeze_transformer': self.freeze_transformer,
            'use_pooling': self.use_pooling,
            'pooling_strategy': self.pooling_strategy,
            'positive_token': self.positive_token,
            'negative_token': self.negative_token,
            'positive_token_id': self.positive_token_id,
            'negative_token_id': self.negative_token_id,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'transformer_parameters': transformer_params,
            'transformer_trainable': not self.freeze_transformer
        }


class MultiTemplateVerbalizerClassifier(VerbalizerClassifier):
    """
    Extended verbalizer classifier that uses multiple templates for robust predictions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Extended templates
        self.templates = {
            'positive': [
                " This statement is positive",
                " This review is positive", 
                " The sentiment is positive",
                " This expresses positive sentiment"
            ],
            'negative': [
                " This statement is negative",
                " This review is negative",
                " The sentiment is negative", 
                " This expresses negative sentiment"
            ]
        }
    
    def forward_multi_template(self, 
                             texts: List[str],
                             max_length: int = 512) -> torch.Tensor:
        """
        Forward pass using multiple templates and averaging results.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Averaged logits [batch_size, num_classes]
        """
        all_logits = []
        
        # Process with all positive templates
        for template in self.templates['positive']:
            # Temporarily override template
            original_template = self.templates['positive']
            self.templates['positive'] = template
            
            inputs = self.create_verbalizer_input(texts, 'positive', max_length)
            logits = self.forward(**inputs)
            all_logits.append(logits)
            
            # Restore original templates
            self.templates['positive'] = original_template
        
        # Average all predictions
        avg_logits = torch.stack(all_logits).mean(dim=0)
        return avg_logits
