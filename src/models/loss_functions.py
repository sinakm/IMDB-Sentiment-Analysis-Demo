"""
Custom loss functions for sentiment analysis.

This module implements custom loss functions that address sentiment classification
nuances through confidence-based weighting and review length considerations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentWeightedLoss(nn.Module):
    """
    Custom loss function for sentiment analysis that weights samples differently
    based on specific criteria (as provided in the problem statement).
    """
    
    def __init__(self, weight_param=1.0):
        super(SentimentWeightedLoss, self).__init__()
        self.weight_param = weight_param
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets, sample_weights=None):
        """
        Forward pass with custom weighting logic.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels
            sample_weights: Optional sample weights
            
        Returns:
            Weighted loss value
        """
        # Basic loss computation
        base_loss = self.base_loss(predictions.squeeze(), targets.float())

        # Apply custom weighting logic
        if sample_weights is not None:
            weighted_loss = base_loss * sample_weights * self.weight_param
        else:
            weighted_loss = base_loss * self.weight_param

        return weighted_loss.mean()


class UniversalCustomLoss(nn.Module):
    """
    Advanced custom loss function that works across all three model types.
    
    This loss function implements:
    1. Confidence-based weighting: Penalizes highly confident wrong predictions more
    2. Length-based weighting: Adjusts loss based on review length
    3. Class balancing: Optional class weights for imbalanced datasets
    
    The rationale is that confident mistakes are more problematic than uncertain
    mistakes, and longer reviews may contain more nuanced sentiment information.
    """
    
    def __init__(self, 
                 confidence_penalty=2.0, 
                 length_weight=0.1, 
                 class_weights=None,
                 temperature=1.0):
        """
        Initialize the custom loss function.
        
        Args:
            confidence_penalty: How much to penalize confident wrong predictions
            length_weight: Weight factor for review length influence
            class_weights: Optional class weights [neg_weight, pos_weight]
            temperature: Temperature scaling for confidence calculation
        """
        super(UniversalCustomLoss, self).__init__()
        self.confidence_penalty = confidence_penalty
        self.length_weight = length_weight
        self.temperature = temperature
        
        # Base loss function
        if class_weights is not None:
            # Convert to tensor if needed
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.base_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=class_weights[1])
        else:
            self.base_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, targets, lengths=None, **kwargs):
        """
        Forward pass with comprehensive weighting.
        
        Args:
            predictions: Model predictions (logits) [batch_size, 1] or [batch_size]
            targets: Ground truth labels [batch_size]
            lengths: Optional review lengths [batch_size]
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            Weighted loss value
        """
        # Ensure predictions are properly shaped
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        
        # Calculate base loss
        base_loss = self.base_loss(predictions, targets.float())
        
        # 1. Length-based weighting
        if lengths is not None:
            # Normalize lengths and apply weighting
            max_length = lengths.max().float()
            normalized_lengths = lengths.float() / max_length
            length_weights = 1.0 + self.length_weight * normalized_lengths
        else:
            length_weights = torch.ones_like(base_loss)
        
        # 2. Confidence-based weighting
        # Apply temperature scaling for better confidence estimation
        scaled_predictions = predictions / self.temperature
        probs = torch.sigmoid(scaled_predictions)
        
        # Calculate confidence (distance from 0.5, scaled to [0, 1])
        confidence = torch.abs(probs - 0.5) * 2.0
        
        # Identify wrong predictions
        predicted_classes = (probs > 0.5).float()
        wrong_predictions = (predicted_classes != targets.float())
        
        # Apply confidence penalty to wrong predictions
        confidence_weights = torch.where(
            wrong_predictions,
            1.0 + self.confidence_penalty * confidence,
            torch.ones_like(confidence)
        )
        
        # 3. Combine all weights
        total_weights = length_weights * confidence_weights
        weighted_loss = base_loss * total_weights
        
        return weighted_loss.mean()
    
    def get_confidence_stats(self, predictions, targets):
        """
        Get statistics about prediction confidence for analysis.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Dictionary with confidence statistics
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
            
        probs = torch.sigmoid(predictions / self.temperature)
        confidence = torch.abs(probs - 0.5) * 2.0
        
        predicted_classes = (probs > 0.5).float()
        correct_predictions = (predicted_classes == targets.float())
        
        return {
            'avg_confidence': confidence.mean().item(),
            'avg_confidence_correct': confidence[correct_predictions].mean().item() if correct_predictions.sum() > 0 else 0,
            'avg_confidence_wrong': confidence[~correct_predictions].mean().item() if (~correct_predictions).sum() > 0 else 0,
            'high_confidence_wrong': ((confidence > 0.8) & (~correct_predictions)).sum().item(),
            'total_wrong': (~correct_predictions).sum().item()
        }


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    This is an alternative loss function that can be used for comparison.
    Focal Loss focuses learning on hard examples by down-weighting easy examples.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Forward pass for Focal Loss.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
            
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets.float(), reduction='none'
        )
        
        # Calculate probabilities
        probs = torch.sigmoid(predictions)
        
        # Calculate focal weight
        # For positive examples: (1 - p)^gamma
        # For negative examples: p^gamma
        focal_weight = torch.where(
            targets == 1,
            (1 - probs) ** self.gamma,
            probs ** self.gamma
        )
        
        # Apply alpha weighting
        alpha_weight = torch.where(
            targets == 1,
            self.alpha,
            1 - self.alpha
        )
        
        # Combine weights
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
