"""
Training modules for sentiment analysis models.

This module provides trainers for all three model approaches with
support for custom loss functions and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path

from ..utils.helpers import save_model, AverageMeter, EarlyStopping


class SingleModelTrainer:
    """
    Trainer for individual models with custom loss function support.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: nn.Module,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 device: Optional[torch.device] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer (optional, will create Adam if not provided)
            scheduler: Learning rate scheduler (optional)
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer setup
        if optimizer is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler setup
        self.scheduler = scheduler
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def _get_model_predictions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get model predictions from batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Model predictions
        """
        # Handle different model types
        if hasattr(self.model, 'create_verbalizer_input'):
            # Verbalizer model
            return self.model(**batch)
        else:
            # LSTM models
            logits, _ = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                lengths=batch.get('lengths')
            )
            return logits
    
    def _compute_loss(self, predictions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss with custom loss function.
        
        Args:
            predictions: Model predictions
            batch: Batch of data
            
        Returns:
            Computed loss
        """
        targets = batch['labels']
        lengths = batch.get('lengths')
        
        # Use custom loss function
        if lengths is not None:
            loss = self.loss_fn(predictions, targets, lengths=lengths)
        else:
            loss = self.loss_fn(predictions, targets)
        
        return loss
    
    def _compute_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute accuracy.
        
        Args:
            predictions: Model predictions (logits)
            targets: Target labels
            
        Returns:
            Accuracy
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        
        probs = torch.sigmoid(predictions)
        pred_labels = (probs > 0.5).long()
        
        if targets.dtype == torch.float:
            targets = targets.long()
        
        correct = (pred_labels == targets).sum().item()
        total = targets.size(0)
        
        return correct / total
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self._get_model_predictions(batch)
            loss = self._compute_loss(predictions, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute accuracy
            accuracy = self._compute_accuracy(predictions, batch['labels'])
            
            # Update meters
            batch_size = batch['labels'].size(0)
            train_loss_meter.update(loss.item(), batch_size)
            train_acc_meter.update(accuracy, batch_size)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{train_loss_meter.avg:.4f}",
                'Acc': f"{train_acc_meter.avg:.4f}"
            })
        
        return {
            'train_loss': train_loss_meter.avg,
            'train_acc': train_acc_meter.avg
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self._get_model_predictions(batch)
                loss = self._compute_loss(predictions, batch)
                
                # Compute accuracy
                accuracy = self._compute_accuracy(predictions, batch['labels'])
                
                # Update meters
                batch_size = batch['labels'].size(0)
                val_loss_meter.update(loss.item(), batch_size)
                val_acc_meter.update(accuracy, batch_size)
        
        return {
            'val_loss': val_loss_meter.avg,
            'val_acc': val_acc_meter.avg
        }
    
    def train(self,
              num_epochs: int,
              save_best: bool = True,
              checkpoint_dir: Optional[str] = None,
              early_stopping_patience: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save the best model
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Early stopping patience (optional)
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        # Early stopping setup
        early_stopping = None
        if early_stopping_patience is not None:
            early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['train_acc'].append(train_metrics['train_acc'])
            self.history['val_acc'].append(val_metrics['val_acc'])
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}")
            print(f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Save best model
            if save_best and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                if checkpoint_dir is not None:
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    save_path = Path(checkpoint_dir) / 'best_model.pt'
                    save_model(
                        self.model,
                        str(save_path),
                        self.optimizer,
                        epoch + 1,
                        val_metrics['val_loss'],
                        metadata={
                            'train_acc': train_metrics['train_acc'],
                            'val_acc': val_metrics['val_acc'],
                            'epoch': epoch + 1
                        }
                    )
                    print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
            
            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_metrics['val_loss'], self.model):
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                    break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.history


class MultiModelTrainer:
    """
    Trainer for comparing multiple models.
    """
    
    def __init__(self,
                 models: Dict[str, nn.Module],
                 data_loaders: Dict[str, tuple],
                 loss_fn: nn.Module,
                 device: Optional[torch.device] = None):
        """
        Initialize multi-model trainer.
        
        Args:
            models: Dictionary of models to train
            data_loaders: Dictionary of data loaders for each model type
            loss_fn: Loss function
            device: Device to train on
        """
        self.models = models
        self.data_loaders = data_loaders
        self.loss_fn = loss_fn
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training results
        self.results = {}
    
    def train_all_models(self,
                        num_epochs: int = 3,
                        learning_rate: float = 1e-3,
                        save_checkpoints: bool = True) -> Dict[str, Dict]:
        """
        Train all models.
        
        Args:
            num_epochs: Number of epochs to train each model
            learning_rate: Learning rate
            save_checkpoints: Whether to save model checkpoints
            
        Returns:
            Dictionary with results for each model
        """
        print("Training all models...")
        
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {model_name.upper()}")
            print(f"{'='*50}")
            
            try:
                # Get appropriate data loaders
                if 'lstm' in model_name:
                    train_loader, val_loader, test_loader = self.data_loaders['lstm']
                else:
                    train_loader, val_loader, test_loader = self.data_loaders['verbalizer']
                
                # Create trainer
                trainer = SingleModelTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    loss_fn=self.loss_fn,
                    learning_rate=learning_rate,
                    device=self.device
                )
                
                # Train model
                start_time = time.time()
                history = trainer.train(
                    num_epochs=num_epochs,
                    save_best=save_checkpoints,
                    checkpoint_dir=f'./checkpoints/{model_name}' if save_checkpoints else None
                )
                training_time = time.time() - start_time
                
                # Store results
                self.results[model_name] = {
                    'history': history,
                    'training_time': training_time,
                    'final_train_loss': history['train_loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'final_train_acc': history['train_acc'][-1],
                    'final_val_acc': history['val_acc'][-1],
                    'best_val_loss': min(history['val_loss']),
                    'best_val_acc': max(history['val_acc'])
                }
                
                print(f"Training completed for {model_name}")
                print(f"Best Val Acc: {self.results[model_name]['best_val_acc']:.4f}")
                print(f"Training Time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
        
        return self.results
    
    def print_comparison(self):
        """Print comparison of all models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        print(f"{'Model':<20} {'Best Val Acc':<12} {'Best Val Loss':<12} {'Time (s)':<10}")
        print("-" * 60)
        
        for model_name, results in self.results.items():
            if 'error' not in results:
                acc = results['best_val_acc']
                loss = results['best_val_loss']
                time_taken = results['training_time']
                print(f"{model_name:<20} {acc:<12.4f} {loss:<12.4f} {time_taken:<10.1f}")
            else:
                print(f"{model_name:<20} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10}")


class AdversarialTrainer(SingleModelTrainer):
    """
    Trainer with adversarial training capabilities.
    """
    
    def __init__(self, *args, adversarial_eps: float = 0.01, **kwargs):
        """
        Initialize adversarial trainer.
        
        Args:
            adversarial_eps: Epsilon for adversarial perturbations
        """
        super().__init__(*args, **kwargs)
        self.adversarial_eps = adversarial_eps
    
    def generate_adversarial_examples(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generate adversarial examples using FGSM.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch with adversarial examples
        """
        # Only works with embedding-based models
        if not hasattr(self.model, 'get_embeddings'):
            return batch
        
        # Get embeddings and enable gradients
        input_ids = batch['input_ids'].clone().detach().requires_grad_(True)
        embeddings = self.model.get_embeddings(input_ids)
        embeddings.requires_grad_(True)
        
        # Forward pass
        predictions = self._get_model_predictions({**batch, 'input_ids': input_ids})
        loss = self._compute_loss(predictions, batch)
        
        # Backward pass to get gradients
        loss.backward()
        
        # Generate adversarial perturbation
        grad_sign = embeddings.grad.sign()
        adversarial_embeddings = embeddings + self.adversarial_eps * grad_sign
        
        # Create adversarial batch (this is a simplified version)
        # In practice, you'd need to map back to token space
        return batch
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train epoch with adversarial examples.
        
        Returns:
            Training metrics
        """
        self.model.train()
        
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        
        progress_bar = tqdm(self.train_loader, desc="Adversarial Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Normal training step
            predictions = self._get_model_predictions(batch)
            loss = self._compute_loss(predictions, batch)
            
            # Adversarial training step (simplified)
            # In practice, you'd implement proper adversarial example generation
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Compute accuracy
            accuracy = self._compute_accuracy(predictions, batch['labels'])
            
            # Update meters
            batch_size = batch['labels'].size(0)
            train_loss_meter.update(loss.item(), batch_size)
            train_acc_meter.update(accuracy, batch_size)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{train_loss_meter.avg:.4f}",
                'Acc': f"{train_acc_meter.avg:.4f}"
            })
        
        return {
            'train_loss': train_loss_meter.avg,
            'train_acc': train_acc_meter.avg
        }
