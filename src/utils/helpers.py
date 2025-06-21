"""
Utility helper functions for the sentiment analysis project.
"""

import torch
import numpy as np
import random
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import pickle


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(dirs: list):
    """
    Create directories if they don't exist.
    
    Args:
        dirs: List of directory paths to create
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_model(model: torch.nn.Module, 
               filepath: str, 
               optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: Optional[int] = None,
               loss: Optional[float] = None,
               metadata: Optional[Dict[str, Any]] = None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model
        optimizer: Optimizer state to save (optional)
        epoch: Current epoch (optional)
        loss: Current loss (optional)
        metadata: Additional metadata (optional)
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(model: torch.nn.Module,
               filepath: str,
               optimizer: Optional[torch.optim.Optimizer] = None,
               device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        filepath: Path to the saved model
        optimizer: Optimizer to load state into (optional)
        device: Device to load the model on (optional)
        
    Returns:
        Dictionary with loaded checkpoint information
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath}")
    
    return {
        'epoch': checkpoint.get('epoch'),
        'loss': checkpoint.get('loss'),
        'metadata': checkpoint.get('metadata', {}),
        'model_info': checkpoint.get('model_info', {})
    }


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the JSON file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"Data saved to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def save_pickle(data: Any, filepath: str):
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save the pickle file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data saved to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_device_info() -> Dict[str, Any]:
    """
    Get device information.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if torch.cuda.is_available():
        device_info['device_name'] = torch.cuda.get_device_name()
        device_info['memory_allocated'] = torch.cuda.memory_allocated()
        device_info['memory_reserved'] = torch.cuda.memory_reserved()
    
    return device_info


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


def get_memory_usage() -> Dict[str, float]:
    """
    Get memory usage information.
    
    Returns:
        Dictionary with memory usage in MB
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    usage = {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
    }
    
    if torch.cuda.is_available():
        usage['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        usage['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    
    return usage


def print_model_summary(model: torch.nn.Module, input_size: tuple = None):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size: Input size for the model (optional)
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    # Model info
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    
    # Parameter count
    param_info = count_parameters(model)
    print(f"\nParameter Count:")
    for key, value in param_info.items():
        print(f"  {key}: {value:,}")
    
    # Model architecture
    print(f"\nModel Architecture:")
    print(model)
    
    print("=" * 60)


def ensure_dir_exists(filepath: str) -> str:
    """
    Ensure directory exists for a given filepath.
    
    Args:
        filepath: File path
        
    Returns:
        The same filepath
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    return filepath


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class AverageMeter:
    """Utility class to track running averages."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update the meter.
        
        Args:
            val: Value to add
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
