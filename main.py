"""
Main entry point for the sentiment analysis challenge.

This script demonstrates two model approaches:
1. Pure LSTM with learned embeddings
2. Verbalizer approach with pre-computed ModernBERT embeddings

Usage:
    python main.py --mode demo    # Quick demo with small dataset
    python main.py --mode full    # Full training and evaluation
    python main.py --mode compare # Compare both models
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Any

# Import our modules
from src.data.preprocessing import (
    load_imdb_dataset, 
    create_train_val_split, 
    build_vocabulary, 
    TextPreprocessor,
    get_dataset_statistics,
    extract_verbalizer_embeddings
)
from src.data.dataset import LSTMDataset
from src.data.verbalizer_dataset import VerbalizerDataset
from src.data.preprocessing import create_data_loaders
from src.models.lstm_pure import PureLSTMClassifier
from src.models.verbalizer import VerbalizerClassifier
from src.models.loss_functions import UniversalCustomLoss
from src.utils.helpers import set_seed, create_directories
from src.training.trainer import SingleModelTrainer
from src.training.evaluator import ModelEvaluator


def setup_experiment(args: argparse.Namespace) -> torch.device:
    """Setup experiment configuration and directories."""
    print("=" * 60)
    print("SENTIMENT ANALYSIS: LSTM vs VERBALIZER")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create necessary directories
    create_directories(['checkpoints/lstm', 'checkpoints/verbalizer', 'results', 'cache'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    
    return device


def load_and_prepare_data(args: argparse.Namespace) -> Dict[str, Tuple[List[str], List[int]]]:
    """Load and prepare IMDB dataset."""
    print("\n" + "=" * 50)
    print("LOADING AND PREPARING DATA")
    print("=" * 50)
    
    # Load IMDB dataset
    data = load_imdb_dataset(cache_dir='./cache')
    
    # Use subset for demo mode
    if args.mode == 'demo':
        print("Demo mode: Using balanced subset of data")
        # Get balanced subset by taking equal numbers from each class
        train_pos_indices = [i for i, label in enumerate(data['train_labels']) if label == 1][:500]
        train_neg_indices = [i for i, label in enumerate(data['train_labels']) if label == 0][:500]
        train_indices = train_pos_indices + train_neg_indices
        
        test_pos_indices = [i for i, label in enumerate(data['test_labels']) if label == 1][:100]
        test_neg_indices = [i for i, label in enumerate(data['test_labels']) if label == 0][:100]
        test_indices = test_pos_indices + test_neg_indices
        
        train_texts = [data['train_texts'][i] for i in train_indices]
        train_labels = [data['train_labels'][i] for i in train_indices]
        test_texts = [data['test_texts'][i] for i in test_indices]
        test_labels = [data['test_labels'][i] for i in test_indices]
    else:
        train_texts = data['train_texts']
        train_labels = data['train_labels']
        test_texts = data['test_texts']
        test_labels = data['test_labels']
    
    # Create train/validation split
    train_texts, train_labels, val_texts, val_labels = create_train_val_split(
        train_texts, train_labels, val_ratio=0.1, random_seed=args.seed
    )
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    train_stats = get_dataset_statistics(train_texts, train_labels)
    val_stats = get_dataset_statistics(val_texts, val_labels)
    test_stats = get_dataset_statistics(test_texts, test_labels)
    
    for split, stats in [("Train", train_stats), ("Val", val_stats), ("Test", test_stats)]:
        print(f"{split}: {stats['num_samples']:,} samples, "
              f"{stats['positive_ratio']:.1%} positive, "
              f"avg length: {stats['avg_length']:.1f} words")
    
    return {
        'train': (train_texts, train_labels),
        'val': (val_texts, val_labels),
        'test': (test_texts, test_labels)
    }


def prepare_lstm_data(data: Dict[str, Tuple[List[str], List[int]]], args: argparse.Namespace):
    """Prepare vocabulary and datasets for LSTM model."""
    print("\n" + "=" * 50)
    print("PREPARING LSTM DATA")
    print("=" * 50)
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    test_texts, test_labels = data['test']
    
    # Build vocabulary for LSTM
    print("Building vocabulary for LSTM...")
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)
    vocab = build_vocabulary(
        train_texts, 
        min_freq=2, 
        max_vocab_size=args.vocab_size,
        preprocessor=preprocessor
    )
    
    print(f"Vocabulary size: {len(vocab):,}")
    
    # Create LSTM datasets
    print("Creating LSTM datasets...")
    train_dataset = LSTMDataset(
        train_texts, train_labels, 
        vocab=vocab,
        max_length=args.max_length,
        preprocessor=preprocessor
    )
    
    val_dataset = LSTMDataset(
        val_texts, val_labels,
        vocab=vocab,
        max_length=args.max_length,
        preprocessor=preprocessor
    )
    
    test_dataset = LSTMDataset(
        test_texts, test_labels,
        vocab=vocab,
        max_length=args.max_length,
        preprocessor=preprocessor
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    return vocab, (train_loader, val_loader, test_loader)


def prepare_verbalizer_data(data: Dict[str, Tuple[List[str], List[int]]], args: argparse.Namespace):
    """Prepare embeddings and datasets for Verbalizer model."""
    print("\n" + "=" * 50)
    print("PREPARING VERBALIZER DATA")
    print("=" * 50)
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    test_texts, test_labels = data['test']
    
    # Extract embeddings for each split
    print("Extracting training embeddings...")
    train_embeddings, train_labels_tensor = extract_verbalizer_embeddings(
        train_texts, train_labels,
        max_samples=None,
        cache_file=f"verbalizer_train_{args.mode}.pkl",
        cache_dir='./cache',
        max_length=args.max_length,
        random_seed=args.seed
    )
    
    print("Extracting validation embeddings...")
    val_embeddings, val_labels_tensor = extract_verbalizer_embeddings(
        val_texts, val_labels,
        max_samples=None,
        cache_file=f"verbalizer_val_{args.mode}.pkl",
        cache_dir='./cache',
        max_length=args.max_length,
        random_seed=args.seed + 1
    )
    
    print("Extracting test embeddings...")
    test_embeddings, test_labels_tensor = extract_verbalizer_embeddings(
        test_texts, test_labels,
        max_samples=None,
        cache_file=f"verbalizer_test_{args.mode}.pkl",
        cache_dir='./cache',
        max_length=args.max_length,
        random_seed=args.seed + 2
    )
    
    # Create verbalizer datasets
    print("Creating verbalizer datasets...")
    train_dataset = VerbalizerDataset(train_embeddings, train_labels_tensor)
    val_dataset = VerbalizerDataset(val_embeddings, val_labels_tensor)
    test_dataset = VerbalizerDataset(test_embeddings, test_labels_tensor)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    return (train_loader, val_loader, test_loader)


def create_models(vocab_size: int, embedding_dim: int, args: argparse.Namespace, device: torch.device):
    """Create LSTM and Verbalizer models."""
    print("\n" + "=" * 50)
    print("CREATING MODELS")
    print("=" * 50)
    
    models = {}
    
    # 1. Pure LSTM Model
    print("Creating Pure LSTM model...")
    models['lstm'] = PureLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True,
        use_attention=True
    ).to(device)
    
    # 2. Verbalizer Model
    print("Creating Verbalizer model...")
    models['verbalizer'] = VerbalizerClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Print model information
    for name, model in models.items():
        info = model.get_model_info()
        print(f"\n{name.upper()}:")
        print(f"  Model Type: {info['model_type']}")
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Trainable parameters: {info['trainable_parameters']:,}")
        if 'description' in info:
            print(f"  Description: {info['description']}")
    
    return models


def train_single_model(model: nn.Module, model_name: str, data_loaders: Tuple, 
                      args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Train a single model."""
    print(f"\n{'='*20} TRAINING {model_name.upper()} {'='*20}")
    
    train_loader, val_loader, test_loader = data_loaders
    
    # Create custom loss function
    custom_loss = UniversalCustomLoss(
        confidence_penalty=args.confidence_penalty,
        length_weight=args.length_weight
    )
    
    # Create trainer
    trainer = SingleModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=custom_loss,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Train model
    start_time = time.time()
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_best=True,
        checkpoint_dir=f'./checkpoints/{model_name}'
    )
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    
    # Evaluate on test set
    evaluator = ModelEvaluator(model, device)
    test_results = evaluator.evaluate(test_loader)
    
    print(f"Test Results for {model_name}:")
    print(f"  Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"  F1 Score: {test_results['f1']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")
    
    return {
        'history': history,
        'test_results': test_results,
        'training_time': training_time,
        'model_info': model.get_model_info()
    }


def compare_models(models: Dict[str, nn.Module], lstm_loaders: Tuple, verbalizer_loaders: Tuple,
                  args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Compare LSTM and Verbalizer models."""
    print("\n" + "=" * 50)
    print("COMPARING LSTM vs VERBALIZER")
    print("=" * 50)
    
    results = {}
    
    # Train LSTM
    try:
        results['lstm'] = train_single_model(
            models['lstm'], 'lstm', lstm_loaders, args, device
        )
    except Exception as e:
        print(f"Error training LSTM: {e}")
        results['lstm'] = {'error': str(e)}
    
    # Train Verbalizer
    try:
        results['verbalizer'] = train_single_model(
            models['verbalizer'], 'verbalizer', verbalizer_loaders, args, device
        )
    except Exception as e:
        print(f"Error training Verbalizer: {e}")
        results['verbalizer'] = {'error': str(e)}
    
    # Print comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"{'Model':<15} {'Accuracy':<10} {'F1':<10} {'Time (min)':<12} {'Parameters':<12}")
    print("-" * 70)
    
    for model_name, result in results.items():
        if 'error' not in result:
            acc = result['test_results']['accuracy']
            f1 = result['test_results']['f1']
            time_taken = result['training_time'] / 60
            params = result['model_info']['trainable_parameters']
            print(f"{model_name:<15} {acc:<10.4f} {f1:<10.4f} {time_taken:<12.1f} {params:<12,}")
        else:
            print(f"{model_name:<15} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<12}")
    
    # Save results
    results_file = Path('./results/lstm_vs_verbalizer_comparison.json')
    results_file.parent.mkdir(exist_ok=True)
    
    # Convert tensors to floats for JSON serialization
    json_results = {}
    for model_name, result in results.items():
        if 'error' not in result:
            json_results[model_name] = {
                'test_accuracy': float(result['test_results']['accuracy']),
                'test_f1': float(result['test_results']['f1']),
                'test_precision': float(result['test_results']['precision']),
                'test_recall': float(result['test_results']['recall']),
                'training_time_seconds': result['training_time'],
                'training_time_minutes': result['training_time'] / 60,
                'model_info': result['model_info']
            }
        else:
            json_results[model_name] = result
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='LSTM vs Verbalizer Sentiment Analysis')
    
    # Mode selection
    parser.add_argument('--mode', choices=['demo', 'full', 'compare'], default='demo',
                       help='Execution mode: demo (small dataset), full (complete), compare (both models)')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=200, 
                       help='Embedding dimension for LSTM (default: 200)')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                       help='Hidden dimension (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2, 
                       help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3, 
                       help='Dropout probability (default: 0.3)')
    parser.add_argument('--vocab_size', type=int, default=20000, 
                       help='Vocabulary size for LSTM (default: 20000)')
    parser.add_argument('--max_length', type=int, default=256, 
                       help='Maximum sequence length (default: 256)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=5, 
                       help='Number of epochs (default: 5)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                       help='Learning rate (default: 0.001)')
    
    # Custom loss parameters
    parser.add_argument('--confidence_penalty', type=float, default=2.0, 
                       help='Confidence penalty for custom loss (default: 2.0)')
    parser.add_argument('--length_weight', type=float, default=0.1,
                       help='Length weight for custom loss (default: 0.1)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Setup experiment
    device = setup_experiment(args)
    
    # Load and prepare data
    data = load_and_prepare_data(args)
    
    # Prepare data for both models
    vocab, lstm_loaders = prepare_lstm_data(data, args)
    verbalizer_loaders = prepare_verbalizer_data(data, args)
    
    # Create models
    embedding_dim = 768  # ModernBERT embedding dimension
    models = create_models(len(vocab), embedding_dim, args, device)
    
    # Execute based on mode
    if args.mode == 'demo':
        print("\n" + "=" * 50)
        print("DEMO MODE - TRAINING LSTM ONLY")
        print("=" * 50)
        
        # Train just LSTM for demo
        result = train_single_model(models['lstm'], 'lstm', lstm_loaders, args, device)
        
    elif args.mode == 'compare':
        # Compare both models
        results = compare_models(models, lstm_loaders, verbalizer_loaders, args, device)
        
    elif args.mode == 'full':
        # Train both models individually
        print("\n" + "=" * 50)
        print("FULL MODE - TRAINING BOTH MODELS")
        print("=" * 50)
        
        lstm_result = train_single_model(models['lstm'], 'lstm', lstm_loaders, args, device)
        verbalizer_result = train_single_model(models['verbalizer'], 'verbalizer', verbalizer_loaders, args, device)
    
    print("\n" + "=" * 60)
    print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Available models:")
    print("  • LSTM: Traditional approach with learned embeddings")
    print("  • Verbalizer: Modern approach with pre-computed ModernBERT embeddings")
    print("\nUse individual training scripts for more control:")
    print("  • python train_lstm_only.py")
    print("  • python train_verbalizer_only.py")


if __name__ == "__main__":
    main()
