"""
Main entry point for the sentiment analysis challenge.

This script demonstrates all three model approaches:
1. Pure LSTM with learned embeddings
2. LSTM with pre-trained GloVe embeddings
3. Verbalizer/PET approach with transformer embeddings

Usage:
    python main.py --mode demo    # Quick demo with small dataset
    python main.py --mode full    # Full training and evaluation
    python main.py --mode compare # Compare all three models
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union

# Import our modules
from src.data.preprocessing import (
    load_imdb_dataset, 
    create_train_val_split, 
    build_vocabulary, 
    TextPreprocessor,
    get_dataset_statistics
)
from src.data.dataset import DatasetFactory
from src.data.modernbert_dataset import create_modernbert_datasets
from src.data.preprocessing import create_data_loaders
from src.models.lstm_pure import PureLSTMClassifier
from src.models.modernbert_classifier import ModernBERTClassifier
from src.models.verbalizer import VerbalizerClassifier
from src.models.loss_functions import UniversalCustomLoss, SentimentWeightedLoss
from src.utils.helpers import set_seed, save_model, create_directories
from src.training.trainer import SingleModelTrainer
from src.training.evaluator import ModelEvaluator


def setup_experiment(args: argparse.Namespace) -> torch.device:
    """Setup experiment configuration and directories.
    
    Args:
        args: Command line arguments containing seed and other config
        
    Returns:
        torch.device: The device (CPU/CUDA) to use for training
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create necessary directories
    create_directories(['checkpoints', 'results', 'cache'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return device


def load_and_prepare_data(args: argparse.Namespace) -> Dict[str, Tuple[List[str], List[int]]]:
    """Load and prepare IMDB dataset.
    
    Args:
        args: Command line arguments containing mode and other config
        
    Returns:
        Dict containing train/val/test splits with texts and labels
        Format: {'train': (texts, labels), 'val': (texts, labels), 'test': (texts, labels)}
    """
    print("=" * 50)
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
        print(f"{split}: {stats['num_samples']} samples, "
              f"{stats['positive_ratio']:.1%} positive, "
              f"avg length: {stats['avg_length']:.1f} words")
    
    return {
        'train': (train_texts, train_labels),
        'val': (val_texts, val_labels),
        'test': (test_texts, test_labels)
    }


def prepare_vocabularies_and_embeddings(
    data: Dict[str, Tuple[List[str], List[int]]], 
    args: argparse.Namespace
) -> Tuple[Dict[str, int], Optional[Any], TextPreprocessor]:
    """Prepare vocabularies for LSTM models and ModernBERT setup.
    
    Args:
        data: Dictionary containing train/val/test splits with texts and labels
        args: Command line arguments containing vocab_size and other config
        
    Returns:
        Tuple of (vocabulary dict, unused_embeddings, text_preprocessor)
    """
    print("\n" + "=" * 50)
    print("PREPARING VOCABULARIES AND MODERNBERT")
    print("=" * 50)
    
    train_texts, train_labels = data['train']
    
    # Build vocabulary for pure LSTM
    print("Building vocabulary for LSTM...")
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)
    vocab = build_vocabulary(
        train_texts, 
        min_freq=2, 
        max_vocab_size=args.vocab_size,
        preprocessor=preprocessor
    )
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # ModernBERT will handle its own tokenization
    print("ModernBERT will use its own tokenizer - no pre-loading needed")
    
    return vocab, None, preprocessor


def create_models(
    vocab: Dict[str, int], 
    unused_embeddings: Optional[Any], 
    args: argparse.Namespace, 
    device: torch.device
) -> Dict[str, nn.Module]:
    """Create all three model instances.
    
    Args:
        vocab: Vocabulary dictionary mapping words to indices
        unused_embeddings: Placeholder for embeddings (None for ModernBERT approach)
        args: Command line arguments containing model hyperparameters
        device: Device to place models on (CPU/CUDA)
        
    Returns:
        Dictionary containing the three model instances
        Format: {'pure_lstm': model, 'modernbert': model, 'verbalizer': model}
    """
    print("\n" + "=" * 50)
    print("CREATING MODELS")
    print("=" * 50)
    
    models = {}
    
    # 1. Pure LSTM Model
    print("Creating Pure LSTM model...")
    models['pure_lstm'] = PureLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True,
        use_attention=True
    ).to(device)
    
    # 2. ModernBERT Model
    print("Creating ModernBERT classifier...")
    models['modernbert'] = ModernBERTClassifier(
        model_name="answerdotai/ModernBERT-base",
        num_classes=1,
        dropout=args.dropout,
        pooling_strategy="cls",
        use_adapter=False,
        freeze_encoder=False,
        max_length=args.max_length
    ).to(device)
    
    # 3. Verbalizer Model
    print("Creating Verbalizer model...")
    models['verbalizer'] = VerbalizerClassifier(
        model_name=args.transformer_model,
        dropout=args.dropout,
        freeze_transformer=args.freeze_transformer
    ).to(device)
    
    # Print model information
    for name, model in models.items():
        info = model.get_model_info()
        print(f"\n{name.upper()}:")
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    
    return models


def create_datasets_and_loaders(
    data: Dict[str, Tuple[List[str], List[int]]], 
    vocab: Dict[str, int], 
    preprocessor: TextPreprocessor, 
    args: argparse.Namespace
) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
    """Create datasets and data loaders for all models.
    
    Args:
        data: Dictionary containing train/val/test splits with texts and labels
        vocab: Vocabulary dictionary mapping words to indices
        preprocessor: Text preprocessing instance
        args: Command line arguments containing batch_size and other config
        
    Returns:
        Dictionary containing data loaders for each model type
        Format: {'lstm': (train_loader, val_loader, test_loader), 'modernbert': (...), 'verbalizer': (...)}
    """
    print("\n" + "=" * 50)
    print("CREATING DATASETS AND DATA LOADERS")
    print("=" * 50)
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    test_texts, test_labels = data['test']
    
    # Create datasets for all models
    datasets = DatasetFactory.create_all_datasets(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        vocab=vocab,
        tokenizer_name=args.transformer_model,
        max_length=args.max_length,
        preprocessor=preprocessor
    )
    
    # Create ModernBERT datasets
    modernbert_train, modernbert_val, modernbert_test = create_modernbert_datasets(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        tokenizer_name="answerdotai/ModernBERT-base",
        max_length=args.max_length
    )
    
    # Create data loaders
    data_loaders = {}
    
    # LSTM data loaders
    lstm_train, lstm_val, lstm_test = datasets['lstm']
    data_loaders['lstm'] = create_data_loaders(
        lstm_train, lstm_val, lstm_test,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # ModernBERT data loaders
    data_loaders['modernbert'] = create_data_loaders(
        modernbert_train, modernbert_val, modernbert_test,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Verbalizer data loaders
    verb_train, verb_val, verb_test = datasets['verbalizer']
    data_loaders['verbalizer'] = create_data_loaders(
        verb_train, verb_val, verb_test,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    print("Datasets and data loaders created successfully!")
    return data_loaders


def train_single_model(
    model: nn.Module, 
    model_name: str, 
    data_loaders: Dict[str, Tuple[DataLoader, DataLoader, DataLoader]], 
    args: argparse.Namespace, 
    device: torch.device
) -> Dict[str, Any]:
    """Train a single model.
    
    Args:
        model: PyTorch model to train
        model_name: Name of the model (used for selecting appropriate data loader)
        data_loaders: Dictionary containing data loaders for each model type
        args: Command line arguments containing training hyperparameters
        device: Device to train on (CPU/CUDA)
        
    Returns:
        Dictionary containing training history, test results, and training time
    """
    print(f"\n{'='*20} TRAINING {model_name.upper()} {'='*20}")
    
    # Get appropriate data loader
    if 'lstm' in model_name:
        train_loader, val_loader, test_loader = data_loaders['lstm']
    elif 'modernbert' in model_name:
        train_loader, val_loader, test_loader = data_loaders['modernbert']
    else:
        train_loader, val_loader, test_loader = data_loaders['verbalizer']
    
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
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    evaluator = ModelEvaluator(model, device)
    test_results = evaluator.evaluate(test_loader)
    
    print(f"Test Results for {model_name}:")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print(f"  F1 Score: {test_results['f1']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")
    
    return {
        'history': history,
        'test_results': test_results,
        'training_time': training_time
    }


def compare_models(
    models: Dict[str, nn.Module], 
    data_loaders: Dict[str, Tuple[DataLoader, DataLoader, DataLoader]], 
    args: argparse.Namespace, 
    device: torch.device
) -> Dict[str, Any]:
    """Compare all three models.
    
    Args:
        models: Dictionary containing the three model instances
        data_loaders: Dictionary containing data loaders for each model type
        args: Command line arguments containing training hyperparameters
        device: Device to train on (CPU/CUDA)
        
    Returns:
        Dictionary containing results for each model (or error info)
    """
    print("\n" + "=" * 50)
    print("COMPARING ALL MODELS")
    print("=" * 50)
    
    results = {}
    
    # Train each model
    for model_name, model in models.items():
        try:
            results[model_name] = train_single_model(
                model, model_name, data_loaders, args, device
            )
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Print comparison
    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS")
    print("=" * 50)
    
    print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'Time (s)':<10}")
    print("-" * 50)
    
    for model_name, result in results.items():
        if 'error' not in result:
            acc = result['test_results']['accuracy']
            f1 = result['test_results']['f1']
            time_taken = result['training_time']
            print(f"{model_name:<20} {acc:<10.4f} {f1:<10.4f} {time_taken:<10.1f}")
        else:
            print(f"{model_name:<20} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
    
    # Save results
    results_file = Path('./results/comparison_results.json')
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
                'training_time': result['training_time']
            }
        else:
            json_results[model_name] = result
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    return results


def demo_verbalizer_analysis(
    models: Dict[str, nn.Module], 
    data: Dict[str, Tuple[List[str], List[int]]], 
    args: argparse.Namespace
) -> None:
    """Demonstrate verbalizer analysis capabilities.
    
    Args:
        models: Dictionary containing the three model instances
        data: Dictionary containing train/val/test splits with texts and labels
        args: Command line arguments (unused but kept for consistency)
        
    Returns:
        None: This function prints analysis results directly
    """
    print("\n" + "=" * 50)
    print("VERBALIZER ANALYSIS DEMO")
    print("=" * 50)
    
    verbalizer_model = models['verbalizer']
    test_texts, test_labels = data['test']
    
    # Take a few examples for analysis
    sample_texts = test_texts[:3]
    sample_labels = test_labels[:3]
    
    print("Analyzing sample reviews with verbalizer approach...")
    
    for i, (text, label) in enumerate(zip(sample_texts, sample_labels)):
        print(f"\nReview {i+1} (True label: {'Positive' if label == 1 else 'Negative'}):")
        print(f"Text: {text[:100]}...")
        
        # Get verbalizer analysis
        analysis = verbalizer_model.predict_with_verbalizer_analysis([text])
        
        # Handle tensor indexing properly
        pred = analysis['predictions'].item() if analysis['predictions'].dim() == 0 else analysis['predictions'][0].item()
        pos_prob = analysis['positive_probs'].item() if analysis['positive_probs'].dim() == 0 else analysis['positive_probs'][0].item()
        neg_prob = analysis['negative_probs'].item() if analysis['negative_probs'].dim() == 0 else analysis['negative_probs'][0].item()
        consistency = analysis['consistency_score'].item() if analysis['consistency_score'].dim() == 0 else analysis['consistency_score'][0].item()
        
        print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
        print(f"Positive template probability: {pos_prob:.4f}")
        print(f"Negative template probability: {neg_prob:.4f}")
        print(f"Consistency score: {consistency:.4f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Challenge')
    
    # Mode selection
    parser.add_argument('--mode', choices=['demo', 'full', 'compare'], default='demo',
                       help='Execution mode')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=200, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')
    parser.add_argument('--vocab_size', type=int, default=20000, help='Vocabulary size')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    
    # Custom loss parameters
    parser.add_argument('--confidence_penalty', type=float, default=2.0, 
                       help='Confidence penalty for custom loss')
    parser.add_argument('--length_weight', type=float, default=0.1,
                       help='Length weight for custom loss')
    
    # Model-specific parameters
    parser.add_argument('--transformer_model', type=str, default='bert-base-uncased',
                       help='Transformer model for verbalizer')
    parser.add_argument('--freeze_transformer', action='store_true',
                       help='Freeze transformer weights')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("Sentiment Analysis Challenge - Three Model Approaches")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Setup experiment
    device = setup_experiment(args)
    
    # Load and prepare data
    data = load_and_prepare_data(args)
    
    # Prepare vocabularies and embeddings
    vocab, glove_embeddings, preprocessor = prepare_vocabularies_and_embeddings(data, args)
    
    # Create models
    models = create_models(vocab, glove_embeddings, args, device)
    
    # Create datasets and data loaders
    data_loaders = create_datasets_and_loaders(data, vocab, preprocessor, args)
    
    # Execute based on mode
    if args.mode == 'demo':
        print("\n" + "=" * 50)
        print("DEMO MODE - QUICK TRAINING")
        print("=" * 50)
        
        # Train just one model for demo
        model_name = 'pure_lstm'
        model = models[model_name]
        result = train_single_model(model, model_name, data_loaders, args, device)
        
        # Demo verbalizer analysis
        demo_verbalizer_analysis(models, data, args)
        
    elif args.mode == 'compare':
        # Compare all models
        results = compare_models(models, data_loaders, args, device)
        
    elif args.mode == 'full':
        # Full training and evaluation
        results = compare_models(models, data_loaders, args, device)
        demo_verbalizer_analysis(models, data, args)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
