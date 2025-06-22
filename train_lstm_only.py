"""
Train only the Pure LSTM model on the full IMDB dataset.

This script trains just the Pure LSTM classifier with learned embeddings
for comparison with the other approaches.

Usage:
    python train_lstm_only.py --num_epochs 5 --batch_size 32
    python train_lstm_only.py --hidden_dim 256 --embed_dim 300 --num_layers 3
"""

import argparse
import torch
import torch.nn as nn
import time
import json
from pathlib import Path

# Import our modules
from src.data.preprocessing import (
    load_imdb_dataset, 
    create_train_val_split, 
    build_vocabulary,
    TextPreprocessor,
    get_dataset_statistics
)
from src.data.dataset import LSTMDataset
from src.data.preprocessing import create_data_loaders
from src.models.lstm_pure import PureLSTMClassifier
from src.models.loss_functions import UniversalCustomLoss
from src.utils.helpers import set_seed, create_directories
from src.training.trainer import SingleModelTrainer
from src.training.evaluator import ModelEvaluator


def setup_experiment(args):
    """Setup experiment configuration."""
    print("=" * 60)
    print("PURE LSTM-ONLY TRAINING")
    print("=" * 60)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    create_directories(['checkpoints/lstm', 'results'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    
    return device


def load_and_prepare_data(args):
    """Load and prepare IMDB dataset."""
    print("\n" + "=" * 50)
    print("LOADING FULL IMDB DATASET")
    print("=" * 50)
    
    # Load full IMDB dataset
    data = load_imdb_dataset(cache_dir='./cache')
    
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


def build_vocabulary_for_lstm(data, args):
    """Build vocabulary for LSTM model."""
    print("\n" + "=" * 50)
    print("BUILDING VOCABULARY")
    print("=" * 50)
    
    train_texts, train_labels = data['train']
    
    # Create text preprocessor
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)
    
    # Build vocabulary
    print("Building vocabulary from training data...")
    vocab = build_vocabulary(
        train_texts, 
        min_freq=2, 
        max_vocab_size=args.vocab_size,
        preprocessor=preprocessor
    )
    
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Max vocab size: {args.vocab_size:,}")
    
    return vocab, preprocessor


def create_lstm_model(vocab, args, device):
    """Create LSTM model."""
    print("\n" + "=" * 50)
    print("CREATING PURE LSTM MODEL")
    print("=" * 50)
    
    model = PureLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True,
        use_attention=True
    ).to(device)
    
    # Print model information
    info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Model Type: {info['model_type']}")
    print(f"  Vocabulary Size: {info['vocab_size']:,}")
    print(f"  Embedding Dim: {info['embed_dim']}")
    print(f"  Hidden Dim: {info['hidden_dim']}")
    print(f"  Num Layers: {info['num_layers']}")
    print(f"  Bidirectional: {info['bidirectional']}")
    print(f"  Use Attention: {info['use_attention']}")
    print(f"  Total Parameters: {info['total_parameters']:,}")
    print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
    
    return model


def create_datasets_and_loaders(data, vocab, preprocessor, args):
    """Create LSTM datasets and data loaders."""
    print("\n" + "=" * 50)
    print("CREATING LSTM DATASETS")
    print("=" * 50)
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    test_texts, test_labels = data['test']
    
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
    
    # Print dataset statistics
    stats = train_dataset.get_token_statistics()
    print(f"\nLSTM Dataset Statistics:")
    print(f"  Average tokens: {stats['avg_tokens']:.1f}")
    print(f"  Min tokens: {stats['min_tokens']}")
    print(f"  Max tokens: {stats['max_tokens']}")
    print(f"  Median tokens: {stats['median_tokens']:.1f}")
    print(f"  Vocabulary size: {train_dataset.get_vocab_size():,}")
    
    return train_loader, val_loader, test_loader


def train_lstm(model, train_loader, val_loader, test_loader, args, device):
    """Train the LSTM model."""
    print("\n" + "=" * 50)
    print("TRAINING PURE LSTM MODEL")
    print("=" * 50)
    
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
    print(f"Starting training for {args.num_epochs} epochs...")
    start_time = time.time()
    
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_best=True,
        checkpoint_dir='./checkpoints/lstm'
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluator = ModelEvaluator(model, device)
    test_results = evaluator.evaluate(test_loader)
    
    # Print results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Training Time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print(f"Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"Test F1 Score: {test_results['f1']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    
    # Save results
    results = {
        'model_type': 'lstm',
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'test_accuracy': float(test_results['accuracy']),
        'test_f1': float(test_results['f1']),
        'test_precision': float(test_results['precision']),
        'test_recall': float(test_results['recall']),
        'model_info': model.get_model_info(),
        'hyperparameters': {
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length,
            'vocab_size': args.vocab_size,
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'confidence_penalty': args.confidence_penalty,
            'length_weight': args.length_weight
        }
    }
    
    results_file = Path('./results/lstm_only_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to {results_file}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Pure LSTM Model Only')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=200, 
                       help='Embedding dimension (default: 200)')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                       help='Hidden dimension (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2, 
                       help='Number of LSTM layers (default: 2)')
    parser.add_argument('--vocab_size', type=int, default=20000, 
                       help='Vocabulary size (default: 20000)')
    parser.add_argument('--max_length', type=int, default=256, 
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.3, 
                       help='Dropout probability (default: 0.3)')
    
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
    
    # Load data
    data = load_and_prepare_data(args)
    
    # Build vocabulary
    vocab, preprocessor = build_vocabulary_for_lstm(data, args)
    
    # Create model
    model = create_lstm_model(vocab, args, device)
    
    # Create datasets and loaders
    train_loader, val_loader, test_loader = create_datasets_and_loaders(data, vocab, preprocessor, args)
    
    # Train model
    results = train_lstm(model, train_loader, val_loader, test_loader, args, device)
    
    print("\n" + "=" * 60)
    print("🎉 PURE LSTM TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"⏱️  Training Time: {results['training_time_minutes']:.1f} minutes")
    print(f"🎯 Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"📊 F1 Score: {results['test_f1']:.4f}")
    print(f"🔧 Trainable Params: {results['model_info']['trainable_parameters']:,}")
    
    # Quick comparison hint
    print(f"\n💡 Expected performance comparison:")
    print(f"   - LSTM trains all {results['model_info']['trainable_parameters']:,} parameters")
    print(f"   - Should be faster than old verbalizer, slower than new verbalizer")
    print(f"   - Accuracy typically 75-85% on IMDB")


if __name__ == "__main__":
    main()
