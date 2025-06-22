"""
Train only the Verbalizer model on the full IMDB dataset.

This script trains just the Verbalizer classifier on pre-computed ModernBERT embeddings
for comparison with the other approaches.

Usage:
    python train_verbalizer_only.py --num_epochs 5 --batch_size 32
    python train_verbalizer_only.py --max_samples 1000 --num_epochs 3
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
    extract_verbalizer_embeddings,
    get_dataset_statistics
)
from src.data.verbalizer_dataset import VerbalizerDataset
from src.data.preprocessing import create_data_loaders
from src.models.verbalizer import VerbalizerClassifier
from src.models.loss_functions import UniversalCustomLoss
from src.utils.helpers import set_seed, create_directories
from src.training.trainer import SingleModelTrainer
from src.training.evaluator import ModelEvaluator


def setup_experiment(args):
    """Setup experiment configuration."""
    print("=" * 60)
    print("VERBALIZER-ONLY TRAINING")
    print("=" * 60)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    create_directories(['checkpoints/verbalizer', 'results'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    
    return device


def load_and_prepare_data(args):
    """Load and prepare IMDB dataset."""
    print("\n" + "=" * 50)
    if args.max_samples:
        print(f"LOADING LIMITED IMDB DATASET ({args.max_samples:,} samples)")
    else:
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


def extract_embeddings_for_verbalizer(data, args):
    """Extract embeddings for verbalizer model."""
    print("\n" + "=" * 50)
    print("EXTRACTING VERBALIZER EMBEDDINGS")
    print("=" * 50)
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    test_texts, test_labels = data['test']
    
    # Calculate proportional limits for quick testing
    if args.max_samples:
        # For quick testing, limit all datasets proportionally
        val_max_samples = max(10, args.max_samples // 5)  # 20% of train size, min 10
        test_max_samples = args.max_samples  # Same as train for fair comparison
        print(f"Quick testing mode: Train={args.max_samples}, Val={val_max_samples}, Test={test_max_samples}")
    else:
        val_max_samples = None
        test_max_samples = None
        print("Full dataset mode: No sample limits")
    
    # Extract embeddings for each split
    print("Extracting training embeddings...")
    train_embeddings, train_labels_tensor = extract_verbalizer_embeddings(
        train_texts, train_labels,
        max_samples=args.max_samples,
        cache_file=f"verbalizer_train_{args.max_samples or 'full'}.pkl",
        cache_dir='./cache',
        max_length=args.max_length,
        random_seed=args.seed
    )
    
    print("Extracting validation embeddings...")
    val_embeddings, val_labels_tensor = extract_verbalizer_embeddings(
        val_texts, val_labels,
        max_samples=val_max_samples,
        cache_file=f"verbalizer_val_{args.max_samples or 'full'}.pkl",
        cache_dir='./cache',
        max_length=args.max_length,
        random_seed=args.seed + 1
    )
    
    print("Extracting test embeddings...")
    test_embeddings, test_labels_tensor = extract_verbalizer_embeddings(
        test_texts, test_labels,
        max_samples=test_max_samples,
        cache_file=f"verbalizer_test_{args.max_samples or 'full'}.pkl",
        cache_dir='./cache',
        max_length=args.max_length,
        random_seed=args.seed + 2
    )
    
    print(f"\nEmbedding Extraction Complete:")
    print(f"  Train: {train_embeddings.shape}")
    print(f"  Val: {val_embeddings.shape}")
    print(f"  Test: {test_embeddings.shape}")
    
    return {
        'train': (train_embeddings, train_labels_tensor),
        'val': (val_embeddings, val_labels_tensor),
        'test': (test_embeddings, test_labels_tensor)
    }


def create_verbalizer_model(embedding_dim, args, device):
    """Create verbalizer model."""
    print("\n" + "=" * 50)
    print("CREATING VERBALIZER MODEL")
    print("=" * 50)
    
    model = VerbalizerClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Print model information
    info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Model Type: {info['model_type']}")
    print(f"  Embedding Dim: {info['embedding_dim']}")
    print(f"  Hidden Dim: {info['hidden_dim']}")
    print(f"  Dropout: {info['dropout']}")
    print(f"  Total Parameters: {info['total_parameters']:,}")
    print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"  Description: {info['description']}")
    
    return model


def create_datasets_and_loaders(embeddings_data, args):
    """Create verbalizer datasets and data loaders."""
    print("\n" + "=" * 50)
    print("CREATING VERBALIZER DATASETS")
    print("=" * 50)
    
    train_embeddings, train_labels = embeddings_data['train']
    val_embeddings, val_labels = embeddings_data['val']
    test_embeddings, test_labels = embeddings_data['test']
    
    # Create verbalizer datasets
    print("Creating verbalizer datasets...")
    train_dataset = VerbalizerDataset(train_embeddings, train_labels)
    val_dataset = VerbalizerDataset(val_embeddings, val_labels)
    test_dataset = VerbalizerDataset(test_embeddings, test_labels)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Print dataset statistics
    stats = train_dataset.get_embedding_statistics()
    print(f"\nVerbalizer Dataset Statistics:")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print(f"  Train samples: {stats['num_samples']:,}")
    print(f"  Positive samples: {stats['positive_samples']:,}")
    print(f"  Negative samples: {stats['negative_samples']:,}")
    print(f"  Embedding mean: {stats['embedding_mean']:.4f}")
    print(f"  Embedding std: {stats['embedding_std']:.4f}")
    
    return train_loader, val_loader, test_loader


def train_verbalizer(model, train_loader, val_loader, test_loader, args, device):
    """Train the verbalizer model."""
    print("\n" + "=" * 50)
    print("TRAINING VERBALIZER MODEL")
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
        checkpoint_dir='./checkpoints/verbalizer'
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
        'model_type': 'verbalizer',
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
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'max_samples': args.max_samples,
            'confidence_penalty': args.confidence_penalty,
            'length_weight': args.length_weight
        }
    }
    
    results_file = Path('./results/verbalizer_only_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to {results_file}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Verbalizer Model Only')
    
    # Data parameters
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of training samples (balanced pos/neg). '
                            'Also limits val/test sets proportionally for quick testing. '
                            'Examples: 50 (train=50, val=10, test=50), 1000 (train=1000, val=200, test=1000)')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for classifier (default: 128)')
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
    
    # Extract embeddings
    embeddings_data = extract_embeddings_for_verbalizer(data, args)
    
    # Get embedding dimension from first batch
    embedding_dim = embeddings_data['train'][0].shape[1]
    
    # Create model
    model = create_verbalizer_model(embedding_dim, args, device)
    
    # Create datasets and loaders
    train_loader, val_loader, test_loader = create_datasets_and_loaders(embeddings_data, args)
    
    # Train model
    results = train_verbalizer(model, train_loader, val_loader, test_loader, args, device)
    
    print("\n" + "=" * 60)
    print("🎉 VERBALIZER TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"⏱️  Training Time: {results['training_time_minutes']:.1f} minutes")
    print(f"🎯 Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"📊 F1 Score: {results['test_f1']:.4f}")
    print(f"🔧 Trainable Params: {results['model_info']['trainable_parameters']:,}")
    
    # Quick comparison hint
    print(f"\n💡 Expected performance comparison:")
    print(f"   - Verbalizer trains only {results['model_info']['trainable_parameters']:,} parameters")
    print(f"   - Should be FASTEST training (pre-computed embeddings)")
    print(f"   - Accuracy typically 85-90% on IMDB")


if __name__ == "__main__":
    main()
