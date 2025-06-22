"""
Export trained models for Lambda deployment.

This script takes the trained model checkpoints and creates lightweight
artifacts optimized for inference in AWS Lambda.
"""

import sys
import os
import json
import torch
import shutil
from pathlib import Path
from typing import Dict, Any
import pickle

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

def export_lstm_model():
    """Export LSTM model and vocabulary for deployment."""
    print("📦 Exporting LSTM model...")
    
    project_root = Path(__file__).parent.parent.parent
    checkpoints_dir = project_root / "checkpoints" / "lstm"
    artifacts_dir = project_root / "artifacts" / "lstm"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the best model checkpoint
    checkpoint_files = list(checkpoints_dir.glob("*.pt"))
    if not checkpoint_files:
        print("❌ No LSTM checkpoint found. Run training first.")
        return False
    
    # Use the most recent checkpoint (or best_model.pt if it exists)
    best_checkpoint = None
    for checkpoint in checkpoint_files:
        if "best" in checkpoint.name:
            best_checkpoint = checkpoint
            break
    
    if best_checkpoint is None:
        best_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Using checkpoint: {best_checkpoint}")
    
    # Copy model checkpoint
    shutil.copy2(best_checkpoint, artifacts_dir / "model.pt")
    
    # Look for vocabulary file in cache or create minimal config
    vocab_file = project_root / "cache" / "vocab.json"
    if vocab_file.exists():
        shutil.copy2(vocab_file, artifacts_dir / "vocab.json")
        print("✅ Vocabulary file copied from cache")
    else:
        # Create minimal vocabulary (this should be improved in production)
        print("⚠️ Creating minimal vocabulary - consider saving vocab during training")
        vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3
        }
        # Add some common words
        common_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
                       "good", "bad", "great", "terrible", "amazing", "awful", "love", "hate", "like", "dislike",
                       "movie", "film", "actor", "acting", "plot", "story", "character", "scene"]
        for i, word in enumerate(common_words):
            vocab[word] = i + 4
        
        with open(artifacts_dir / "vocab.json", 'w') as f:
            json.dump(vocab, f, indent=2)
    
    # Create model configuration
    model_config = {
        "vocab_size": 20000,  # This should match training config
        "embed_dim": 200,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True,
        "use_attention": True,
        "max_length": 256
    }
    
    # Try to load actual config if available
    config_file = checkpoints_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
            model_config.update(saved_config)
        print("✅ Loaded saved model configuration")
    
    # Save model configuration
    with open(artifacts_dir / "config.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print("✅ LSTM model exported")
    return True

def export_verbalizer_model():
    """Export Verbalizer model for deployment."""
    print("📦 Exporting Verbalizer model...")
    
    project_root = Path(__file__).parent.parent.parent
    checkpoints_dir = project_root / "checkpoints" / "verbalizer"
    artifacts_dir = project_root / "artifacts" / "verbalizer"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the best model checkpoint
    checkpoint_files = list(checkpoints_dir.glob("*.pt"))
    if not checkpoint_files:
        print("❌ No Verbalizer checkpoint found. Run training first.")
        return False
    
    # Use the most recent checkpoint (or best_model.pt if it exists)
    best_checkpoint = None
    for checkpoint in checkpoint_files:
        if "best" in checkpoint.name:
            best_checkpoint = checkpoint
            break
    
    if best_checkpoint is None:
        best_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Using checkpoint: {best_checkpoint}")
    
    # Copy model checkpoint
    shutil.copy2(best_checkpoint, artifacts_dir / "model.pt")
    
    # Create model configuration
    model_config = {
        "embedding_dim": 768,  # ModernBERT embedding dimension
        "hidden_dim": 128,
        "dropout": 0.3,
        "max_length": 256,
        "template": " The sentiment of this statement is"
    }
    
    # Try to load actual config if available
    config_file = checkpoints_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
            model_config.update(saved_config)
        print("✅ Loaded saved model configuration")
    
    # Save model configuration
    with open(artifacts_dir / "config.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print("✅ Verbalizer model exported")
    return True

def create_deployment_manifest():
    """Create a manifest file with deployment information."""
    print("📋 Creating deployment manifest...")
    
    project_root = Path(__file__).parent.parent.parent
    artifacts_dir = project_root / "artifacts"
    
    manifest = {
        "version": "2.0.0",
        "description": "LSTM vs Verbalizer sentiment analysis models",
        "models": {
            "lstm": {
                "type": "pytorch_lstm",
                "files": ["model.pt", "config.json", "vocab.json"],
                "memory_mb": 150,
                "description": "Traditional LSTM with learned embeddings"
            },
            "verbalizer": {
                "type": "pytorch_verbalizer",
                "files": ["model.pt", "config.json"],
                "memory_mb": 600,  # Includes ModernBERT for embeddings
                "description": "Modern approach with pre-computed ModernBERT embeddings"
            }
        },
        "total_memory_mb": 750,
        "lambda_config": {
            "memory": 2048,  # Reduced from 4096 since we removed ModernBERT fine-tuning
            "timeout": 300,   # Reduced timeout
            "runtime": "python3.9"
        },
        "api_endpoints": {
            "predict": {
                "method": "POST",
                "path": "/predict",
                "description": "Run sentiment analysis",
                "parameters": {
                    "text": "string (required) - Text to analyze",
                    "model": "string (optional) - 'lstm', 'verbalizer', or 'both' (default)"
                }
            },
            "health": {
                "method": "GET", 
                "path": "/health",
                "description": "Health check endpoint"
            }
        }
    }
    
    with open(artifacts_dir / "deployment_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("✅ Deployment manifest created")

def validate_exports():
    """Validate that all required files are present."""
    print("🔍 Validating exports...")
    
    project_root = Path(__file__).parent.parent.parent
    artifacts_dir = project_root / "artifacts"
    
    required_files = {
        "lstm": ["model.pt", "config.json", "vocab.json"],
        "verbalizer": ["model.pt", "config.json"]
    }
    
    all_valid = True
    total_size = 0
    
    for model_name, files in required_files.items():
        model_dir = artifacts_dir / model_name
        print(f"  Checking {model_name}...")
        
        if not model_dir.exists():
            print(f"    ❌ Directory {model_name} missing")
            all_valid = False
            continue
        
        for file_name in files:
            file_path = model_dir / file_name
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"    ✅ {file_name} ({size_mb:.1f} MB)")
            else:
                print(f"    ❌ {file_name} missing")
                all_valid = False
    
    print(f"\nTotal artifacts size: {total_size:.1f} MB")
    
    # Check if size is reasonable for Lambda
    if total_size > 500:  # 500MB is a reasonable limit
        print(f"⚠️ Warning: Total size ({total_size:.1f} MB) is quite large for Lambda")
    
    return all_valid

def create_copy_script():
    """Create a script to copy artifacts to Lambda directory."""
    print("📝 Creating copy script...")
    
    project_root = Path(__file__).parent.parent.parent
    
    # Windows batch script
    batch_script = """@echo off
echo Copying model artifacts to Lambda directory...

if exist "deployment\\lambda\\models" (
    rmdir /s /q "deployment\\lambda\\models"
)

mkdir "deployment\\lambda\\models"
xcopy /E /I "artifacts" "deployment\\lambda\\models"

echo SUCCESS: Models copied successfully!
echo Next step: cd deployment/cdk && cdk deploy
"""
    
    with open(project_root / "copy_models.bat", 'w') as f:
        f.write(batch_script)
    
    # Unix shell script
    shell_script = """#!/bin/bash
echo "Copying model artifacts to Lambda directory..."

rm -rf deployment/lambda/models
mkdir -p deployment/lambda/models
cp -r artifacts/* deployment/lambda/models/

echo "SUCCESS: Models copied successfully!"
echo "Next step: cd deployment/cdk && cdk deploy"
"""
    
    with open(project_root / "copy_models.sh", 'w') as f:
        f.write(shell_script)
    
    # Make shell script executable
    os.chmod(project_root / "copy_models.sh", 0o755)
    
    print("✅ Copy scripts created (copy_models.bat and copy_models.sh)")
    
    # Also copy the deployment manifest directly
    artifacts_dir = project_root / "artifacts"
    lambda_dir = project_root / "deployment" / "lambda"
    
    if (artifacts_dir / "deployment_manifest.json").exists():
        lambda_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            artifacts_dir / "deployment_manifest.json",
            lambda_dir / "deployment_manifest.json"
        )
        print("✅ Deployment manifest copied to Lambda directory")

def main():
    """Main export pipeline."""
    print("=" * 60)
    print("📦 MODEL EXPORT PIPELINE FOR LAMBDA DEPLOYMENT")
    print("=" * 60)
    print("Exporting LSTM and Verbalizer models for AWS Lambda...")
    
    success = True
    
    # Export each model
    success &= export_lstm_model()
    success &= export_verbalizer_model()
    
    if success:
        # Create deployment manifest
        create_deployment_manifest()
        
        # Create copy scripts
        create_copy_script()
        
        # Validate exports
        if validate_exports():
            print("\n" + "=" * 60)
            print("🎉 MODEL EXPORT COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("Next steps:")
            print("1. Copy models to Lambda:")
            print("   Windows: copy_models.bat")
            print("   Unix:    ./copy_models.sh")
            print("2. Deploy with CDK:")
            print("   cd deployment/cdk && cdk deploy")
            print(f"3. Artifacts ready in: {Path(__file__).parent.parent.parent / 'artifacts'}")
            
            # Show manifest summary
            manifest_file = Path(__file__).parent.parent.parent / "artifacts" / "deployment_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                print(f"\n📋 Deployment Summary:")
                print(f"   Version: {manifest['version']}")
                print(f"   Models: {', '.join(manifest['models'].keys())}")
                print(f"   Total Memory: {manifest['total_memory_mb']} MB")
                print(f"   Lambda Memory: {manifest['lambda_config']['memory']} MB")
        else:
            print("❌ Export validation failed")
            success = False
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
