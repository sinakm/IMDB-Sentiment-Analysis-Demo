#!/usr/bin/env python3
"""
Download ModernBERT model for local packaging in Lambda.

This script downloads the ModernBERT model files and packages them
for inclusion in the Lambda Docker container.
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_modernbert(output_dir: str = "deployment/lambda/models/modernbert"):
    """
    Download ModernBERT model and tokenizer for local use.
    
    Args:
        output_dir: Directory to save the model files
    """
    logger.info("Starting ModernBERT download...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download tokenizer
        logger.info("Downloading ModernBERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        tokenizer.save_pretrained(output_path)
        logger.info(f"Tokenizer saved to {output_path}")
        
        # Download model
        logger.info("Downloading ModernBERT model...")
        model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        model.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")
        
        # Get directory size
        total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        logger.info(f"Total ModernBERT size: {size_mb:.1f} MB")
        
        # Create a manifest file
        manifest = {
            "model_name": "answerdotai/ModernBERT-base",
            "download_date": str(Path().cwd()),
            "total_size_mb": round(size_mb, 1),
            "files": [str(f.relative_to(output_path)) for f in output_path.rglob('*') if f.is_file()]
        }
        
        import json
        with open(output_path / "download_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info("✅ ModernBERT download completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download ModernBERT: {e}")
        # Clean up partial download
        if output_path.exists():
            shutil.rmtree(output_path)
        return False

def verify_download(model_dir: str = "deployment/lambda/models/modernbert"):
    """
    Verify that the downloaded model works correctly.
    
    Args:
        model_dir: Directory containing the model files
    """
    logger.info("Verifying ModernBERT download...")
    
    try:
        model_path = Path(model_dir)
        
        # Test loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        logger.info("✅ Tokenizer loads successfully")
        
        # Test loading model
        model = AutoModel.from_pretrained(str(model_path))
        logger.info("✅ Model loads successfully")
        
        # Test inference
        test_text = "This is a test sentence."
        inputs = tokenizer(test_text, return_tensors="pt", max_length=128, padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            logger.info(f"✅ Test inference successful - embedding shape: {embeddings.shape}")
        
        logger.info("✅ ModernBERT verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ ModernBERT verification failed: {e}")
        return False

def clean_cache():
    """Clean up Hugging Face cache to save space."""
    try:
        import transformers
        cache_dir = Path.home() / ".cache" / "huggingface"
        if cache_dir.exists():
            logger.info(f"Cleaning cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            logger.info("✅ Cache cleaned")
    except Exception as e:
        logger.warning(f"Could not clean cache: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ModernBERT for Lambda deployment")
    parser.add_argument("--output-dir", default="deployment/lambda/models/modernbert",
                       help="Output directory for model files")
    parser.add_argument("--verify", action="store_true",
                       help="Verify download after completion")
    parser.add_argument("--clean-cache", action="store_true",
                       help="Clean Hugging Face cache after download")
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent.parent.parent
    os.chdir(script_dir)
    
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Download ModernBERT
    success = download_modernbert(args.output_dir)
    
    if not success:
        logger.error("Download failed!")
        sys.exit(1)
    
    # Verify if requested
    if args.verify:
        verify_success = verify_download(args.output_dir)
        if not verify_success:
            logger.error("Verification failed!")
            sys.exit(1)
    
    # Clean cache if requested
    if args.clean_cache:
        clean_cache()
    
    logger.info("🎉 ModernBERT download process completed successfully!")

if __name__ == "__main__":
    main()
