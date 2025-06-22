"""
Lightweight inference engine for AWS Lambda.

This module provides a unified interface for running inference on both
sentiment analysis models: LSTM and Verbalizer.
"""

import json
import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModel
import logging

# Configure PyTorch for Lambda environment
os.environ['TORCH_COMPILE_DISABLE'] = '1'  # Disable torch.compile
torch.set_num_threads(1)  # Single thread for Lambda
torch.set_num_interop_threads(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMInference:
    """Lightweight LSTM inference class."""
    
    def __init__(self, model_path: str, config_path: str, vocab_path: str):
        """Initialize LSTM inference model."""
        self.device = torch.device('cpu')  # Lambda uses CPU
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        # Create reverse vocabulary for unknown words
        self.vocab_size = len(self.vocab)
        self.unk_token = self.vocab.get('<UNK>', 1)
        self.pad_token = self.vocab.get('<PAD>', 0)
        
        # Load model
        self.model = self._create_model()
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Training checkpoint format
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict format
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        logger.info(f"LSTM model loaded with {self.vocab_size} vocabulary size")
    
    def _create_model(self):
        """Create LSTM model architecture matching training."""
        # Import the actual model class
        class PureLSTMClassifier(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'], padding_idx=0)
                self.lstm = nn.LSTM(
                    config['embed_dim'], 
                    config['hidden_dim'],
                    config['num_layers'],
                    batch_first=True,
                    bidirectional=config['bidirectional'],
                    dropout=config['dropout'] if config['num_layers'] > 1 else 0
                )
                
                # Attention mechanism
                lstm_output_dim = config['hidden_dim'] * (2 if config['bidirectional'] else 1)
                self.attention = nn.Linear(lstm_output_dim, 1)
                
                # Classifier (direct Linear layer to match training)
                self.classifier = nn.Linear(lstm_output_dim, 1)
                
            def forward(self, input_ids, attention_mask=None, lengths=None):
                # Embedding
                embedded = self.embedding(input_ids)
                
                # LSTM
                lstm_out, (hidden, cell) = self.lstm(embedded)
                
                # Attention mechanism
                if attention_mask is not None:
                    # Apply attention mask
                    attention_weights = self.attention(lstm_out).squeeze(-1)
                    attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
                    attention_weights = torch.softmax(attention_weights, dim=1)
                    
                    # Weighted sum
                    context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
                else:
                    # Use final hidden state
                    if self.lstm.bidirectional:
                        context = torch.cat([hidden[-2], hidden[-1]], dim=1)
                    else:
                        context = hidden[-1]
                
                # Classification
                logits = self.classifier(context)
                return logits, attention_weights if attention_mask is not None else None
        
        return PureLSTMClassifier(self.config)
    
    def _tokenize(self, text: str, max_length: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text for LSTM."""
        # Basic tokenization (should match training preprocessing)
        words = text.lower().split()[:max_length]
        token_ids = [self.vocab.get(word, self.unk_token) for word in words]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(self.pad_token)
            attention_mask.append(0)
        
        return (
            torch.tensor([token_ids], dtype=torch.long),
            torch.tensor([attention_mask], dtype=torch.long)
        )
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Run inference on text."""
        start_time = time.time()
        
        # Tokenize input
        input_ids, attention_mask = self._tokenize(text)
        
        # Run inference
        with torch.no_grad():
            logits, _ = self.model(input_ids, attention_mask)
            probability = torch.sigmoid(logits).item()
            prediction = "positive" if probability > 0.5 else "negative"
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "prediction": prediction,
            "confidence": probability,
            "probability": probability,
            "processing_time_ms": processing_time
        }

class VerbalizerInference:
    """Lightweight Verbalizer inference class."""
    
    def __init__(self, model_path: str, config_path: str):
        """Initialize Verbalizer inference model."""
        self.device = torch.device('cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load ModernBERT from local directory
        try:
            # Check if we have local ModernBERT files
            models_dir = Path(model_path).parent.parent
            modernbert_path = models_dir / "modernbert"
            
            logger.info(f"Looking for ModernBERT at: {modernbert_path}")
            logger.info(f"ModernBERT path exists: {modernbert_path.exists()}")
            
            if modernbert_path.exists():
                # Load from local files
                logger.info("Loading ModernBERT tokenizer from local files...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(modernbert_path))
                
                logger.info("Loading ModernBERT model from local files...")
                self.bert_model = AutoModel.from_pretrained(str(modernbert_path))
                self.bert_model.eval()
                
                logger.info("ModernBERT loaded successfully from local files")
            else:
                logger.error(f"ModernBERT not found at {modernbert_path}")
                self.tokenizer = None
                self.bert_model = None
        except Exception as e:
            logger.error(f"Failed to load ModernBERT: {e}")
            import traceback
            logger.error(f"ModernBERT traceback: {traceback.format_exc()}")
            self.tokenizer = None
            self.bert_model = None
        
        # Load verbalizer classifier
        self.classifier = self._create_classifier()
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Training checkpoint format
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict format
            self.classifier.load_state_dict(checkpoint)
        
        self.classifier.eval()
        
        # Verbalizer template
        self.template = " The sentiment of this statement is"
        
        logger.info("Verbalizer model loaded")
    
    def _create_classifier(self):
        """Create verbalizer classifier architecture matching training."""
        class VerbalizerClassifier(nn.Module):
            def __init__(self, config):
                super().__init__()
                embedding_dim = config.get('embedding_dim', 768)
                hidden_dim = config.get('hidden_dim', 128)
                dropout = config.get('dropout', 0.3)
                
                # Improved classifier architecture - deeper and more robust
                self.classifier = nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim * 2),  # 256 neurons
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.Dropout(dropout),
                    
                    nn.Linear(hidden_dim * 2, hidden_dim),     # 128 neurons
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout * 0.7),  # Slightly less dropout in deeper layers
                    
                    nn.Linear(hidden_dim, hidden_dim // 2),    # 64 neurons
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                    
                    nn.Linear(hidden_dim // 2, 1)              # Output layer
                )
            
            def forward(self, input_ids=None, attention_mask=None, lengths=None, embeddings=None):
                # Use embeddings directly if provided, otherwise use input_ids
                if embeddings is not None:
                    input_embeddings = embeddings
                elif input_ids is not None:
                    input_embeddings = input_ids
                else:
                    raise ValueError("Either embeddings or input_ids must be provided")
                
                # Forward pass through classifier
                logits = self.classifier(input_embeddings)
                return logits, None  # Return tuple for compatibility
        
        return VerbalizerClassifier(self.config)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Run inference on text using verbalizer approach."""
        start_time = time.time()
        
        # Check if ModernBERT is available
        if self.tokenizer is None or self.bert_model is None:
            # Fallback: return error for now since verbalizer needs ModernBERT
            processing_time = int((time.time() - start_time) * 1000)
            return {
                "prediction": "error",
                "confidence": 0.0,
                "probability": 0.0,
                "processing_time_ms": processing_time,
                "error": "ModernBERT not available for verbalizer model"
            }
        
        # Add verbalizer template
        verbalized_text = text + self.template
        
        # Tokenize with ModernBERT
        inputs = self.tokenizer(
            verbalized_text,
            max_length=self.config.get('max_length', 256),
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract embeddings with ModernBERT
        with torch.no_grad():
            # Get ModernBERT embeddings
            bert_outputs = self.bert_model(**inputs)
            last_hidden_state = bert_outputs.last_hidden_state
            
            # Extract last meaningful token embedding
            attention_mask = inputs['attention_mask']
            last_pos = (attention_mask[0] == 1).nonzero(as_tuple=True)[0][-1].item()
            embedding = last_hidden_state[0, last_pos].unsqueeze(0)
            
            # Run through verbalizer classifier
            logits, _ = self.classifier(embeddings=embedding)
            probability = torch.sigmoid(logits).item()
            prediction = "positive" if probability > 0.5 else "negative"
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "prediction": prediction,
            "confidence": probability,
            "probability": probability,
            "processing_time_ms": processing_time
        }

class SentimentInferenceEngine:
    """Unified inference engine for both LSTM and Verbalizer models."""
    
    def __init__(self, models_dir: str = None):
        """Initialize both models."""
        # Use environment variable or default path
        if models_dir is None:
            models_dir = os.environ.get("MODEL_PATH", "/var/task/models")
        
        self.models_dir = Path(models_dir)
        self.models = {}
        
        logger.info("Initializing sentiment inference engine...")
        
        # Debug: Check if models directory exists
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Models directory exists: {self.models_dir.exists()}")
        if self.models_dir.exists():
            logger.info(f"Contents: {list(self.models_dir.iterdir())}")
        
        # Initialize LSTM
        try:
            lstm_model_path = str(self.models_dir / "lstm" / "model.pt")
            lstm_config_path = str(self.models_dir / "lstm" / "config.json")
            lstm_vocab_path = str(self.models_dir / "lstm" / "vocab.json")
            
            logger.info(f"LSTM paths - model: {lstm_model_path}, config: {lstm_config_path}, vocab: {lstm_vocab_path}")
            logger.info(f"LSTM files exist - model: {Path(lstm_model_path).exists()}, config: {Path(lstm_config_path).exists()}, vocab: {Path(lstm_vocab_path).exists()}")
            
            self.models['lstm'] = LSTMInference(
                model_path=lstm_model_path,
                config_path=lstm_config_path,
                vocab_path=lstm_vocab_path
            )
            logger.info("LSTM model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LSTM: {e}")
            import traceback
            logger.error(f"LSTM traceback: {traceback.format_exc()}")
            self.models['lstm'] = None
        
        # Initialize Verbalizer
        try:
            verbalizer_model_path = str(self.models_dir / "verbalizer" / "model.pt")
            verbalizer_config_path = str(self.models_dir / "verbalizer" / "config.json")
            
            logger.info(f"Verbalizer paths - model: {verbalizer_model_path}, config: {verbalizer_config_path}")
            logger.info(f"Verbalizer files exist - model: {Path(verbalizer_model_path).exists()}, config: {Path(verbalizer_config_path).exists()}")
            
            self.models['verbalizer'] = VerbalizerInference(
                model_path=verbalizer_model_path,
                config_path=verbalizer_config_path
            )
            logger.info("Verbalizer model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Verbalizer: {e}")
            import traceback
            logger.error(f"Verbalizer traceback: {traceback.format_exc()}")
            self.models['verbalizer'] = None
    
    def predict_single(self, text: str, model_name: str) -> Dict[str, Any]:
        """Run inference on a single model."""
        if model_name not in self.models or self.models[model_name] is None:
            return {
                "prediction": "error",
                "confidence": 0.0,
                "processing_time_ms": 0,
                "error": f"Model {model_name} not available"
            }
        
        try:
            return self.models[model_name].predict(text)
        except Exception as e:
            logger.error(f"Error in {model_name}: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "processing_time_ms": 0,
                "error": str(e)
            }
    
    def predict_all(self, text: str) -> Dict[str, Any]:
        """Run inference on all available models."""
        start_time = time.time()
        
        predictions = {}
        successful_models = 0
        
        # Run inference on each model
        for model_name in ['lstm', 'verbalizer']:
            if self.models.get(model_name) is not None:
                try:
                    predictions[model_name] = self.models[model_name].predict(text)
                    successful_models += 1
                except Exception as e:
                    logger.error(f"Error in {model_name}: {e}")
                    predictions[model_name] = {
                        "prediction": "error",
                        "confidence": 0.0,
                        "processing_time_ms": 0,
                        "error": str(e)
                    }
        
        # Calculate consensus
        consensus = self._calculate_consensus(predictions)
        
        total_time = int((time.time() - start_time) * 1000)
        
        return {
            "text": text,
            "predictions": predictions,
            "consensus": consensus,
            "total_processing_time_ms": total_time,
            "models_available": successful_models,
            "models": list(self.models.keys())
        }
    
    def _calculate_consensus(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate consensus from both model predictions."""
        valid_predictions = [
            pred for pred in predictions.values() 
            if pred.get("prediction") not in ["error", None]
        ]
        
        if not valid_predictions:
            return {
                "prediction": "error",
                "agreement": False,
                "avg_confidence": 0.0
            }
        
        # Count predictions
        positive_count = sum(1 for pred in valid_predictions if pred["prediction"] == "positive")
        negative_count = sum(1 for pred in valid_predictions if pred["prediction"] == "negative")
        
        # Determine consensus
        if positive_count > negative_count:
            consensus_prediction = "positive"
            agreement = positive_count == len(valid_predictions)
        elif negative_count > positive_count:
            consensus_prediction = "negative"
            agreement = negative_count == len(valid_predictions)
        else:
            consensus_prediction = "neutral"
            agreement = False
        
        # Calculate average confidence
        avg_confidence = sum(pred["confidence"] for pred in valid_predictions) / len(valid_predictions)
        
        return {
            "prediction": consensus_prediction,
            "agreement": agreement,
            "avg_confidence": round(avg_confidence, 4),
            "models_count": len(valid_predictions)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "available_models": [name for name, model in self.models.items() if model is not None],
            "total_models": len([model for model in self.models.values() if model is not None]),
            "model_types": {
                "lstm": "Traditional LSTM with learned embeddings",
                "verbalizer": "Modern approach with pre-computed ModernBERT embeddings"
            }
        }
