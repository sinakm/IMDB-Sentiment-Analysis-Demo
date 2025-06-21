"""
Dataset classes for sentiment analysis.

This module implements PyTorch Dataset classes for all three model approaches:
1. LSTM datasets (pure and pre-trained embeddings)
2. Verbalizer dataset for transformer-based approach
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from .preprocessing import TextPreprocessor, text_to_indices, create_attention_mask


class IMDBDataset(Dataset):
    """Base IMDB dataset class."""
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 max_length: int = 512):
        """
        Initialize IMDB dataset.
        
        Args:
            texts: List of review texts
            labels: List of labels (0=negative, 1=positive)
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        assert len(texts) == len(labels), "Texts and labels must have same length"
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def get_text_lengths(self) -> List[int]:
        """Get text lengths in words."""
        return [len(text.split()) for text in self.texts]
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get label distribution."""
        from collections import Counter
        return dict(Counter(self.labels))


class LSTMDataset(IMDBDataset):
    """
    Dataset for LSTM models (both pure and pre-trained embeddings).
    
    This dataset converts texts to token indices using a vocabulary.
    """
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 vocab: Dict[str, int],
                 max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize LSTM dataset.
        
        Args:
            texts: List of review texts
            labels: List of labels
            vocab: Vocabulary mapping
            max_length: Maximum sequence length
            preprocessor: Text preprocessor
        """
        super().__init__(texts, labels, max_length)
        
        self.vocab = vocab
        self.preprocessor = preprocessor or TextPreprocessor(
            lowercase=True, 
            remove_punctuation=False
        )
        
        # Pre-process and convert texts to indices
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Pre-process all texts and convert to indices."""
        print("Preprocessing texts for LSTM dataset...")
        
        # Convert texts to token indices
        self.token_ids, self.lengths = text_to_indices(
            self.texts, 
            self.vocab, 
            self.max_length, 
            self.preprocessor
        )
        
        # Create attention masks
        self.attention_masks = create_attention_mask(self.lengths, self.max_length)
        
        # Convert labels to tensor
        self.label_tensor = torch.tensor(self.labels, dtype=torch.long)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary with model inputs
        """
        return {
            'input_ids': self.token_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'lengths': self.lengths[idx],
            'labels': self.label_tensor[idx]
        }
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_token_statistics(self) -> Dict[str, float]:
        """Get token-level statistics."""
        all_lengths = self.lengths.numpy()
        return {
            'avg_tokens': float(np.mean(all_lengths)),
            'median_tokens': float(np.median(all_lengths)),
            'min_tokens': int(np.min(all_lengths)),
            'max_tokens': int(np.max(all_lengths)),
            'std_tokens': float(np.std(all_lengths))
        }


class VerbalizerDataset(IMDBDataset):
    """
    Dataset for verbalizer/PET approach.
    
    This dataset appends verbalizer templates to texts and uses transformer tokenization.
    """
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 tokenizer_name: str = 'bert-base-uncased',
                 max_length: int = 512,
                 template_type: str = 'positive',
                 use_both_templates: bool = False):
        """
        Initialize verbalizer dataset.
        
        Args:
            texts: List of review texts
            labels: List of labels
            tokenizer_name: Transformer tokenizer name
            max_length: Maximum sequence length
            template_type: Template type ('positive', 'negative')
            use_both_templates: Whether to use both templates for comparison
        """
        super().__init__(texts, labels, max_length)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.template_type = template_type
        self.use_both_templates = use_both_templates
        
        # Verbalizer templates
        self.templates = {
            'positive': " This statement is positive",
            'negative': " This statement is negative"
        }
        
        # Get verbalizer token IDs
        self.positive_token_id = self.tokenizer.convert_tokens_to_ids("positive")
        self.negative_token_id = self.tokenizer.convert_tokens_to_ids("negative")
        
        # Pre-process data
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Pre-process all texts with verbalizer templates."""
        print("Preprocessing texts for verbalizer dataset...")
        
        if self.use_both_templates:
            # Process with both templates
            self._process_both_templates()
        else:
            # Process with single template
            self._process_single_template()
        
        # Convert labels to tensor
        self.label_tensor = torch.tensor(self.labels, dtype=torch.float)
    
    def _process_single_template(self):
        """Process texts with single template."""
        template = self.templates[self.template_type]
        target_token_id = (self.positive_token_id if self.template_type == 'positive' 
                          else self.negative_token_id)
        
        # Add template to texts
        verbalized_texts = [text + template for text in self.texts]
        
        # Tokenize
        encoding = self.tokenizer(
            verbalized_texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        self.input_ids = encoding['input_ids']
        self.attention_masks = encoding['attention_mask']
        
        # Find verbalizer token positions
        self.verbalizer_positions = self._find_verbalizer_positions(
            self.input_ids, target_token_id
        )
    
    def _process_both_templates(self):
        """Process texts with both templates for comparison."""
        # Process positive template
        pos_template = self.templates['positive']
        pos_texts = [text + pos_template for text in self.texts]
        
        pos_encoding = self.tokenizer(
            pos_texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process negative template
        neg_template = self.templates['negative']
        neg_texts = [text + neg_template for text in self.texts]
        
        neg_encoding = self.tokenizer(
            neg_texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Store both encodings
        self.pos_input_ids = pos_encoding['input_ids']
        self.pos_attention_masks = pos_encoding['attention_mask']
        self.neg_input_ids = neg_encoding['input_ids']
        self.neg_attention_masks = neg_encoding['attention_mask']
        
        # Find verbalizer positions
        self.pos_verbalizer_positions = self._find_verbalizer_positions(
            self.pos_input_ids, self.positive_token_id
        )
        self.neg_verbalizer_positions = self._find_verbalizer_positions(
            self.neg_input_ids, self.negative_token_id
        )
    
    def _find_verbalizer_positions(self, 
                                 input_ids: torch.Tensor, 
                                 target_token_id: int) -> torch.Tensor:
        """
        Find positions of verbalizer tokens.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            target_token_id: Target token ID to find
            
        Returns:
            Positions of verbalizer tokens [batch_size]
        """
        positions = []
        
        for i, sequence in enumerate(input_ids):
            # Find all positions of target token
            token_positions = (sequence == target_token_id).nonzero(as_tuple=True)[0]
            
            if len(token_positions) > 0:
                # Use the last occurrence (most likely from our template)
                positions.append(token_positions[-1].item())
            else:
                # Fallback: use position before [SEP] or [PAD]
                sep_positions = (sequence == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                if len(sep_positions) > 0:
                    positions.append(sep_positions[0].item() - 1)
                else:
                    positions.append(self.max_length - 2)  # Before [PAD]
        
        return torch.tensor(positions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary with model inputs
        """
        if self.use_both_templates:
            return {
                'pos_input_ids': self.pos_input_ids[idx],
                'pos_attention_mask': self.pos_attention_masks[idx],
                'pos_verbalizer_positions': self.pos_verbalizer_positions[idx],
                'neg_input_ids': self.neg_input_ids[idx],
                'neg_attention_mask': self.neg_attention_masks[idx],
                'neg_verbalizer_positions': self.neg_verbalizer_positions[idx],
                'labels': self.label_tensor[idx]
            }
        else:
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_masks[idx],
                'verbalizer_positions': self.verbalizer_positions[idx],
                'labels': self.label_tensor[idx]
            }
    
    def get_verbalizer_statistics(self) -> Dict[str, Union[int, float]]:
        """Get verbalizer-specific statistics."""
        if self.use_both_templates:
            pos_positions = self.pos_verbalizer_positions.numpy()
            neg_positions = self.neg_verbalizer_positions.numpy()
            
            return {
                'avg_pos_verbalizer_position': float(np.mean(pos_positions)),
                'avg_neg_verbalizer_position': float(np.mean(neg_positions)),
                'pos_verbalizer_position_std': float(np.std(pos_positions)),
                'neg_verbalizer_position_std': float(np.std(neg_positions))
            }
        else:
            positions = self.verbalizer_positions.numpy()
            return {
                'avg_verbalizer_position': float(np.mean(positions)),
                'verbalizer_position_std': float(np.std(positions)),
                'min_verbalizer_position': int(np.min(positions)),
                'max_verbalizer_position': int(np.max(positions))
            }


class MultiModelDataset:
    """
    Wrapper class that creates datasets for all three model approaches.
    
    This is useful for training and comparing all models on the same data splits.
    """
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 vocab: Dict[str, int],
                 tokenizer_name: str = 'bert-base-uncased',
                 max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize multi-model dataset wrapper.
        
        Args:
            texts: List of review texts
            labels: List of labels
            vocab: Vocabulary for LSTM models
            tokenizer_name: Transformer tokenizer name
            max_length: Maximum sequence length
            preprocessor: Text preprocessor for LSTM models
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.preprocessor = preprocessor
        
        # Create datasets for all models
        self._create_datasets()
    
    def _create_datasets(self):
        """Create datasets for all model types."""
        print("Creating datasets for all model types...")
        
        # LSTM dataset (for both pure and pre-trained models)
        self.lstm_dataset = LSTMDataset(
            texts=self.texts,
            labels=self.labels,
            vocab=self.vocab,
            max_length=self.max_length,
            preprocessor=self.preprocessor
        )
        
        # Verbalizer dataset
        self.verbalizer_dataset = VerbalizerDataset(
            texts=self.texts,
            labels=self.labels,
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            template_type='positive'
        )
        
        # Verbalizer dataset with both templates (for analysis)
        self.verbalizer_both_dataset = VerbalizerDataset(
            texts=self.texts,
            labels=self.labels,
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            use_both_templates=True
        )
    
    def get_lstm_dataset(self) -> LSTMDataset:
        """Get LSTM dataset."""
        return self.lstm_dataset
    
    def get_verbalizer_dataset(self, use_both_templates: bool = False) -> VerbalizerDataset:
        """Get verbalizer dataset."""
        if use_both_templates:
            return self.verbalizer_both_dataset
        else:
            return self.verbalizer_dataset
    
    def get_all_datasets(self) -> Dict[str, Dataset]:
        """Get all datasets."""
        return {
            'lstm': self.lstm_dataset,
            'verbalizer': self.verbalizer_dataset,
            'verbalizer_both': self.verbalizer_both_dataset
        }
    
    def get_dataset_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all datasets."""
        return {
            'lstm': {
                **self.lstm_dataset.get_token_statistics(),
                **self.lstm_dataset.get_label_distribution()
            },
            'verbalizer': {
                **self.verbalizer_dataset.get_verbalizer_statistics(),
                **self.verbalizer_dataset.get_label_distribution()
            },
            'verbalizer_both': {
                **self.verbalizer_both_dataset.get_verbalizer_statistics(),
                **self.verbalizer_both_dataset.get_label_distribution()
            }
        }


class DatasetFactory:
    """Factory class for creating datasets."""
    
    @staticmethod
    def create_lstm_datasets(train_texts: List[str],
                           train_labels: List[int],
                           val_texts: List[str],
                           val_labels: List[int],
                           test_texts: List[str],
                           test_labels: List[int],
                           vocab: Dict[str, int],
                           max_length: int = 512,
                           preprocessor: Optional[TextPreprocessor] = None) -> Tuple[LSTMDataset, LSTMDataset, LSTMDataset]:
        """
        Create LSTM datasets for train/val/test splits.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = LSTMDataset(train_texts, train_labels, vocab, max_length, preprocessor)
        val_dataset = LSTMDataset(val_texts, val_labels, vocab, max_length, preprocessor)
        test_dataset = LSTMDataset(test_texts, test_labels, vocab, max_length, preprocessor)
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def create_verbalizer_datasets(train_texts: List[str],
                                 train_labels: List[int],
                                 val_texts: List[str],
                                 val_labels: List[int],
                                 test_texts: List[str],
                                 test_labels: List[int],
                                 tokenizer_name: str = 'bert-base-uncased',
                                 max_length: int = 512,
                                 template_type: str = 'positive') -> Tuple[VerbalizerDataset, VerbalizerDataset, VerbalizerDataset]:
        """
        Create verbalizer datasets for train/val/test splits.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = VerbalizerDataset(train_texts, train_labels, tokenizer_name, max_length, template_type)
        val_dataset = VerbalizerDataset(val_texts, val_labels, tokenizer_name, max_length, template_type)
        test_dataset = VerbalizerDataset(test_texts, test_labels, tokenizer_name, max_length, template_type)
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def create_all_datasets(train_texts: List[str],
                          train_labels: List[int],
                          val_texts: List[str],
                          val_labels: List[int],
                          test_texts: List[str],
                          test_labels: List[int],
                          vocab: Dict[str, int],
                          tokenizer_name: str = 'bert-base-uncased',
                          max_length: int = 512,
                          preprocessor: Optional[TextPreprocessor] = None) -> Dict[str, Tuple[Dataset, Dataset, Dataset]]:
        """
        Create all datasets for all model types.
        
        Returns:
            Dictionary with datasets for each model type
        """
        # LSTM datasets
        lstm_datasets = DatasetFactory.create_lstm_datasets(
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
            vocab, max_length, preprocessor
        )
        
        # Verbalizer datasets
        verbalizer_datasets = DatasetFactory.create_verbalizer_datasets(
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
            tokenizer_name, max_length
        )
        
        return {
            'lstm': lstm_datasets,
            'verbalizer': verbalizer_datasets
        }
