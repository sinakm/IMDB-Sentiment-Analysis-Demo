"""
Data preprocessing utilities for sentiment analysis.

This module handles vocabulary building, GloVe embedding loading,
and data loader creation for all three model approaches.
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Union
import re
import string
from datasets import load_dataset
from torch.utils.data import DataLoader
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import requests
import zipfile
import os
from tqdm import tqdm


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_stopwords: bool = False,
                 remove_numbers: bool = False):
        """
        Initialize text preprocessor.
        
        Args:
            lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stopwords
            remove_numbers: Whether to remove numbers
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        if remove_stopwords:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize and remove stopwords
        if self.remove_stopwords:
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in self.stop_words]
            text = ' '.join(tokens)
        
        return text
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]


def build_vocabulary(texts: List[str], 
                    min_freq: int = 2,
                    max_vocab_size: int = 50000,
                    preprocessor: Optional[TextPreprocessor] = None) -> Dict[str, int]:
    """
    Build vocabulary from texts.
    
    Args:
        texts: List of texts
        min_freq: Minimum frequency for word inclusion
        max_vocab_size: Maximum vocabulary size
        preprocessor: Optional text preprocessor
        
    Returns:
        Word to index mapping
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)
    
    # Preprocess texts
    processed_texts = preprocessor.preprocess_texts(texts)
    
    # Count word frequencies
    word_counts = Counter()
    for text in processed_texts:
        tokens = word_tokenize(text.lower())
        word_counts.update(tokens)
    
    # Filter by frequency and limit vocabulary size
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Sort by frequency (most frequent first)
    filtered_words = sorted(filtered_words, key=lambda x: word_counts[x], reverse=True)
    
    # Limit vocabulary size
    if len(filtered_words) > max_vocab_size - 4:  # Reserve space for special tokens
        filtered_words = filtered_words[:max_vocab_size - 4]
    
    # Create vocabulary with special tokens
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<SOS>': 2,
        '<EOS>': 3
    }
    
    # Add words to vocabulary
    for i, word in enumerate(filtered_words):
        vocab[word] = i + 4
    
    return vocab


def download_glove_embeddings(glove_dim: int = 300, 
                            glove_version: str = '6B',
                            cache_dir: str = './cache') -> str:
    """
    Download GloVe embeddings if not already cached.
    
    Args:
        glove_dim: Embedding dimension (50, 100, 200, 300)
        glove_version: GloVe version ('6B', '42B', '840B')
        cache_dir: Cache directory
        
    Returns:
        Path to GloVe file
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # GloVe URLs
    glove_urls = {
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    }
    
    if glove_version not in glove_urls:
        raise ValueError(f"Unsupported GloVe version: {glove_version}")
    
    # File paths
    zip_path = os.path.join(cache_dir, f'glove.{glove_version}.zip')
    glove_file = os.path.join(cache_dir, f'glove.{glove_version}.{glove_dim}d.txt')
    
    # Check if already downloaded
    if os.path.exists(glove_file):
        print(f"GloVe embeddings already cached at {glove_file}")
        return glove_file
    
    # Download GloVe
    print(f"Downloading GloVe {glove_version} embeddings...")
    url = glove_urls[glove_version]
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    # Extract GloVe
    print("Extracting GloVe embeddings...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(cache_dir)
    
    # Remove zip file
    os.remove(zip_path)
    
    return glove_file


def load_glove_embeddings(vocab: Dict[str, int], 
                         glove_dim: int = 300,
                         glove_version: str = '6B',
                         cache_dir: str = './cache') -> torch.Tensor:
    """
    Load GloVe embeddings for vocabulary.
    
    Args:
        vocab: Vocabulary mapping
        glove_dim: Embedding dimension
        glove_version: GloVe version
        cache_dir: Cache directory
        
    Returns:
        Embedding matrix [vocab_size, embed_dim]
    """
    # Download GloVe if needed
    glove_file = download_glove_embeddings(glove_dim, glove_version, cache_dir)
    
    # Initialize embedding matrix
    vocab_size = len(vocab)
    embeddings = torch.zeros(vocab_size, glove_dim)
    
    # Load GloVe embeddings
    print("Loading GloVe embeddings...")
    found_words = 0
    
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading embeddings"):
            parts = line.strip().split()
            word = parts[0]
            
            if word in vocab:
                vector = torch.tensor([float(x) for x in parts[1:]])
                embeddings[vocab[word]] = vector
                found_words += 1
    
    print(f"Found embeddings for {found_words}/{vocab_size} words ({found_words/vocab_size*100:.1f}%)")
    
    # Initialize special tokens with random embeddings
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    for token in special_tokens:
        if token in vocab:
            if token == '<PAD>':
                # Keep padding as zeros
                embeddings[vocab[token]] = torch.zeros(glove_dim)
            else:
                # Random initialization for other special tokens
                embeddings[vocab[token]] = torch.randn(glove_dim) * 0.1
    
    return embeddings


def text_to_indices(texts: List[str], 
                   vocab: Dict[str, int],
                   max_length: int = 512,
                   preprocessor: Optional[TextPreprocessor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert texts to token indices.
    
    Args:
        texts: List of texts
        vocab: Vocabulary mapping
        max_length: Maximum sequence length
        preprocessor: Optional text preprocessor
        
    Returns:
        Tuple of (token_ids, lengths)
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)
    
    # Preprocess texts
    processed_texts = preprocessor.preprocess_texts(texts)
    
    token_ids = []
    lengths = []
    
    for text in processed_texts:
        tokens = word_tokenize(text.lower())
        
        # Convert to indices
        indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        
        # Truncate or pad
        if len(indices) > max_length:
            indices = indices[:max_length]
        
        original_length = len(indices)
        
        # Pad with <PAD> tokens
        while len(indices) < max_length:
            indices.append(vocab['<PAD>'])
        
        token_ids.append(indices)
        lengths.append(original_length)
    
    return torch.tensor(token_ids), torch.tensor(lengths)


def create_attention_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Create attention mask from lengths.
    
    Args:
        lengths: Sequence lengths [batch_size]
        max_length: Maximum sequence length
        
    Returns:
        Attention mask [batch_size, max_length]
    """
    batch_size = lengths.size(0)
    attention_mask = torch.zeros(batch_size, max_length)
    
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    
    return attention_mask


def load_imdb_dataset(cache_dir: str = './cache') -> Dict[str, List]:
    """
    Load IMDB dataset from HuggingFace.
    
    Args:
        cache_dir: Cache directory
        
    Returns:
        Dictionary with train/test splits
    """
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb", cache_dir=cache_dir)
    
    return {
        'train_texts': dataset['train']['text'],
        'train_labels': dataset['train']['label'],
        'test_texts': dataset['test']['text'],
        'test_labels': dataset['test']['label']
    }


def create_train_val_split(texts: List[str], 
                          labels: List[int],
                          val_ratio: float = 0.1,
                          random_seed: int = 42) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Create train/validation split.
    
    Args:
        texts: List of texts
        labels: List of labels
        val_ratio: Validation ratio
        random_seed: Random seed
        
    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels)
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Create indices
    indices = np.arange(len(texts))
    np.random.shuffle(indices)
    
    # Split indices
    val_size = int(len(texts) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Split data
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_texts, train_labels, val_texts, val_labels


def create_data_loaders(train_dataset, 
                       val_dataset, 
                       test_dataset,
                       batch_size: int = 16,
                       num_workers: int = 0,
                       shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def extract_verbalizer_embeddings(texts: List[str], 
                                 labels: List[int],
                                 max_samples: Optional[int] = None,
                                 cache_file: str = "verbalizer_embeddings.pkl",
                                 cache_dir: str = "./cache",
                                 max_length: int = 256,
                 verbalizer_template: str = " The sentiment of this statement is",
                                 random_seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract verbalizer embeddings from texts using ModernBERT.
    
    This function:
    1. Limits dataset to max_samples (balanced positive/negative)
    2. Adds verbalizer template to each text
    3. Extracts last token embedding from ModernBERT
    4. Saves embeddings to cache file
    5. Returns embeddings and labels tensors
    
    Args:
        texts: List of input texts
        labels: List of labels
        max_samples: Maximum number of samples (balanced)
        cache_file: Cache file name
        cache_dir: Cache directory
        max_length: Maximum sequence length
        verbalizer_template: Template to append to texts
        random_seed: Random seed
        
    Returns:
        Tuple of (embeddings, labels) tensors
    """
    import pickle
    from transformers import AutoModel, AutoTokenizer
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_file)
    
    # Check if embeddings are already cached
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            return data['embeddings'], data['labels']
    
    # Limit dataset if specified
    if max_samples and max_samples < len(texts):
        texts, labels = _limit_balanced_dataset(texts, labels, max_samples, random_seed)
        print(f"Limited dataset to {len(texts):,} balanced samples")
    
    print(f"Extracting verbalizer embeddings for {len(texts):,} texts...")
    
    # Load ModernBERT
    print("Loading ModernBERT model...")
    model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Extract embeddings
    embeddings = []
    batch_size = 32  # Process in batches to avoid memory issues
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Add verbalizer template
        verbalized_texts = [text + verbalizer_template for text in batch_texts]
        
        # Tokenize
        encoding = tokenizer(
            verbalized_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            
            # Extract last meaningful token for each sequence
            batch_embeddings = []
            for j in range(last_hidden_state.size(0)):
                # Find last non-padding token
                last_pos = (attention_mask[j] == 1).nonzero(as_tuple=True)[0][-1].item()
                batch_embeddings.append(last_hidden_state[j, last_pos].cpu())
            
            embeddings.extend(batch_embeddings)
    
    # Convert to tensors
    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    print(f"✅ Extracted embeddings: {embeddings.shape}")
    print(f"✅ Labels: {labels.shape}")
    
    # Save to cache
    cache_data = {
        'embeddings': embeddings,
        'labels': labels,
        'template': verbalizer_template,
        'max_length': max_length,
        'num_samples': len(texts)
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"💾 Cached embeddings to {cache_path}")
    
    return embeddings, labels


def _limit_balanced_dataset(texts: List[str], 
                           labels: List[int], 
                           max_samples: int, 
                           random_seed: int) -> Tuple[List[str], List[int]]:
    """
    Limit dataset to max_samples with balanced positive/negative samples.
    
    Args:
        texts: List of texts
        labels: List of labels
        max_samples: Maximum number of samples
        random_seed: Random seed
        
    Returns:
        Tuple of (limited_texts, limited_labels)
    """
    import random
    random.seed(random_seed)
    
    # Separate positive and negative samples
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]
    
    # Calculate how many of each class to take
    samples_per_class = max_samples // 2
    
    # Randomly sample from each class
    selected_positive = random.sample(positive_indices, 
                                    min(samples_per_class, len(positive_indices)))
    selected_negative = random.sample(negative_indices, 
                                    min(samples_per_class, len(negative_indices)))
    
    # Combine and shuffle
    selected_indices = selected_positive + selected_negative
    random.shuffle(selected_indices)
    
    # Extract selected samples
    limited_texts = [texts[i] for i in selected_indices]
    limited_labels = [labels[i] for i in selected_indices]
    
    return limited_texts, limited_labels


def get_dataset_statistics(texts: List[str], labels: List[int]) -> Dict[str, Union[int, float]]:
    """
    Get dataset statistics.
    
    Args:
        texts: List of texts
        labels: List of labels
        
    Returns:
        Dictionary with statistics
    """
    # Text length statistics
    lengths = [len(text.split()) for text in texts]
    
    # Label distribution
    label_counts = Counter(labels)
    
    return {
        'num_samples': len(texts),
        'num_positive': label_counts.get(1, 0),
        'num_negative': label_counts.get(0, 0),
        'positive_ratio': label_counts.get(1, 0) / len(labels),
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'std_length': np.std(lengths)
    }
