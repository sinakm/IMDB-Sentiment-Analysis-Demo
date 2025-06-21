"""
Sentiment Analysis Challenge - Three Model Approaches
====================================================

This package implements three different approaches to sentiment analysis:
1. Pure LSTM with learned embeddings
2. LSTM with pre-trained GloVe embeddings  
3. Verbalizer/PET approach using transformer embeddings

Each approach uses a custom loss function that weights samples based on
confidence and review length.
"""

__version__ = "0.1.0"
__author__ = "ML Engineer Challenge"
