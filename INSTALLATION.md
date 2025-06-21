# Installation Guide

## Quick Setup

1. **Install Python Dependencies:**

```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn tqdm pandas nltk tensorboard
```

2. **Run Demo:**

```bash
python main.py --mode demo
```

## Alternative Installation Methods

### Using pip with requirements file:

```bash
pip install -e .
```

### Using conda:

```bash
conda install pytorch transformers datasets scikit-learn matplotlib seaborn tqdm pandas nltk tensorboard -c pytorch -c huggingface -c conda-forge
```

## Verification

To verify the installation worked correctly:

```bash
python -c "import torch; import transformers; import datasets; print('✓ All dependencies installed successfully!')"
```

## Expected Output

When running `python main.py --mode demo`, you should see:

```
Sentiment Analysis Challenge - Three Model Approaches
============================================================
Mode: demo
Device: CUDA / CPU

==================================================
LOADING AND PREPARING DATA
==================================================
Loading IMDB dataset...
Demo mode: Using subset of data

Dataset Statistics:
Train: 900 samples, 50.0% positive, avg length: 231.4 words
Val: 100 samples, 50.0% positive, avg length: 234.1 words
Test: 200 samples, 50.0% positive, avg length: 230.8 words

==================================================
PREPARING VOCABULARIES AND EMBEDDINGS
==================================================
Building vocabulary...
Vocabulary size: 8,432
Loading GloVe embeddings...
Found embeddings for 6,234/8,432 words (73.9%)

==================================================
CREATING MODELS
==================================================
Creating Pure LSTM model...
Creating LSTM with GloVe embeddings...
Creating Verbalizer model...

PURE_LSTM:
  Total parameters: 2,847,105
  Trainable parameters: 2,847,105

PRETRAINED_LSTM:
  Total parameters: 2,847,105
  Trainable parameters: 2,847,105

VERBALIZER:
  Total parameters: 109,483,009
  Trainable parameters: 768
```

## Troubleshooting

### CUDA Out of Memory

```bash
python main.py --mode demo --batch_size 8 --max_length 128
```

### Slow Download

The first run downloads IMDB dataset and GloVe embeddings. Subsequent runs use cached data.

### Import Errors

Make sure you're in the buildops directory:

```bash
cd buildops
python main.py --mode demo
```
