# IMDB Sentiment Analysis Demo

A comprehensive comparison of three different approaches to sentiment analysis on the IMDB movie reviews dataset.

## 🎯 Overview

This project implements and compares three different sentiment analysis approaches:

1. **Pure LSTM** - LSTM with learned embeddings from scratch (2.4M parameters)
2. **ModernBERT** - State-of-the-art transformer model (149M parameters)
3. **Verbalizer** - Template-based approach using BERT (109M parameters)

## 🏗️ Project Structure

```
├── main.py                 # Main entry point with type hints
├── pyproject.toml          # Project dependencies
├── requirements.txt        # Alternative dependency file
├── .gitignore             # Git ignore rules
├── INSTALLATION.md        # Setup instructions
├── PROJECT_SUMMARY.md     # Detailed project overview
├── problem_statement.md   # Original problem description
├── test_imports.py        # Import verification script
├── src/                   # Source code
│   ├── data/              # Data handling modules
│   │   ├── dataset.py     # PyTorch datasets
│   │   ├── modernbert_dataset.py  # ModernBERT-specific dataset
│   │   └── preprocessing.py       # Data preprocessing utilities
│   ├── models/            # Model implementations
│   │   ├── lstm_pure.py           # Pure LSTM classifier
│   │   ├── modernbert_classifier.py  # ModernBERT classifier
│   │   ├── verbalizer.py          # Template-based classifier
│   │   └── loss_functions.py      # Custom loss functions
│   ├── training/          # Training infrastructure
│   │   ├── trainer.py     # Training loops
│   │   └── evaluator.py   # Model evaluation
│   └── utils/             # Utility functions
│       ├── helpers.py     # General utilities
│       └── config.py      # Configuration management
├── notebooks/             # Jupyter notebooks (optional)
├── checkpoints/           # Model checkpoints (gitignored)
├── results/              # Training results (gitignored)
└── cache/                # Dataset cache (gitignored)
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/sinakm/IMDB-Sentiment-Analysis-Demo.git
cd IMDB-Sentiment-Analysis-Demo
```

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 3. Verify Installation

```bash
python test_imports.py
```

### 4. Run Demo

```bash
# Quick demo (trains only Pure LSTM)
python main.py --mode demo

# Compare all three models
python main.py --mode compare

# Full training with analysis
python main.py --mode full
```

## 📊 Usage Examples

### Demo Mode (Quick Test)

```bash
python main.py --mode demo
```

- Uses balanced subset of 1,000 training samples
- Trains only Pure LSTM model
- Shows verbalizer analysis examples
- Takes ~2-3 minutes

### Compare Mode (Full Comparison)

```bash
python main.py --mode compare
```

- Trains all three models on full dataset
- Generates comparison table
- Saves results to `results/comparison_results.json`
- Takes ~30-60 minutes depending on hardware

### Full Mode (Complete Analysis)

```bash
python main.py --mode full
```

- Trains all three models
- Includes verbalizer analysis demo
- Most comprehensive evaluation

### Custom Parameters

```bash
python main.py --mode compare \
    --batch_size 32 \
    --num_epochs 5 \
    --learning_rate 0.001 \
    --hidden_dim 256
```

## 🎛️ Configuration Options

| Parameter         | Default | Description                               |
| ----------------- | ------- | ----------------------------------------- |
| `--mode`          | `demo`  | Execution mode: `demo`, `compare`, `full` |
| `--batch_size`    | `16`    | Training batch size                       |
| `--num_epochs`    | `3`     | Number of training epochs                 |
| `--learning_rate` | `0.001` | Learning rate for optimization            |
| `--embed_dim`     | `200`   | Embedding dimension for LSTM              |
| `--hidden_dim`    | `128`   | Hidden dimension for LSTM                 |
| `--vocab_size`    | `20000` | Maximum vocabulary size                   |
| `--max_length`    | `256`   | Maximum sequence length                   |
| `--dropout`       | `0.3`   | Dropout probability                       |

## 📈 Expected Results

### Demo Mode Results

```
Pure LSTM:
  Accuracy: ~66.0%
  F1 Score: ~55.3%
  Training Time: ~2 minutes
```

### Full Comparison (Expected)

```
Model          Accuracy    F1 Score    Time (min)
Pure LSTM      ~75-80%     ~75-80%     ~5-10
ModernBERT     ~88-92%     ~88-92%     ~20-40
Verbalizer     ~85-90%     ~85-90%     ~15-30
```

## 🔧 Technical Details

### Model Architectures

**Pure LSTM:**

- Bidirectional LSTM with attention
- Learned embeddings from scratch
- Custom loss function with confidence penalty

**ModernBERT:**

- State-of-the-art transformer architecture
- Pre-trained on large text corpus
- Fine-tuned for sentiment classification

**Verbalizer:**

- Template-based approach using BERT
- Uses natural language templates
- Consistency scoring for predictions

### Dataset Information

- **Source**: IMDB Movie Reviews Dataset
- **Size**: 50,000 training + 50,000 test reviews
- **Balance**: 50% positive, 50% negative
- **Preprocessing**: Automatic download and caching

## 🛠️ Development

### Adding New Models

1. Create model class in `src/models/`
2. Implement required methods: `forward()`, `get_model_info()`
3. Add to model creation in `main.py`
4. Create appropriate dataset if needed

### Custom Loss Functions

- Implement in `src/models/loss_functions.py`
- Inherit from `nn.Module`
- Add confidence penalties or length weighting

### Data Processing

- Custom preprocessors in `src/data/preprocessing.py`
- Dataset classes in `src/data/dataset.py`
- Support for different tokenization schemes

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.20+
- scikit-learn
- tqdm
- datasets

See `pyproject.toml` for complete dependency list.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- IMDB dataset from Stanford AI Lab
- ModernBERT from Answer.AI
- Hugging Face Transformers library
- PyTorch team for the framework

## 📞 Contact

- GitHub: [@sinakm](https://github.com/sinakm)
- Repository: [IMDB-Sentiment-Analysis-Demo](https://github.com/sinakm/IMDB-Sentiment-Analysis-Demo)

---

**Note**: This project demonstrates modern NLP techniques and is intended for educational and research purposes.
