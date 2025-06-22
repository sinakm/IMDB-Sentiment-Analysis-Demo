# IMDB Sentiment Analysis: LSTM vs Verbalizer

A focused comparison of two different approaches to sentiment analysis on the IMDB movie reviews dataset.

## 🎯 Overview

This project implements and compares two fundamentally different sentiment analysis approaches:

1. **Pure LSTM** - Traditional approach with learned embeddings from scratch (~500K-1M parameters)
2. **Verbalizer** - Modern approach using pre-computed ModernBERT embeddings (~239K parameters)

## 🏗️ Project Structure

```
├── main.py                 # Main entry point for comparison
├── train_lstm_only.py      # Individual LSTM training
├── train_verbalizer_only.py # Individual Verbalizer training
├── compare_all_models.py   # Automated comparison pipeline
├── pyproject.toml          # Project dependencies
├── requirements.txt        # Alternative dependency file
├── .gitignore             # Git ignore rules
├── TRAINING_GUIDE.md      # Detailed training instructions
├── problem_statement.md   # Original problem description
├── test_imports.py        # Import verification script
├── src/                   # Source code
│   ├── data/              # Data handling modules
│   │   ├── dataset.py     # LSTM dataset
│   │   ├── verbalizer_dataset.py  # Verbalizer dataset
│   │   └── preprocessing.py       # Data preprocessing utilities
│   ├── models/            # Model implementations
│   │   ├── lstm_pure.py           # Pure LSTM classifier
│   │   ├── verbalizer.py          # Verbalizer classifier
│   │   └── loss_functions.py      # Custom loss functions
│   ├── training/          # Training infrastructure
│   │   ├── trainer.py     # Training loops
│   │   └── evaluator.py   # Model evaluation
│   └── utils/             # Utility functions
│       ├── helpers.py     # General utilities
│       └── config.py      # Configuration management
├── deployment/            # AWS deployment (Docker Lambda + CDK)
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks (optional)
├── checkpoints/           # Model checkpoints (gitignored)
├── results/              # Training results (gitignored)
└── cache/                # Dataset cache (gitignored)
```

## 🚀 Quick Start

### 1. Setup Environment

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

### 2. Verify Installation

```bash
python test_imports.py
```

### 3. Run Quick Demo

```bash
# Quick demo (LSTM only, small dataset)
python main.py --mode demo

# Compare both models (recommended)
python main.py --mode compare

# Individual model training
python train_lstm_only.py --max_samples 1000 --num_epochs 5
python train_verbalizer_only.py --max_samples 1000 --num_epochs 5
```

## 📊 Usage Examples

### Quick Comparison

```bash
# Automated comparison pipeline
python compare_all_models.py --quick
```

- Trains both models with 1K samples
- Generates comparison table and insights
- Takes ~5-10 minutes

### Full Comparison

```bash
# Full dataset comparison
python compare_all_models.py --full
```

- Trains both models on full IMDB dataset
- Comprehensive evaluation and analysis
- Takes ~30-60 minutes

### Individual Model Training

```bash
# Train LSTM only
python train_lstm_only.py --num_epochs 10 --batch_size 32

# Train Verbalizer only
python train_verbalizer_only.py --max_samples 5000 --num_epochs 5

# Quick testing
python train_verbalizer_only.py --max_samples 100 --num_epochs 3
```

## 🎛️ Configuration Options

### Common Parameters

| Parameter         | Default | Description                            |
| ----------------- | ------- | -------------------------------------- |
| `--batch_size`    | `32`    | Training batch size                    |
| `--num_epochs`    | `5`     | Number of training epochs              |
| `--learning_rate` | `0.001` | Learning rate for optimization         |
| `--max_length`    | `256`   | Maximum sequence length                |
| `--dropout`       | `0.3`   | Dropout probability                    |
| `--max_samples`   | `None`  | Limit dataset size (for quick testing) |

### LSTM-Specific Parameters

| Parameter      | Default | Description             |
| -------------- | ------- | ----------------------- |
| `--embed_dim`  | `200`   | Embedding dimension     |
| `--hidden_dim` | `128`   | Hidden dimension        |
| `--num_layers` | `2`     | Number of LSTM layers   |
| `--vocab_size` | `20000` | Maximum vocabulary size |

### Verbalizer-Specific Parameters

| Parameter      | Default | Description                 |
| -------------- | ------- | --------------------------- |
| `--hidden_dim` | `128`   | Classifier hidden dimension |

## 📈 Expected Results

### Quick Test (1K samples, 3-5 epochs)

```
Model          Accuracy    F1 Score    Time (min)    Parameters
LSTM           ~70-75%     ~70-75%     ~3-5          ~500K-1M
Verbalizer     ~68-75%     ~68-75%     ~0.5-1        ~239K
```

### Full Dataset (25K training samples)

```
Model          Accuracy    F1 Score    Time (min)    Parameters
LSTM           ~80-85%     ~80-85%     ~15-30        ~500K-1M
Verbalizer     ~85-90%     ~85-90%     ~3-5          ~239K
```

## 🔧 Technical Details

### Model Comparison

| Aspect               | LSTM                           | Verbalizer                     |
| -------------------- | ------------------------------ | ------------------------------ |
| **Approach**         | Learn everything from scratch  | Leverage pre-trained knowledge |
| **Embeddings**       | Learned during training        | Pre-computed ModernBERT        |
| **Architecture**     | Bidirectional LSTM + Attention | Deep feedforward classifier    |
| **Training Speed**   | Slower (learns embeddings)     | Faster (pre-computed)          |
| **Parameters**       | ~500K-1M                       | ~239K                          |
| **Dependencies**     | Self-contained                 | Requires ModernBERT            |
| **Interpretability** | Traditional, well-understood   | Modern transfer learning       |

### LSTM Architecture

```python
Embedding(vocab_size, embed_dim) →
Bidirectional LSTM(hidden_dim, num_layers) →
Attention Mechanism →
Classifier(hidden_dim*2, 1)
```

### Verbalizer Architecture

```python
Pre-computed ModernBERT embeddings →
Linear(768, 256) → ReLU → BatchNorm → Dropout →
Linear(256, 128) → ReLU → BatchNorm → Dropout →
Linear(128, 64) → ReLU → Dropout →
Linear(64, 1)
```

### Verbalizer Template

The verbalizer uses the template: `" The sentiment of this statement is"`

Example:

- Input: `"This movie was fantastic!"`
- Processed: `"This movie was fantastic! The sentiment of this statement is"`
- ModernBERT extracts the final token embedding for classification

## 🛠️ Development

### Adding Custom Models

1. Create model class in `src/models/`
2. Implement required methods: `forward()`, `get_model_info()`
3. Create training script following existing patterns
4. Add to comparison pipeline

### Custom Loss Functions

```python
# Example: Custom loss with confidence penalty
class CustomLoss(nn.Module):
    def __init__(self, confidence_penalty=2.0):
        super().__init__()
        self.confidence_penalty = confidence_penalty
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        # Standard BCE loss
        bce_loss = self.bce(logits, labels)

        # Confidence penalty
        probs = torch.sigmoid(logits)
        confidence = torch.abs(probs - 0.5)
        penalty = -self.confidence_penalty * confidence.mean()

        return bce_loss + penalty
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.20+
- scikit-learn
- tqdm
- datasets
- pandas (for comparison tables)

See `pyproject.toml` for complete dependency list.

## 🚀 Production Deployment

This project includes a complete AWS deployment using Docker Lambda and CDK.

### Quick Deployment

```bash
# 1. Train both models
python compare_all_models.py --full

# 2. Export for deployment
python deployment/scripts/export_models.py

# 3. Deploy with CDK
cd deployment/cdk
pip install -r requirements.txt
cdk deploy
```

### What Gets Deployed

- **AWS Lambda Function**: Docker container with both models
- **API Gateway**: REST API with authentication
- **CloudWatch**: Logging and monitoring

### API Usage

```bash
curl -X POST https://your-api-url/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{"text": "This movie was fantastic!", "model": "verbalizer"}'
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment instructions.

## 🎯 Key Insights

### When to Use LSTM

- **Self-contained deployment** (no external model dependencies)
- **Traditional approach** preferred
- **Learning task-specific representations** from scratch
- **Interpretable architecture** required

### When to Use Verbalizer

- **Fast training** required (5-10x speedup)
- **Leveraging pre-trained knowledge** preferred
- **Fewer parameters** needed
- **Modern transfer learning** approach acceptable

### Performance Trade-offs

- **Verbalizer**: Faster training, fewer parameters, leverages pre-trained knowledge
- **LSTM**: Self-contained, learns task-specific representations, traditional approach
- **Both**: Achieve similar accuracy (~80-90% on IMDB)

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

- Repository: [IMDB-Sentiment-Analysis-Demo](https://github.com/sinakm/IMDB-Sentiment-Analysis-Demo)

---

**Note**: This project demonstrates the comparison between traditional and modern NLP approaches for educational and research purposes.
