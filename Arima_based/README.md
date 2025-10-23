# RippleNet-TFT: ARIMA + GNN + Temporal Fusion Transformer for Energy Commodity Price Forecasting

A state-of-the-art forecasting system that combines ARIMA baselines, Graph Neural Networks for ripple effects, and Temporal Fusion Transformers to predict energy commodity prices under geopolitical shocks.

## 🚀 Features

- **Multi-Modal Architecture**: Integrates ARIMA, GNN, and TFT for comprehensive forecasting
- **Geopolitical Ripple Effects**: Models how geopolitical events propagate through trade networks
- **News Sentiment Analysis**: Uses FinBERT for semantic news embedding
- **Multi-GPU Training**: Optimized for NVIDIA DGX A100 with mixed precision
- **Comprehensive Evaluation**: Multiple metrics, visualizations, and ablation studies
- **Free Data Sources**: Uses only free APIs and public datasets

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ARIMA Models  │    │  Ripple Graph   │    │  News Encoder   │
│   (Baseline)    │    │     (GNN)       │    │   (FinBERT)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │  Temporal Fusion         │
                    │  Transformer (TFT)       │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Regression Head        │
                    │   (Price Predictions)     │
                    └───────────────────────────┘
```

## 📊 Data Sources

- **Market Data**: Yahoo Finance (OHLCV for energy commodities)
- **Energy Fundamentals**: EIA Open Data API
- **Geopolitical Events**: GDELT Project
- **News Headlines**: NewsAPI (free tier)
- **Macro Indicators**: GPR, EPU indices
- **Trade Data**: UN Comtrade, EIA bilateral flows

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended for DGX A100)

### Setup

1. **Clone and navigate to the project:**
```bash
cd /Users/kailhashed/Desktop/Everything/projects/Geopolitcial_/Arima_based
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp env_example.txt .env
# Edit .env with your API keys
```

### API Keys Setup

Create a `.env` file with your API keys:

```bash
# News API (free tier: 1000 requests/day)
NEWS_API_KEY=your_news_api_key_here

# EIA API (free tier: 5000 requests/day)
EIA_API_KEY=your_eia_api_key_here

# Optional: Weights & Biases
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=ripplenet-tft
WANDB_ENTITY=your_entity
```

## 🚀 Quick Start

### 1. Data Collection

```bash
# Fetch all data sources
python data/data_fetcher.py

# Preprocess and merge data
python data/preprocess.py
```

### 2. Training

```bash
# Train RippleNet-TFT
python train.py --config config.yaml --data data/merged.csv --device cuda:0,1,2,3
```

### 3. Evaluation

```bash
# Evaluate trained model
python evaluate.py --checkpoint checkpoints/best_model.pt --data data/merged.csv
```

## 📁 Project Structure

```
Arima_based/
├── data/                          # Data processing modules
│   ├── data_fetcher.py           # Fetch data from APIs
│   ├── preprocess.py             # Clean and merge data
│   ├── ripple_graph.py           # Build ripple graph
│   ├── news_encoder.py           # News sentiment analysis
│   ├── arima_baseline.py         # ARIMA baseline models
│   └── dataset.py                # Dataset and data loaders
├── model/                         # Model implementations
│   └── ripple_tft.py             # TFT model architecture
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── config.yaml                    # Configuration file
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## ⚙️ Configuration

Key configuration parameters in `config.yaml`:

```yaml
# Model Configuration
model:
  arima:
    max_p: 5
    max_d: 2
    max_q: 5
    seasonal: true
  
  ripple_graph:
    embedding_dim: 64
    num_layers: 3
    propagation_method: "gnn"
  
  tft:
    hidden_size: 64
    lstm_layers: 2
    attention_head_size: 4

# Training Configuration
training:
  batch_size: 128
  learning_rate: 1e-4
  epochs: 100
  use_mixed_precision: true
  gpu_ids: [0, 1, 2, 3]
```

## 🔬 Model Components

### 1. ARIMA Baseline
- Per-commodity ARIMA models with automatic parameter selection
- Seasonal decomposition and stationarity testing
- Residual analysis for TFT input

### 2. Ripple Graph Module
- Country-commodity trade network construction
- Geopolitical event impact propagation
- Graph Neural Network or diffusion-based propagation

### 3. News Encoder
- FinBERT for semantic news embedding
- VADER sentiment analysis fallback
- Daily attention pooling

### 4. Temporal Fusion Transformer
- Multi-head attention mechanism
- Variable selection networks
- Positional encoding and transformer layers

## 📈 Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Pearson Correlation**: Linear correlation
- **Directional Accuracy**: Sign prediction accuracy

## 🎯 DGX A100 Optimization

### Multi-GPU Training
```bash
# Use multiple GPUs
python train.py --device cuda:0,1,2,3

# Mixed precision training (automatic)
# Batch size optimization for A100
```

### Memory Optimization
- Mixed precision training with `torch.cuda.amp`
- Gradient accumulation for large batches
- Model checkpointing for memory efficiency

## 📊 Results and Visualizations

The evaluation script generates:

1. **Predictions vs Actual**: Scatter plots comparing predictions to actual values
2. **Time Series Predictions**: Line plots showing forecast accuracy over time
3. **Metrics Comparison**: Bar charts comparing different models
4. **Error Distribution**: Histograms of prediction errors
5. **Attention Weights**: Visualization of model attention patterns

## 🔧 Advanced Usage

### Custom Data Sources
```python
# Add custom data fetchers
from data.data_fetcher import DataFetcher

fetcher = DataFetcher(config)
custom_data = fetcher.fetch_custom_source()
```

### Model Ablation Studies
```python
# Remove components for ablation
config['model']['ripple_graph']['enabled'] = False
config['model']['news_encoder']['enabled'] = False
```

### Hyperparameter Tuning
```yaml
# Grid search configuration
training:
  learning_rates: [1e-5, 1e-4, 1e-3]
  batch_sizes: [64, 128, 256]
  hidden_sizes: [32, 64, 128]
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision

2. **API Rate Limits**:
   - Implement request delays
   - Use data caching
   - Consider paid API tiers

3. **Data Quality Issues**:
   - Check for missing values
   - Validate date ranges
   - Verify API responses

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python train.py --config config.yaml
```

## 📚 References

- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- [Graph Neural Networks for Financial Time Series](https://arxiv.org/abs/2006.10025)
- [FinBERT: Financial Sentiment Analysis](https://arxiv.org/abs/1908.10063)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- HuggingFace for transformer models
- GDELT Project for geopolitical data
- EIA for energy data APIs

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the configuration examples

---

**Note**: This project is designed for research and educational purposes. Always validate results and consider market conditions when using for actual trading decisions.
