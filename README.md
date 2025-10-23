# Geopolitical Energy Price Forecasting

A machine learning system that forecasts energy commodity prices by learning geopolitical ripple effects using a Dual-Stream Attention Bi-LSTM model.

## Overview

This project implements a hybrid deep learning model that combines:
- **Temporal Stream**: Market data sequences (OHLCV + technical indicators)
- **Contextual Stream**: Geopolitical indicators, sentiment, and policy uncertainty
- **Ripple Effects**: Propagation of geopolitical events through trade-linked countries
- **Attention Mechanism**: Learns importance of different feature groups

## Features

- 🌍 **Geopolitical Ripple Effects**: Models how events propagate through global trade networks
- 📊 **Multi-Modal Data**: Integrates market data, news sentiment, and policy indicators
- 🧠 **Attention Mechanism**: Learns which features are most important for predictions
- 🚀 **Multi-GPU Training**: Optimized for DGX A100 with distributed data parallel
- 📈 **Comprehensive Evaluation**: Accuracy, F1-score, ROC-AUC, and attention visualization

## Architecture

```
Input Data Sources:
├── Yahoo Finance (OHLCV + Technical Indicators)
├── GDELT (Geopolitical Events)
├── NewsAPI (Sentiment Analysis)
├── Policy Uncertainty (GPR, EPU)
└── Trade Weights (UN Comtrade)

Dual-Stream Model:
├── Temporal Stream (Bi-LSTM)
├── Contextual Stream (Dense Layers)
├── Multi-Head Attention
├── Feature Fusion
└── Classification + Regression Heads
```

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Data Collection**
   ```bash
   python src/data_collector.py
   ```

3. **Train Model**
   ```bash
   python src/train.py --config config.yaml
   ```

### Server Deployment (DGX A100)

1. **Deploy to Server**
   ```bash
   python deploy_to_server.py deploy
   ```

2. **Start Training**
   ```bash
   python deploy_to_server.py train
   ```

3. **Monitor Progress**
   ```bash
   python deploy_to_server.py monitor
   ```

4. **Download Results**
   ```bash
   python deploy_to_server.py download
   ```

## Server Configuration

The system is optimized for your DGX A100 server:
- **Server**: `srmist54@172.16.0.32`
- **GPUs**: 2x A100 (CUDA_VISIBLE_DEVICES=0,1)
- **Distributed Training**: PyTorch DDP with 2 processes
- **Memory**: Optimized for A100's 40GB VRAM

## Data Sources

### Free APIs Used
- **Yahoo Finance**: OHLCV data for energy commodities
- **GDELT**: Daily geopolitical events and actor relationships
- **NewsAPI**: Energy-related news headlines (requires free API key)
- **Policy Uncertainty**: GPR and EPU indices
- **UN Comtrade**: Trade weights between countries

### Commodities Covered
- Crude Oil (CL=F)
- Natural Gas (NG=F)
- Coal (CO1=F)
- Electricity (EL=F)

## Model Details

### Dual-Stream Architecture
- **Temporal Stream**: 30-day lookback with Bi-LSTM
- **Contextual Stream**: Dense layers for static features
- **Attention**: Multi-head attention for feature importance
- **Output**: 3-class classification (Up/Down/Stable) + regression

### Training Configuration
- **Sequence Length**: 30 days
- **Forecast Horizon**: 1 day
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 100 (with early stopping)
- **Optimizer**: AdamW with weight decay

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Macro-averaged scores
- **ROC-AUC**: Multi-class AUC score
- **Confusion Matrix**: Class distribution analysis
- **Attention Visualization**: Feature importance plots

## File Structure

```
├── src/
│   ├── data_collector.py      # Data fetching from APIs
│   ├── feature_engineering.py # Technical indicators & ripple effects
│   ├── model.py              # Dual-Stream Attention Bi-LSTM
│   └── train.py              # Training pipeline
├── config.yaml               # Model and data parameters
├── requirements.txt          # Python dependencies
├── deploy_to_server.py       # Server deployment script
└── README.md                 # This file
```

## Usage Examples

### Basic Training
```python
from src.train import MultiGPUTrainer
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize trainer
trainer = MultiGPUTrainer(config, use_ddp=True)

# Train model
results = trainer.train()
```

### Custom Configuration
```yaml
model:
  sequence_length: 30
  hidden_size: 128
  num_layers: 2
  attention_heads: 8
  learning_rate: 0.001
  batch_size: 32
  epochs: 100

data_sources:
  yahoo_finance:
    symbols: ["CL=F", "NG=F", "CO1=F", "EL=F"]
    period: "5y"
```

## Server Commands

### SSH Access
```bash
ssh srmist54@172.16.0.32
cd /home/srmist54/geopolitical_energy_forecasting
```

### Manual Training
```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 src/train.py --config config.yaml --gpus 2
```

### Monitor GPU Usage
```bash
watch -n 5 nvidia-smi
```

## Results

After training, the system generates:
- `best_model.pth`: Trained model weights
- `results.json`: Evaluation metrics and training history
- `training_history.png`: Loss curves
- `confusion_matrix.png`: Classification performance
- `attention_weights.png`: Feature importance visualization

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config.yaml
   - Use gradient accumulation
   - Enable mixed precision training

2. **Data Collection Errors**
   - Check internet connection
   - Verify API keys for NewsAPI
   - Use fallback synthetic data

3. **Distributed Training Issues**
   - Ensure all GPUs are visible: `nvidia-smi`
   - Check NCCL backend compatibility
   - Verify PyTorch DDP setup

### Performance Optimization

1. **Multi-GPU Setup**
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1
   export OMP_NUM_THREADS=4
   ```

2. **Memory Optimization**
   - Use gradient checkpointing
   - Enable mixed precision
   - Optimize data loading

3. **Training Speed**
   - Increase batch size
   - Use multiple workers
   - Enable pin_memory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{geopolitical_energy_forecasting,
  title={Forecasting Ripple Effects of Geopolitical Instability on Global Energy Commodity Prices},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/geopolitical-energy-forecasting}
}
```

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.
