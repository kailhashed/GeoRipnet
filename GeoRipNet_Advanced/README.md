
# GeoRipNet Advanced: Maximum-Accuracy Deep Learning Framework

**State-of-the-art deep learning system for country-level oil price prediction with ripple effect propagation.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

GeoRipNet Advanced is a production-ready PyTorch implementation for predicting country-specific oil prices by modeling:

1. **Global benchmark prices** (WTI, Brent, Oman/Dubai)
2. **Local deviations** due to country-specific factors
3. **Ripple propagation** of shocks through trade networks using Graph Neural Networks

### Key Features

✅ **Advanced Architectures**
- Temporal Fusion Transformer (TFT) for benchmark prediction
- Temporal CNN with country embeddings for local modeling
- Graph Attention Networks (GAT) for ripple propagation

✅ **State-of-the-Art Training**
- Mixed Precision Training (AMP)
- Stochastic Weight Averaging (SWA)
- Advanced LR scheduling (OneCycleLR, CosineAnnealing)
- Gradient clipping and early stopping
- Model checkpointing and resuming

✅ **Comprehensive Evaluation**
- Multi-objective loss (Huber + Directional + Correlation + Quantile)
- Per-country and aggregate metrics
- Uncertainty quantification (MC Dropout, Quantile Regression)
- Extensive visualization suite

---

## 📐 Mathematical Formulation

### Country Price Prediction

For each country `c` at time `t`:

```
P_c(t) = B_c(t) + Δ_c(t)
```

Where:
- **B_c(t)** = Trade-weighted benchmark price
  ```
  B_c(t) = Σ_s W^(b)_{c,s} * P_s(t)
  W^(b)_{c,s} = V_{c←s} / Σ_{s'} V_{c←s'}
  ```

- **Δ_c(t)** = Country-specific deviation
  ```
  Δ_c(t) = g_θ(x_c(t))
  ```

### Ripple Propagation

Shock propagation through graph neural networks:

```
Δ(t+1) = α ⊙ Δ(t) + β (W ⊙ M) φ(Δ(t), E(t))
```

Where:
- `α`: Learnable persistence parameters (per country)
- `β`: Global propagation weight
- `W`: Trade adjacency matrix
- `M`: Learned attention mask
- `φ`: Message function combining deltas and event embeddings
- `E`: News/event embeddings

### Multi-Objective Loss

```
L_total = λ₁·L_huber + λ₂·L_dir + λ₃·L_corr + λ₄·L_quantile
```

**Default weights:** λ₁=1.0, λ₂=0.3, λ₃=0.2, λ₄=0.0

---

## 🏗️ Architecture

```
GeoRipNet_Advanced/
├── models/
│   ├── benchmark_model.py       # Global benchmark predictor (TFT)
│   ├── local_delta_model.py     # Country-specific deviation model (Temporal CNN)
│   ├── ripple_gnn.py            # Ripple propagation (GAT)
│   └── georipnet_model.py       # Complete integrated model
├── losses/
│   └── multi_objective_loss.py  # Composite loss functions
├── data/
│   ├── data_loader.py           # PyTorch datasets and dataloaders
│   └── preprocessing.py         # Feature engineering and scaling
├── training/
│   └── trainer.py               # Advanced training loop
├── utils/
│   ├── metrics.py               # Evaluation metrics
│   └── visualization.py         # Plotting utilities
├── train_geo_ripnet.py          # End-to-end training script
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd GeoRipNet_Advanced

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib seaborn tqdm pyyaml
```

### Basic Usage

```python
from models import GeoRipNetModel
from losses.multi_objective_loss import MultiObjectiveLoss
from training.trainer import GeoRipNetTrainer
import torch

# Initialize model
model = GeoRipNetModel(
    benchmark_input_dim=50,
    num_benchmarks=3,
    local_input_dim=40,
    num_countries=20,
    event_embed_dim=384,
    d_model=256,
    dropout=0.3
)

# Setup training
criterion = MultiObjectiveLoss(
    lambda_huber=1.0,
    lambda_directional=0.3,
    lambda_correlation=0.2
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)

trainer = GeoRipNetTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda',
    use_amp=True,
    use_swa=True
)

# Train
trainer.fit(train_loader, val_loader, num_epochs=100)
```

### Full Training Pipeline

```bash
python train_geo_ripnet.py \
    --config config.yaml \
    --data_dir data \
    --output_dir outputs \
    --num_epochs 100 \
    --batch_size 32 \
    --device cuda \
    --use_amp \
    --use_swa
```

---

## 📊 Model Components

### 1. BenchmarkModel

**Architecture:** Temporal Fusion Transformer (TFT)

**Input:**
- Global macro indicators (OPEC production, inventories, etc.)
- Historical benchmark prices
- Sequence length: 30 days (configurable)

**Output:**
- Predicted WTI, Brent, Oman/Dubai prices
- Quantile predictions (10th, 50th, 90th percentiles)
- Attention weights

**Key Features:**
- Multi-head self-attention (8 heads)
- Gated residual connections
- Variable selection network
- Positional encoding

```python
from models import BenchmarkModel

benchmark_model = BenchmarkModel(
    input_dim=50,
    d_model=256,
    nhead=8,
    num_layers=4,
    dropout=0.3,
    num_benchmarks=3
)

output = benchmark_model(benchmark_features, return_quantiles=True)
# output['predictions']: (batch, 3)
# output['quantiles']: (batch, 3, 3)
```

### 2. LocalDeltaModel

**Architecture:** Temporal Convolutional Network (TCN) with Country Embeddings

**Input:**
- Country-specific features (FX, GDP, CPI, policy index, sentiment)
- Country ID (for learned embeddings)
- Sequence length: 30 days

**Output:**
- Local deviation from benchmark
- Uncertainty estimates

**Key Features:**
- Causal dilated convolutions
- Learnable country embeddings (64-dim)
- Multi-head temporal attention
- Autoregressive extension available

```python
from models import LocalDeltaModel

local_model = LocalDeltaModel(
    input_dim=40,
    num_countries=20,
    country_embedding_dim=64,
    hidden_channels=[128, 256, 256, 128],
    dropout=0.3
)

output = local_model(local_features, country_ids, return_uncertainty=True)
# output['delta']: (batch, 1)
# output['uncertainty']: (batch, 1)
```

### 3. RippleGNNLayer

**Architecture:** Graph Attention Network (GAT) with Message Passing

**Input:**
- Delta values (batch, num_countries, 1)
- Event embeddings (batch, num_countries, 384)
- Trade adjacency matrix (batch, num_countries, num_countries)

**Output:**
- Propagated deltas incorporating neighbor influences
- Attention weights showing ripple pathways

**Key Features:**
- Multi-head graph attention (4 heads)
- Learnable message function
- Adaptive persistence parameters
- Influence matrix computation

```python
from models import RippleGNNLayer

ripple_gnn = RippleGNNLayer(
    num_countries=20,
    delta_dim=1,
    event_embed_dim=384,
    hidden_dim=128,
    num_heads=4,
    num_gnn_layers=2
)

deltas_next, attention = ripple_gnn(
    deltas, event_embeddings, trade_adjacency, return_attention=True
)
```

### 4. GeoRipNetModel (Complete Model)

**Combines all three components** with:
- Trade-weighted benchmark computation
- Ensemble prediction support
- MC Dropout for uncertainty
- Output calibration layer

```python
from models import GeoRipNetModel

model = GeoRipNetModel(
    benchmark_input_dim=50,
    num_benchmarks=3,
    local_input_dim=40,
    num_countries=20,
    event_embed_dim=384,
    use_ensemble=False  # Set True for ensemble of 3 local models
)

output = model(
    benchmark_features, local_features, country_ids,
    event_embeddings, trade_adjacency, trade_weights,
    return_components=True, return_uncertainty=True
)

# output['predictions']: (batch, num_countries)
# output['benchmark_prices']: (batch, 3)
# output['country_benchmarks']: (batch, num_countries)
# output['base_deltas']: (batch, num_countries)
# output['propagated_deltas']: (batch, num_countries)
# output['uncertainties']: (batch, num_countries)
```

---

## 📉 Loss Functions

### 1. Huber Loss
Robust to outliers, smooth L1 loss for price prediction.

### 2. Directional Loss
Penalizes incorrect sign predictions (up/down movement).

### 3. Ripple Correlation Loss
Preserves correlation structure across countries in delta predictions.

### 4. Quantile Loss (Optional)
Pinball loss for uncertainty quantification.

```python
from losses.multi_objective_loss import MultiObjectiveLoss

criterion = MultiObjectiveLoss(
    lambda_huber=1.0,
    lambda_directional=0.3,
    lambda_correlation=0.2,
    lambda_quantile=0.0,  # Disabled by default
    huber_delta=1.0
)

losses = criterion(
    predictions, targets, prev_targets,
    predicted_deltas, true_deltas
)

# losses['total_loss']
# losses['huber_loss']
# losses['directional_loss']
# losses['correlation_loss']
# losses['directional_accuracy']
```

---

## 📈 Evaluation Metrics

### Aggregate Metrics
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)
- **Directional Accuracy** (% correct sign predictions)
- **Ripple Correlation** (Correlation structure preservation)

### Per-Country Metrics
All above metrics computed individually for each country.

### Uncertainty Metrics
- **Coverage** (% of actuals within confidence intervals)
- **Calibration Error**
- **Interval Width**

```python
from utils.metrics import MetricsCalculator

calculator = MetricsCalculator(country_names)

metrics = calculator.compute_all_metrics(
    predictions, targets, prev_targets,
    predicted_deltas, true_deltas,
    lower_bounds, upper_bounds
)

calculator.print_summary(metrics)
report = calculator.create_report(metrics)
```

---

## 📊 Visualization

### Available Plots

1. **Training History** - Loss and LR curves
2. **Predictions vs Actual** - Scatter plots per country
3. **Time Series** - Predictions overlaid on actuals
4. **Ripple Heatmap** - Country-to-country influence matrix
5. **Attention Weights** - Learned attention patterns
6. **Uncertainty Quantification** - Confidence intervals
7. **Per-Country Metrics** - Bar charts of performance
8. **Error Distribution** - Histogram and Q-Q plot

```python
from utils.visualization import GeoRipNetVisualizer

visualizer = GeoRipNetVisualizer(save_dir='plots')

# Generate comprehensive report
visualizer.create_comprehensive_report(
    predictions, targets, prev_targets,
    history, country_names,
    lower_bounds, upper_bounds
)
```

---

## ⚙️ Configuration

Create `config.yaml`:

```yaml
model:
  benchmark_input_dim: 50
  num_benchmarks: 3
  local_input_dim: 40
  event_embed_dim: 384
  d_model: 256
  local_hidden_channels: [128, 256, 256, 128]
  ripple_hidden_dim: 128
  ripple_num_heads: 4
  ripple_num_layers: 2
  dropout: 0.3
  seq_len: 30

training:
  learning_rate: 5.0e-5
  weight_decay: 1.0e-5
  gradient_clip_norm: 1.0
  swa_start_epoch: 10
  early_stopping_patience: 15
  scheduler_type: 'onecycle'

loss:
  lambda_huber: 1.0
  lambda_directional: 0.3
  lambda_correlation: 0.2
  lambda_quantile: 0.0
  huber_delta: 1.0
```

---

## 🎯 Target Performance

### Goals
- **R² ≥ 0.90** (country-wise validation)
- **MAPE ≤ 20%**
- **Directional Accuracy ≥ 75%**
- **Ripple Correlation ≥ 0.8**

### Optimization Strategy
1. Deep architectures (4+ layers)
2. Ensemble averaging (3-5 models)
3. High dropout (0.2-0.4)
4. Strong regularization (weight_decay=1e-5)
5. Advanced training techniques (AMP, SWA)

---

## 🔧 Advanced Features

### Mixed Precision Training
```python
trainer = GeoRipNetTrainer(
    model, criterion, optimizer,
    use_amp=True  # Automatic mixed precision
)
```

### Stochastic Weight Averaging
```python
trainer = GeoRipNetTrainer(
    model, criterion, optimizer,
    use_swa=True,
    swa_start_epoch=10
)
```

### Ensemble Models
```python
from models import EnsembleGeoRipNet

ensemble = EnsembleGeoRipNet(
    num_models=5,
    **model_config
)
```

### MC Dropout Uncertainty
```python
mean_pred, lower, upper = model.predict_with_confidence(
    benchmark_features, local_features, country_ids,
    event_embeddings, trade_adjacency, trade_weights,
    num_samples=10
)
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{georipnet_advanced,
  title={GeoRipNet Advanced: Deep Learning for Oil Price Prediction with Ripple Effects},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/georipnet}
}
```

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues for bugs/features.

---

## 📧 Contact

For questions or collaboration: [your.email@example.com]

---

## 🙏 Acknowledgments

- Temporal Fusion Transformer: [Lim et al., 2021]
- Graph Attention Networks: [Veličković et al., 2018]
- Stochastic Weight Averaging: [Izmailov et al., 2018]

---

**Built with PyTorch • Research-Grade • Production-Ready**

