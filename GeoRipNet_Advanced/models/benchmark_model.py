"""
BenchmarkModel: Predicts global oil benchmark prices (WTI, Brent, Oman/Dubai).

This module implements a Temporal Fusion Transformer-based model for predicting
base oil prices that serve as the foundation for country-specific predictions.

Mathematical Formulation:
    B_c(t) = Σ_s W^(b)_{c,s} * P_s(t)
    where W^(b)_{c,s} = V_{c←s} / Σ_{s'} V_{c←s'}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalFusionBlock(nn.Module):
    """
    Temporal Fusion Transformer block with multi-head attention and gating.
    
    Combines temporal self-attention with feed-forward networks and
    gated residual connections for effective time-series modeling.
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Gated Linear Unit (GLU) components
        self.gate_linear1 = nn.Linear(d_model, d_model)
        self.gate_linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: (seq_len, batch, d_model)
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask
        """
        # Multi-head self-attention with gating
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        gate = torch.sigmoid(self.gate_linear1(src))
        src = src + self.dropout1(gate * src2)
        src = self.norm1(src)
        
        # Feed-forward with gating
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        gate = torch.sigmoid(self.gate_linear2(F.gelu(self.linear1(src))))
        src = src + self.dropout2(gate * src2)
        src = self.norm2(src)
        
        return src, attn_weights


class BenchmarkModel(nn.Module):
    """
    Global benchmark oil price predictor using Temporal Fusion Transformer.
    
    Predicts next-day or next-week prices for WTI, Brent, and Oman/Dubai benchmarks
    based on global macro indicators, OPEC variables, and historical prices.
    
    Args:
        input_dim: Number of input features (macro indicators + price history)
        d_model: Hidden dimension size (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of TFT blocks (default: 4)
        dim_feedforward: Feed-forward network dimension (default: 1024)
        dropout: Dropout probability (default: 0.3)
        num_benchmarks: Number of benchmark prices to predict (default: 3)
        seq_len: Input sequence length (default: 30)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        num_benchmarks: int = 3,
        seq_len: int = 30
    ):
        super().__init__()
        self.d_model = d_model
        self.num_benchmarks = num_benchmarks
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        # Temporal Fusion Transformer blocks
        self.tft_blocks = nn.ModuleList([
            TemporalFusionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Variable selection network (learns feature importance)
        self.variable_selection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        
        # Output heads for each benchmark
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
            for _ in range(num_benchmarks)
        ])
        
        # Quantile regression heads (for uncertainty estimation)
        self.quantile_heads = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
                )
                for _ in range(3)  # 10th, 50th, 90th percentiles
            ])
            for _ in range(num_benchmarks)
        ])
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights using Xavier/Kaiming initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, return_attention=False, return_quantiles=False):
        """
        Forward pass through the benchmark model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            return_attention: If True, return attention weights
            return_quantiles: If True, return quantile predictions
        
        Returns:
            predictions: (batch, num_benchmarks) - Point predictions
            quantiles: (batch, num_benchmarks, 3) - Optional quantile predictions
            attention_weights: List of attention weight tensors - Optional
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection: (batch, seq_len, input_dim) -> (seq_len, batch, d_model)
        x = self.input_projection(x).transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Variable selection (feature importance)
        selection_weights = self.variable_selection(x.mean(dim=0))  # (batch, d_model)
        x = x * selection_weights.unsqueeze(0)  # Broadcast over seq_len
        
        # Pass through TFT blocks
        attention_weights = []
        for tft_block in self.tft_blocks:
            x, attn = tft_block(x)
            if return_attention:
                attention_weights.append(attn)
        
        # Extract final hidden state: (batch, d_model)
        h = x[-1]  # Last time step
        
        # Generate predictions for each benchmark
        predictions = []
        quantiles = []
        
        for i in range(self.num_benchmarks):
            # Point prediction
            pred = self.output_heads[i](h)  # (batch, 1)
            predictions.append(pred)
            
            # Quantile predictions (10th, 50th, 90th percentiles)
            if return_quantiles:
                q_preds = [q_head(h) for q_head in self.quantile_heads[i]]
                quantiles.append(torch.cat(q_preds, dim=-1))  # (batch, 3)
        
        predictions = torch.cat(predictions, dim=-1)  # (batch, num_benchmarks)
        
        result = {'predictions': predictions}
        
        if return_quantiles:
            quantiles = torch.stack(quantiles, dim=1)  # (batch, num_benchmarks, 3)
            result['quantiles'] = quantiles
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result
    
    def compute_trade_weighted_benchmark(self, benchmark_prices, trade_weights):
        """
        Compute country-specific benchmark prices using trade weights.
        
        Mathematical formulation:
            W^(b)_{c,s} = V_{c←s} / Σ_{s'} V_{c←s'}
            B_c(t) = Σ_s W^(b)_{c,s} * P_s(t)
        
        Args:
            benchmark_prices: (batch, num_benchmarks) - Predicted benchmark prices
            trade_weights: (num_countries, num_benchmarks) - Normalized trade volumes
        
        Returns:
            country_benchmarks: (batch, num_countries) - Trade-weighted benchmarks
        """
        # Ensure trade weights are normalized
        trade_weights_normalized = trade_weights / (trade_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute weighted combination: (batch, num_countries)
        country_benchmarks = torch.matmul(benchmark_prices, trade_weights_normalized.T)
        
        return country_benchmarks


if __name__ == "__main__":
    # Test the BenchmarkModel
    print("Testing BenchmarkModel...")
    
    batch_size = 16
    seq_len = 30
    input_dim = 50  # e.g., 20 macro indicators + 30 price history features
    
    model = BenchmarkModel(
        input_dim=input_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.3,
        num_benchmarks=3
    )
    
    # Random input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x, return_attention=True, return_quantiles=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Quantiles shape: {output['quantiles'].shape}")
    print(f"Number of attention weight tensors: {len(output['attention_weights'])}")
    
    # Test trade-weighted benchmark computation
    num_countries = 20
    trade_weights = torch.rand(num_countries, 3)  # Random trade weights
    country_benchmarks = model.compute_trade_weighted_benchmark(
        output['predictions'], trade_weights
    )
    print(f"Country benchmarks shape: {country_benchmarks.shape}")
    
    print("\n✓ BenchmarkModel test passed!")

