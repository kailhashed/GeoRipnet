"""
LocalDeltaModel: Predicts country-specific deviations from benchmark prices.

This module implements a Temporal CNN with country embeddings to capture
local factors affecting oil prices (FX, GDP, CPI, policy, sentiment, etc.).

Mathematical Formulation:
    Δ_c(t) = g_θ(x_c(t))
    where x_c includes: FX_c, GDP_c, CPI_c, policy_c, sentiment_c, lagged_deltas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """
    Temporal Convolutional Block with dilated convolutions and residual connections.
    
    Uses causal (masked) convolutions to prevent information leakage from future.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # Causal padding
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, seq_len)
        
        Returns:
            out: (batch, out_channels, seq_len)
        """
        # First convolution with causal masking
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Remove future padding
        out = F.gelu(self.norm1(out))
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.norm2(out)
        
        # Residual connection
        res = x if self.residual is None else self.residual(x)
        
        # Match sequence length (in case of size mismatch)
        if out.size(2) != res.size(2):
            res = res[:, :, :out.size(2)]
        
        out = F.gelu(out + res)
        return out


class CountryEmbedding(nn.Module):
    """
    Learnable country embeddings to capture country-specific characteristics.
    
    Each country gets a unique embedding that captures its structural properties
    (economic size, oil dependency, political stability, etc.).
    """
    
    def __init__(self, num_countries: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_countries, embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, country_ids):
        """
        Args:
            country_ids: (batch,) - Integer country IDs
        
        Returns:
            embeddings: (batch, embedding_dim)
        """
        emb = self.embedding(country_ids)
        emb = F.gelu(self.projection(emb))
        return emb


class LocalDeltaModel(nn.Module):
    """
    Country-specific deviation predictor using Temporal CNN.
    
    Predicts Δ_c(t+1) - the local deviation from the benchmark price
    based on country-specific features and learned country embeddings.
    
    Args:
        input_dim: Number of input features per time step
        num_countries: Total number of countries in the dataset
        country_embedding_dim: Dimension of country embeddings (default: 64)
        hidden_channels: List of hidden channel sizes for TCN (default: [128, 256, 256, 128])
        kernel_size: Convolution kernel size (default: 3)
        dropout: Dropout probability (default: 0.3)
        seq_len: Input sequence length (default: 30)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_countries: int,
        country_embedding_dim: int = 64,
        hidden_channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
        seq_len: int = 30
    ):
        super().__init__()
        self.input_dim = input_dim
        self.country_embedding_dim = country_embedding_dim
        
        if hidden_channels is None:
            hidden_channels = [128, 256, 256, 128]
        
        # Country embeddings
        self.country_embedding = CountryEmbedding(num_countries, country_embedding_dim)
        
        # Project country embeddings to all time steps
        self.country_projection = nn.Linear(country_embedding_dim, seq_len * country_embedding_dim)
        
        # Input fusion: combine features with country embeddings
        fusion_dim = input_dim + country_embedding_dim
        
        # Temporal Convolutional Network
        self.tcn_blocks = nn.ModuleList()
        in_ch = fusion_dim
        
        for i, out_ch in enumerate(hidden_channels):
            dilation = 2 ** i  # Exponentially increasing dilation
            self.tcn_blocks.append(
                TemporalConvBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch
        
        # Feature extraction network (for sentiment, macro indicators)
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_channels[-1], 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism over temporal features
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output heads
        self.delta_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, country_ids, return_uncertainty=False):
        """
        Forward pass through the local delta model.
        
        Args:
            x: (batch, seq_len, input_dim) - Country-specific time series features
            country_ids: (batch,) - Integer country IDs
            return_uncertainty: If True, return uncertainty estimates
        
        Returns:
            delta: (batch, 1) - Predicted deviation from benchmark
            uncertainty: (batch, 1) - Optional uncertainty estimate
        """
        batch_size, seq_len, _ = x.shape
        
        # Get country embeddings
        country_emb = self.country_embedding(country_ids)  # (batch, country_embedding_dim)
        
        # Project country embeddings across time
        country_temporal = self.country_projection(country_emb)  # (batch, seq_len * emb_dim)
        country_temporal = country_temporal.view(batch_size, seq_len, self.country_embedding_dim)
        
        # Fuse input features with country embeddings
        x_fused = torch.cat([x, country_temporal], dim=-1)  # (batch, seq_len, fusion_dim)
        
        # Reshape for temporal convolution: (batch, channels, seq_len)
        x_fused = x_fused.transpose(1, 2)
        
        # Pass through TCN blocks
        for tcn_block in self.tcn_blocks:
            x_fused = tcn_block(x_fused)
        
        # Transpose back: (batch, seq_len, channels)
        x_fused = x_fused.transpose(1, 2)
        
        # Extract features
        features = self.feature_extractor(x_fused)  # (batch, seq_len, 128)
        
        # Temporal attention
        attn_out, attn_weights = self.temporal_attention(
            features, features, features
        )  # (batch, seq_len, 128)
        
        # Use last time step for prediction
        h = attn_out[:, -1, :]  # (batch, 128)
        
        # Predict delta
        delta = self.delta_head(h)  # (batch, 1)
        
        result = {'delta': delta}
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(h)  # (batch, 1)
            result['uncertainty'] = uncertainty
        
        return result
    
    def compute_lagged_features(self, deltas, lags=[1, 2, 3, 7, 14]):
        """
        Compute lagged delta features for autoregressive modeling.
        
        Args:
            deltas: (batch, seq_len) - Historical delta values
            lags: List of lag values to compute
        
        Returns:
            lagged_features: (batch, seq_len, len(lags))
        """
        batch_size, seq_len = deltas.shape
        lagged_features = []
        
        for lag in lags:
            if lag < seq_len:
                # Shift deltas by lag and pad with zeros
                lagged = F.pad(deltas[:, :-lag], (lag, 0), value=0.0)
            else:
                lagged = torch.zeros_like(deltas)
            lagged_features.append(lagged.unsqueeze(-1))
        
        return torch.cat(lagged_features, dim=-1)


class AutoregressiveDeltaModel(nn.Module):
    """
    Extended LocalDeltaModel with explicit autoregressive component.
    
    Mathematical formulation:
        Δ_c(t+1) = α_c * Δ_c(t) + β * f_θ(x_c(t))
    
    where α_c is a learnable country-specific persistence parameter.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_countries: int,
        **kwargs
    ):
        super().__init__()
        self.num_countries = num_countries
        
        # Base delta model
        self.delta_model = LocalDeltaModel(input_dim, num_countries, **kwargs)
        
        # Learnable persistence parameters (one per country)
        self.alpha = nn.Parameter(torch.ones(num_countries) * 0.5)
        
        # Learnable innovation weight
        self.beta = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, country_ids, prev_delta, return_uncertainty=False):
        """
        Args:
            x: (batch, seq_len, input_dim)
            country_ids: (batch,)
            prev_delta: (batch, 1) - Previous time step delta
            return_uncertainty: If True, return uncertainty
        
        Returns:
            delta: (batch, 1) - Predicted delta with AR component
        """
        # Get base prediction from delta model
        result = self.delta_model(x, country_ids, return_uncertainty)
        base_delta = result['delta']
        
        # Apply autoregressive formula
        alpha_c = torch.sigmoid(self.alpha[country_ids]).unsqueeze(-1)  # (batch, 1)
        delta = alpha_c * prev_delta + self.beta * base_delta
        
        result['delta'] = delta
        result['alpha'] = alpha_c
        
        return result


if __name__ == "__main__":
    # Test the LocalDeltaModel
    print("Testing LocalDeltaModel...")
    
    batch_size = 16
    seq_len = 30
    input_dim = 40  # FX, GDP, CPI, policy index, inventories, sentiment, etc.
    num_countries = 50
    
    model = LocalDeltaModel(
        input_dim=input_dim,
        num_countries=num_countries,
        country_embedding_dim=64,
        hidden_channels=[128, 256, 256, 128],
        dropout=0.3
    )
    
    # Random input
    x = torch.randn(batch_size, seq_len, input_dim)
    country_ids = torch.randint(0, num_countries, (batch_size,))
    
    # Forward pass
    output = model(x, country_ids, return_uncertainty=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Country IDs shape: {country_ids.shape}")
    print(f"Delta predictions shape: {output['delta'].shape}")
    print(f"Uncertainty shape: {output['uncertainty'].shape}")
    
    # Test AutoregressiveDeltaModel
    print("\nTesting AutoregressiveDeltaModel...")
    ar_model = AutoregressiveDeltaModel(input_dim, num_countries, dropout=0.3)
    prev_delta = torch.randn(batch_size, 1)
    
    ar_output = ar_model(x, country_ids, prev_delta, return_uncertainty=True)
    
    print(f"AR Delta predictions shape: {ar_output['delta'].shape}")
    print(f"Alpha (persistence) shape: {ar_output['alpha'].shape}")
    
    # Test lagged features
    deltas = torch.randn(batch_size, seq_len)
    lagged = model.compute_lagged_features(deltas)
    print(f"\nLagged features shape: {lagged.shape}")
    
    print("\n✓ LocalDeltaModel test passed!")

