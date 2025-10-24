"""
RippleNet-TFT Model Implementation
Temporal Fusion Transformer with Ripple Graph and News Encoder integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import yaml
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for TFT"""
    
    def __init__(self, input_size: int, hidden_size: int, num_variables: int, dropout: float = 0.1):
        super(VariableSelectionNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_variables = num_variables
        
        # Variable selection weights
        self.variable_selection = nn.Linear(input_size, num_variables)
        
        # Variable processing
        self.variable_processing = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism
        self.gate = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_variables, input_size)
        batch_size, seq_len, num_vars, input_size = x.shape
        
        # Reshape for processing
        x_flat = x.view(batch_size * seq_len, num_vars, input_size)
        
        # Variable selection weights
        selection_weights = F.softmax(self.variable_selection(x_flat), dim=-1)
        
        # Process variables
        processed_vars = self.variable_processing(x_flat)
        
        # Apply gating
        gated_vars = torch.sigmoid(self.gate(processed_vars)) * processed_vars
        
        # Apply selection weights
        selected_vars = selection_weights.unsqueeze(-1) * gated_vars
        
        # Reshape back - handle multiple variables correctly
        # For multi-feature case, we need to handle the reshaping differently
        if num_vars == 1:
            output = selected_vars.view(batch_size, seq_len, num_vars, self.hidden_size)
        else:
            # For multiple features, reshape to [batch, seq, features, hidden]
            output = selected_vars.view(batch_size, seq_len, num_vars, self.hidden_size)
        
        return output, selection_weights

class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer core"""
    
    def __init__(self, config: Dict):
        super(TemporalFusionTransformer, self).__init__()
        
        self.config = config
        self.hidden_size = config['model']['tft']['hidden_size']
        self.lstm_layers = config['model']['tft']['lstm_layers']
        self.attention_head_size = config['model']['tft']['attention_head_size']
        self.dropout = config['model']['tft']['dropout']
        
        # Input dimensions (will be set during forward pass)
        self.past_target_dim = None
        self.observed_cov_dim = None
        self.known_future_dim = None
        
        # LSTM for temporal processing
        self.past_target_lstm = nn.LSTM(
            input_size=1,  # Will be set dynamically
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0
        )
        
        # Variable selection networks
        self.past_target_vsn = None
        self.observed_cov_vsn = None
        self.known_future_vsn = None
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.attention_head_size,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.hidden_size)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.attention_head_size,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.dropout,
                batch_first=True
            )
            for _ in range(2)  # Number of transformer layers
        ])
        
        # Output layers
        self.output_projection = nn.Linear(self.hidden_size, 1)
        
        # Linear layers for processing
        self.past_target_linear = None
        self.observed_cov_linear = None
        
    def _initialize_variable_selection_networks(self, past_target_dim: int, 
                                              observed_cov_dim: int, 
                                              known_future_dim: int):
        """Initialize variable selection networks"""
        # Initialize linear layers for processing
        if self.past_target_linear is None:
            self.past_target_linear = nn.Linear(past_target_dim, self.hidden_size)
        
        if self.observed_cov_linear is None and observed_cov_dim > 0:
            self.observed_cov_linear = nn.Linear(observed_cov_dim, self.hidden_size)
        
        if self.past_target_vsn is None:
            self.past_target_vsn = VariableSelectionNetwork(
                input_size=past_target_dim,
                hidden_size=self.hidden_size,
                num_variables=1,  # Single target variable
                dropout=self.dropout
            )
        
        if self.observed_cov_vsn is None and observed_cov_dim > 0:
            self.observed_cov_vsn = VariableSelectionNetwork(
                input_size=observed_cov_dim,
                hidden_size=self.hidden_size,
                num_variables=1,
                dropout=self.dropout
            )
        
        if self.known_future_vsn is None and known_future_dim > 0:
            self.known_future_vsn = VariableSelectionNetwork(
                input_size=known_future_dim,
                hidden_size=self.hidden_size,
                num_variables=1,
                dropout=self.dropout
            )
    
    def forward(self, past_target: torch.Tensor, 
                observed_covariates: torch.Tensor = None,
                known_future: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of TFT
        
        Args:
            past_target: (batch_size, seq_len, past_target_dim)
            observed_covariates: (batch_size, seq_len, observed_cov_dim)
            known_future: (batch_size, forecast_horizon, known_future_dim)
        
        Returns:
            predictions: (batch_size, forecast_horizon, 1)
        """
        batch_size, seq_len = past_target.shape[:2]
        
        # Initialize variable selection networks
        self._initialize_variable_selection_networks(
            past_target.shape[-1],
            observed_covariates.shape[-1] if observed_covariates is not None else 0,
            known_future.shape[-1] if known_future is not None else 0
        )
        
        # Process past target - use simple linear transformation for multi-feature case
        # Flatten features and apply linear transformation
        batch_size, seq_len, num_features = past_target.shape
        past_target_flat = past_target.view(batch_size * seq_len, num_features)
        past_target_processed = self.past_target_linear(past_target_flat)
        past_target_processed = past_target_processed.view(batch_size, seq_len, -1)
        
        # Process observed covariates - use simple linear transformation
        if observed_covariates is not None:
            obs_batch_size, obs_seq_len, obs_num_features = observed_covariates.shape
            observed_flat = observed_covariates.view(obs_batch_size * obs_seq_len, obs_num_features)
            observed_processed = self.observed_cov_linear(observed_flat)
            observed_processed = observed_processed.view(obs_batch_size, obs_seq_len, -1)
        else:
            observed_processed = torch.zeros_like(past_target_processed)
        
        # Combine past target and observed covariates
        combined_input = past_target_processed + observed_processed
        
        # Add positional encoding
        combined_input = self.pos_encoding(combined_input.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            combined_input = transformer_layer(combined_input)
        
        # Get the last timestep for prediction
        last_timestep = combined_input[:, -1, :]  # (batch_size, hidden_size)
        
        # Return hidden representation instead of final prediction
        # The regression head will handle the final prediction
        return last_timestep  # (batch_size, hidden_size)

class RippleNetTFT(nn.Module):
    """Complete RippleNet-TFT model"""
    
    def __init__(self, config: Dict):
        super(RippleNetTFT, self).__init__()
        
        self.config = config
        self.hidden_size = config['model']['tft']['hidden_size']
        
        # Ripple Graph Encoder
        self.ripple_encoder = RippleGraphEncoder(config)
        
        # News Encoder
        self.news_encoder = NewsEncoder(config)
        
        # TFT Core
        self.tft = TemporalFusionTransformer(config)
        
        # Regression Head
        self.regression_head = RegressionHead(config)
        
    def forward(self, past_target: torch.Tensor,
                observed_covariates: torch.Tensor = None,
                known_future: torch.Tensor = None,
                ripple_embeddings: torch.Tensor = None,
                news_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of RippleNet-TFT
        
        Args:
            past_target: Historical target values
            observed_covariates: Observed covariates
            known_future: Known future inputs
            ripple_embeddings: Ripple graph embeddings
            news_embeddings: News embeddings
        
        Returns:
            predictions: Model predictions
        """
        # Process ripple embeddings
        if ripple_embeddings is not None:
            ripple_processed = self.ripple_encoder(ripple_embeddings)
        else:
            ripple_processed = None
        
        # Process news embeddings
        if news_embeddings is not None:
            news_processed = self.news_encoder(news_embeddings)
        else:
            news_processed = None
        
        # Combine embeddings with observed covariates
        if observed_covariates is not None:
            if ripple_processed is not None:
                observed_covariates = torch.cat([observed_covariates, ripple_processed], dim=-1)
            if news_processed is not None:
                observed_covariates = torch.cat([observed_covariates, news_processed], dim=-1)
        
        # TFT forward pass
        tft_output = self.tft(past_target, observed_covariates, known_future)
        
        # TFT now outputs hidden representation directly
        # Regression head
        predictions = self.regression_head(tft_output)
        
        return predictions

class RippleGraphEncoder(nn.Module):
    """Ripple Graph Encoder"""
    
    def __init__(self, config: Dict):
        super(RippleGraphEncoder, self).__init__()
        
        self.embedding_dim = config['model']['ripple_graph']['embedding_dim']
        self.hidden_size = config['model']['tft']['hidden_size']
        
        # Simple MLP for ripple embeddings
        self.ripple_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
    
    def forward(self, ripple_embeddings: torch.Tensor) -> torch.Tensor:
        """Process ripple embeddings"""
        return self.ripple_mlp(ripple_embeddings)

class NewsEncoder(nn.Module):
    """News Encoder"""
    
    def __init__(self, config: Dict):
        super(NewsEncoder, self).__init__()
        
        self.embedding_dim = config['model']['news_encoder']['embedding_dim']
        self.hidden_size = config['model']['tft']['hidden_size']
        
        # Simple MLP for news embeddings
        self.news_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
    
    def forward(self, news_embeddings: torch.Tensor) -> torch.Tensor:
        """Process news embeddings"""
        return self.news_mlp(news_embeddings)

class RegressionHead(nn.Module):
    """Regression head for final predictions"""
    
    def __init__(self, config: Dict):
        super(RegressionHead, self).__init__()
        
        self.hidden_size = config['model']['tft']['hidden_size']
        self.hidden_layers = config['model']['regression_head']['hidden_layers']
        self.dropout = config['model']['regression_head']['dropout']
        self.activation = config['model']['regression_head']['activation']
        
        # Build regression head
        layers = []
        input_size = self.hidden_size
        
        for hidden_size in self.hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU() if self.activation == 'relu' else nn.Tanh(),
                nn.Dropout(self.dropout)
            ])
            input_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(input_size, 1))
        
        self.regression_head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of regression head"""
        return self.regression_head(x)

def create_model(config: Dict) -> RippleNetTFT:
    """Create RippleNet-TFT model"""
    logger.info("Creating RippleNet-TFT model")
    
    model = RippleNetTFT(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    model.apply(init_weights)
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    return model

def main():
    """Main function for model creation"""
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config)
    
    # Test forward pass
    batch_size = 32
    seq_len = 60
    past_target_dim = 10
    observed_cov_dim = 20
    known_future_dim = 8
    
    past_target = torch.randn(batch_size, seq_len, past_target_dim)
    observed_covariates = torch.randn(batch_size, seq_len, observed_cov_dim)
    known_future = torch.randn(batch_size, 1, known_future_dim)
    
    with torch.no_grad():
        output = model(past_target, observed_covariates, known_future)
    
    print(f"Model output shape: {output.shape}")
    print("Model creation completed successfully!")

if __name__ == "__main__":
    main()
