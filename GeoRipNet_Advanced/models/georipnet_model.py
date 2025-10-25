"""
GeoRipNetModel: Complete deep learning framework for country-level oil price prediction.

This module integrates BenchmarkModel, LocalDeltaModel, and RippleGNNLayer
to predict country-specific oil prices with ripple effect propagation.

Mathematical Formulation:
    P_c(t) = B_c(t) + Δ_c(t)
    
    where:
    - B_c(t) = Σ_s W^(b)_{c,s} * P_s(t)  [trade-weighted benchmark]
    - Δ_c(t) = g_θ(x_c(t))                [local deviation]
    - Δ_c(t+1) = α_c Δ_c(t) + β Σ_i W_{i→c} φ(Δ_i(t), E_i(t))  [ripple propagation]
    
    Final prediction:
    P̂_c(t+1) = B̂_c(t+1) + Δ̂_c_prop(t+1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .benchmark_model import BenchmarkModel
from .local_delta_model import LocalDeltaModel
from .ripple_gnn import RippleGNNLayer


class GeoRipNetModel(nn.Module):
    """
    Complete GeoRipNet model for country-level oil price prediction.
    
    Combines global benchmark prediction, local delta modeling, and
    ripple effect propagation through graph neural networks.
    
    Args:
        # Benchmark model config
        benchmark_input_dim: Input features for benchmark model
        num_benchmarks: Number of benchmark prices (WTI, Brent, Oman)
        
        # Local delta model config
        local_input_dim: Input features per country for local model
        num_countries: Total number of countries
        
        # Ripple GNN config
        event_embed_dim: Dimension of news/event embeddings
        
        # Architecture config
        d_model: Hidden dimension for benchmark model (default: 256)
        local_hidden_channels: Hidden channels for local model (default: [128, 256, 256, 128])
        ripple_hidden_dim: Hidden dimension for ripple GNN (default: 128)
        ripple_num_heads: Number of attention heads in GNN (default: 4)
        ripple_num_layers: Number of GNN layers (default: 2)
        
        # Training config
        dropout: Dropout probability (default: 0.3)
        seq_len: Input sequence length (default: 30)
        
        # Ensemble config
        use_ensemble: Whether to use ensemble prediction (default: False)
        num_ensemble_models: Number of models in ensemble (default: 3)
    """
    
    def __init__(
        self,
        # Data dimensions
        benchmark_input_dim: int,
        num_benchmarks: int,
        local_input_dim: int,
        num_countries: int,
        event_embed_dim: int = 384,
        
        # Architecture
        d_model: int = 256,
        local_hidden_channels: list = None,
        ripple_hidden_dim: int = 128,
        ripple_num_heads: int = 4,
        ripple_num_layers: int = 2,
        
        # Training
        dropout: float = 0.3,
        seq_len: int = 30,
        
        # Ensemble
        use_ensemble: bool = False,
        num_ensemble_models: int = 3
    ):
        super().__init__()
        
        self.num_benchmarks = num_benchmarks
        self.num_countries = num_countries
        self.use_ensemble = use_ensemble
        self.num_ensemble_models = num_ensemble_models
        
        if local_hidden_channels is None:
            local_hidden_channels = [128, 256, 256, 128]
        
        # ============ Component 1: Global Benchmark Model ============
        self.benchmark_model = BenchmarkModel(
            input_dim=benchmark_input_dim,
            d_model=d_model,
            nhead=8,
            num_layers=4,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            num_benchmarks=num_benchmarks,
            seq_len=seq_len
        )
        
        # ============ Component 2: Local Delta Model ============
        if use_ensemble:
            # Create ensemble of local delta models
            self.local_delta_models = nn.ModuleList([
                LocalDeltaModel(
                    input_dim=local_input_dim,
                    num_countries=num_countries,
                    country_embedding_dim=64,
                    hidden_channels=local_hidden_channels,
                    dropout=dropout + 0.05 * i,  # Slight variation
                    seq_len=seq_len
                )
                for i in range(num_ensemble_models)
            ])
        else:
            self.local_delta_model = LocalDeltaModel(
                input_dim=local_input_dim,
                num_countries=num_countries,
                country_embedding_dim=64,
                hidden_channels=local_hidden_channels,
                dropout=dropout,
                seq_len=seq_len
            )
        
        # ============ Component 3: Ripple Propagation GNN ============
        self.ripple_gnn = RippleGNNLayer(
            num_countries=num_countries,
            delta_dim=1,
            event_embed_dim=event_embed_dim,
            hidden_dim=ripple_hidden_dim,
            num_heads=ripple_num_heads,
            num_gnn_layers=ripple_num_layers,
            dropout=dropout
        )
        
        # ============ Trade Weight Processing ============
        # This will be provided as input, but we can learn to refine it
        self.trade_weight_processor = nn.Sequential(
            nn.Linear(num_countries, num_countries),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_countries, num_countries),
            nn.Softmax(dim=-1)
        )
        
        # ============ Output Calibration ============
        # Final calibration layer to adjust predictions
        self.output_calibration = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1)
        )
        
        # ============ Uncertainty Quantification ============
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(d_model + local_hidden_channels[-1] + ripple_hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_countries),
            nn.Softplus()  # Ensure positive uncertainties
        )
    
    def forward(
        self,
        benchmark_features: torch.Tensor,
        local_features: torch.Tensor,
        country_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
        trade_adjacency: torch.Tensor,
        trade_weights: torch.Tensor,
        return_components: bool = False,
        return_uncertainty: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete GeoRipNet model.
        
        Args:
            benchmark_features: (batch, seq_len, benchmark_input_dim) - Global features
            local_features: (batch, num_countries, seq_len, local_input_dim) - Country features
            country_ids: (batch, num_countries) - Country ID matrix
            event_embeddings: (batch, num_countries, event_embed_dim) - News embeddings
            trade_adjacency: (batch, num_countries, num_countries) - Trade network
            trade_weights: (num_countries, num_benchmarks) - Trade weights for benchmarks
            return_components: If True, return intermediate predictions
            return_uncertainty: If True, return uncertainty estimates
            return_attention: If True, return attention weights
        
        Returns:
            Dictionary containing:
                - predictions: (batch, num_countries) - Final price predictions
                - benchmark_prices: (batch, num_benchmarks) - Optional
                - country_benchmarks: (batch, num_countries) - Optional
                - base_deltas: (batch, num_countries) - Optional
                - propagated_deltas: (batch, num_countries) - Optional
                - uncertainties: (batch, num_countries) - Optional
                - attention_weights: Dict of attention tensors - Optional
        """
        batch_size = benchmark_features.size(0)
        
        # ============ Step 1: Predict Global Benchmarks ============
        benchmark_output = self.benchmark_model(
            benchmark_features,
            return_attention=return_attention,
            return_quantiles=return_uncertainty
        )
        benchmark_prices = benchmark_output['predictions']  # (batch, num_benchmarks)
        
        # Compute trade-weighted country benchmarks: B_c(t+1)
        country_benchmarks = self.benchmark_model.compute_trade_weighted_benchmark(
            benchmark_prices, trade_weights
        )  # (batch, num_countries)
        
        # ============ Step 2: Predict Local Deltas ============
        # Process each country's features through local delta model
        if self.use_ensemble:
            # Ensemble prediction
            ensemble_deltas = []
            ensemble_uncertainties = []
            
            for model in self.local_delta_models:
                country_deltas = []
                country_uncertainties = []
                
                for c in range(self.num_countries):
                    # Get features for country c: (batch, seq_len, local_input_dim)
                    country_c_features = local_features[:, c, :, :]
                    country_c_ids = country_ids[:, c]
                    
                    # Predict delta
                    delta_output = model(
                        country_c_features,
                        country_c_ids,
                        return_uncertainty=return_uncertainty
                    )
                    country_deltas.append(delta_output['delta'])
                    
                    if return_uncertainty:
                        country_uncertainties.append(delta_output['uncertainty'])
                
                # Stack all countries: (batch, num_countries, 1)
                deltas_all = torch.cat(country_deltas, dim=1)
                ensemble_deltas.append(deltas_all)
                
                if return_uncertainty:
                    uncertainties_all = torch.cat(country_uncertainties, dim=1)
                    ensemble_uncertainties.append(uncertainties_all)
            
            # Average ensemble predictions
            base_deltas = torch.stack(ensemble_deltas, dim=0).mean(dim=0)  # (batch, num_countries)
            
            if return_uncertainty:
                # Combine uncertainties (variance of ensemble + mean uncertainty)
                ensemble_variance = torch.stack(ensemble_deltas, dim=0).var(dim=0)
                mean_uncertainty = torch.stack(ensemble_uncertainties, dim=0).mean(dim=0)
                base_uncertainties = ensemble_variance + mean_uncertainty
        else:
            # Single model prediction
            country_deltas = []
            country_uncertainties = []
            
            for c in range(self.num_countries):
                country_c_features = local_features[:, c, :, :]
                country_c_ids = country_ids[:, c]
                
                delta_output = self.local_delta_model(
                    country_c_features,
                    country_c_ids,
                    return_uncertainty=return_uncertainty
                )
                country_deltas.append(delta_output['delta'])
                
                if return_uncertainty:
                    country_uncertainties.append(delta_output['uncertainty'])
            
            base_deltas = torch.cat(country_deltas, dim=1)  # (batch, num_countries)
            
            if return_uncertainty:
                base_uncertainties = torch.cat(country_uncertainties, dim=1)
        
        # ============ Step 3: Ripple Propagation ============
        # Reshape for GNN: (batch, num_countries, 1)
        deltas_for_gnn = base_deltas.unsqueeze(-1)
        
        # Propagate through GNN: Δ̂_c_prop(t+1)
        propagated_deltas, attention_weights = self.ripple_gnn(
            deltas_for_gnn,
            event_embeddings,
            trade_adjacency,
            return_attention=True
        )
        propagated_deltas = propagated_deltas.squeeze(-1)  # (batch, num_countries)
        
        # ============ Step 4: Combine Components ============
        # P̂_c(t+1) = B̂_c(t+1) + Δ̂_c_prop(t+1)
        raw_predictions = country_benchmarks + propagated_deltas
        
        # Apply output calibration
        calibrated_predictions = self.output_calibration(
            raw_predictions.unsqueeze(-1)
        ).squeeze(-1)
        
        # ============ Prepare Output ============
        output = {
            'predictions': calibrated_predictions
        }
        
        if return_components:
            output['benchmark_prices'] = benchmark_prices
            output['country_benchmarks'] = country_benchmarks
            output['base_deltas'] = base_deltas
            output['propagated_deltas'] = propagated_deltas
            
            if 'quantiles' in benchmark_output:
                output['benchmark_quantiles'] = benchmark_output['quantiles']
        
        if return_uncertainty:
            output['uncertainties'] = base_uncertainties if self.use_ensemble else base_uncertainties
            
            # Add calibration uncertainty based on prediction variance
            if self.use_ensemble:
                output['epistemic_uncertainty'] = ensemble_variance
        
        if return_attention:
            output['attention_weights'] = {
                'benchmark_attention': benchmark_output.get('attention_weights', []),
                'ripple_attention': attention_weights
            }
        
        return output
    
    def predict_with_confidence(
        self,
        benchmark_features: torch.Tensor,
        local_features: torch.Tensor,
        country_ids: torch.Tensor,
        event_embeddings: torch.Tensor,
        trade_adjacency: torch.Tensor,
        trade_weights: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with confidence intervals using MC Dropout.
        
        Args:
            (same as forward)
            num_samples: Number of MC dropout samples
        
        Returns:
            mean_predictions: (batch, num_countries) - Mean predictions
            lower_bound: (batch, num_countries) - 10th percentile
            upper_bound: (batch, num_countries) - 90th percentile
        """
        self.train()  # Enable dropout
        
        predictions_samples = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                output = self.forward(
                    benchmark_features, local_features, country_ids,
                    event_embeddings, trade_adjacency, trade_weights,
                    return_components=False, return_uncertainty=False
                )
                predictions_samples.append(output['predictions'])
        
        # Stack: (num_samples, batch, num_countries)
        predictions_samples = torch.stack(predictions_samples, dim=0)
        
        # Compute statistics
        mean_predictions = predictions_samples.mean(dim=0)
        lower_bound = torch.quantile(predictions_samples, 0.1, dim=0)
        upper_bound = torch.quantile(predictions_samples, 0.9, dim=0)
        
        self.eval()  # Restore eval mode
        
        return mean_predictions, lower_bound, upper_bound


class EnsembleGeoRipNet(nn.Module):
    """
    Ensemble of multiple GeoRipNet models for improved robustness.
    
    Trains multiple models with different initializations and averages predictions.
    """
    
    def __init__(
        self,
        num_models: int = 5,
        **model_kwargs
    ):
        super().__init__()
        self.num_models = num_models
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            GeoRipNetModel(**model_kwargs)
            for _ in range(num_models)
        ])
    
    def forward(self, *args, **kwargs):
        """Average predictions from all models in ensemble."""
        outputs = []
        
        for model in self.models:
            output = model(*args, **kwargs)
            outputs.append(output['predictions'])
        
        # Average predictions
        ensemble_prediction = torch.stack(outputs, dim=0).mean(dim=0)
        
        # Compute ensemble uncertainty (variance across models)
        ensemble_variance = torch.stack(outputs, dim=0).var(dim=0)
        
        return {
            'predictions': ensemble_prediction,
            'ensemble_variance': ensemble_variance
        }


if __name__ == "__main__":
    # Test GeoRipNetModel
    print("Testing GeoRipNetModel...")
    
    batch_size = 4
    seq_len = 30
    num_countries = 20
    num_benchmarks = 3
    
    benchmark_input_dim = 50
    local_input_dim = 40
    event_embed_dim = 384
    
    model = GeoRipNetModel(
        benchmark_input_dim=benchmark_input_dim,
        num_benchmarks=num_benchmarks,
        local_input_dim=local_input_dim,
        num_countries=num_countries,
        event_embed_dim=event_embed_dim,
        d_model=256,
        ripple_hidden_dim=128,
        dropout=0.3,
        seq_len=seq_len,
        use_ensemble=False
    )
    
    # Create dummy inputs
    benchmark_features = torch.randn(batch_size, seq_len, benchmark_input_dim)
    local_features = torch.randn(batch_size, num_countries, seq_len, local_input_dim)
    country_ids = torch.arange(num_countries).unsqueeze(0).expand(batch_size, -1)
    event_embeddings = torch.randn(batch_size, num_countries, event_embed_dim)
    
    trade_adjacency = torch.rand(batch_size, num_countries, num_countries)
    trade_adjacency = (trade_adjacency + trade_adjacency.transpose(1, 2)) / 2
    eye = torch.eye(num_countries).unsqueeze(0).expand(batch_size, -1, -1)
    trade_adjacency = trade_adjacency + eye
    
    trade_weights = torch.rand(num_countries, num_benchmarks)
    
    # Forward pass
    output = model(
        benchmark_features, local_features, country_ids,
        event_embeddings, trade_adjacency, trade_weights,
        return_components=True,
        return_uncertainty=True,
        return_attention=True
    )
    
    print(f"\nPredictions shape: {output['predictions'].shape}")
    print(f"Benchmark prices shape: {output['benchmark_prices'].shape}")
    print(f"Country benchmarks shape: {output['country_benchmarks'].shape}")
    print(f"Base deltas shape: {output['base_deltas'].shape}")
    print(f"Propagated deltas shape: {output['propagated_deltas'].shape}")
    print(f"Uncertainties shape: {output['uncertainties'].shape}")
    
    # Test MC Dropout confidence intervals
    print("\nTesting MC Dropout predictions...")
    mean_pred, lower, upper = model.predict_with_confidence(
        benchmark_features, local_features, country_ids,
        event_embeddings, trade_adjacency, trade_weights,
        num_samples=10
    )
    
    print(f"Mean predictions shape: {mean_pred.shape}")
    print(f"Lower bound shape: {lower.shape}")
    print(f"Upper bound shape: {upper.shape}")
    
    print("\n✓ GeoRipNetModel test passed!")

