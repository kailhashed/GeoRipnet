"""
Example Usage: GeoRipNet Advanced Framework

This script demonstrates how to use various components of GeoRipNet.
"""

import torch
import numpy as np
from pathlib import Path

# Import GeoRipNet components
from models import (
    BenchmarkModel,
    LocalDeltaModel,
    RippleGNNLayer,
    GeoRipNetModel,
    EnsembleGeoRipNet
)
from losses.multi_objective_loss import MultiObjectiveLoss
from data.data_loader import GeoRipNetDataset, create_dataloaders
from data.preprocessing import DataPreprocessor, FeatureEngineer
from training.trainer import GeoRipNetTrainer
from utils.metrics import MetricsCalculator
from utils.visualization import GeoRipNetVisualizer


def example_1_benchmark_model():
    """Example 1: Using BenchmarkModel standalone."""
    print("\n" + "="*80)
    print("EXAMPLE 1: BenchmarkModel - Global Benchmark Prediction")
    print("="*80)
    
    # Model configuration
    batch_size = 16
    seq_len = 30
    input_dim = 50
    
    # Initialize model
    model = BenchmarkModel(
        input_dim=input_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.3,
        num_benchmarks=3
    )
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x, return_quantiles=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Sample prediction (WTI, Brent, Oman): {output['predictions'][0].detach().cpu().numpy()}")
    print(f"Quantiles available: {output['quantiles'].shape}")
    
    # Compute trade-weighted benchmarks
    num_countries = 20
    trade_weights = torch.rand(num_countries, 3)
    trade_weights = trade_weights / trade_weights.sum(dim=1, keepdim=True)
    
    country_benchmarks = model.compute_trade_weighted_benchmark(
        output['predictions'], trade_weights
    )
    print(f"Country-specific benchmarks shape: {country_benchmarks.shape}")


def example_2_local_delta_model():
    """Example 2: Using LocalDeltaModel for country-specific deviations."""
    print("\n" + "="*80)
    print("EXAMPLE 2: LocalDeltaModel - Country-Specific Deviations")
    print("="*80)
    
    batch_size = 16
    seq_len = 30
    input_dim = 40
    num_countries = 20
    
    # Initialize model
    model = LocalDeltaModel(
        input_dim=input_dim,
        num_countries=num_countries,
        country_embedding_dim=64,
        hidden_channels=[128, 256, 256, 128],
        dropout=0.3
    )
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, input_dim)
    country_ids = torch.randint(0, num_countries, (batch_size,))
    
    # Forward pass
    output = model(x, country_ids, return_uncertainty=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Country IDs: {country_ids[:5].tolist()}")
    print(f"Delta predictions shape: {output['delta'].shape}")
    print(f"Sample delta: {output['delta'][0].item():.4f}")
    print(f"Uncertainty shape: {output['uncertainty'].shape}")
    print(f"Sample uncertainty: {output['uncertainty'][0].item():.4f}")


def example_3_ripple_gnn():
    """Example 3: Using RippleGNNLayer for shock propagation."""
    print("\n" + "="*80)
    print("EXAMPLE 3: RippleGNNLayer - Shock Propagation")
    print("="*80)
    
    batch_size = 8
    num_countries = 20
    event_embed_dim = 384
    
    # Initialize model
    model = RippleGNNLayer(
        num_countries=num_countries,
        delta_dim=1,
        event_embed_dim=event_embed_dim,
        hidden_dim=128,
        num_heads=4,
        num_gnn_layers=2
    )
    
    # Dummy inputs
    deltas = torch.randn(batch_size, num_countries, 1)
    event_embeddings = torch.randn(batch_size, num_countries, event_embed_dim)
    trade_adjacency = torch.rand(batch_size, num_countries, num_countries)
    
    # Make adjacency symmetric
    trade_adjacency = (trade_adjacency + trade_adjacency.transpose(1, 2)) / 2
    
    # Forward pass
    propagated_deltas, attention = model(
        deltas, event_embeddings, trade_adjacency, return_attention=True
    )
    
    print(f"Input deltas shape: {deltas.shape}")
    print(f"Propagated deltas shape: {propagated_deltas.shape}")
    print(f"Sample input delta: {deltas[0, 0, 0].item():.4f}")
    print(f"Sample propagated delta: {propagated_deltas[0, 0, 0].item():.4f}")
    print(f"Number of attention weight tensors: {len(attention)}")
    
    # Compute influence matrix
    influence = model.compute_ripple_influence(deltas, trade_adjacency)
    print(f"Influence matrix shape: {influence.shape}")


def example_4_complete_model():
    """Example 4: Using complete GeoRipNetModel."""
    print("\n" + "="*80)
    print("EXAMPLE 4: GeoRipNetModel - Complete Pipeline")
    print("="*80)
    
    batch_size = 4
    seq_len = 30
    num_countries = 20
    num_benchmarks = 3
    
    # Initialize model
    model = GeoRipNetModel(
        benchmark_input_dim=50,
        num_benchmarks=num_benchmarks,
        local_input_dim=40,
        num_countries=num_countries,
        event_embed_dim=384,
        d_model=256,
        dropout=0.3,
        seq_len=seq_len
    )
    
    # Dummy inputs
    benchmark_features = torch.randn(batch_size, seq_len, 50)
    local_features = torch.randn(batch_size, num_countries, seq_len, 40)
    country_ids = torch.arange(num_countries).unsqueeze(0).expand(batch_size, -1)
    event_embeddings = torch.randn(batch_size, num_countries, 384)
    trade_adjacency = torch.rand(batch_size, num_countries, num_countries)
    trade_weights = torch.rand(num_countries, num_benchmarks)
    
    # Forward pass
    output = model(
        benchmark_features, local_features, country_ids,
        event_embeddings, trade_adjacency, trade_weights,
        return_components=True, return_uncertainty=True
    )
    
    print(f"\nComplete Model Output:")
    print(f"  Final predictions shape: {output['predictions'].shape}")
    print(f"  Benchmark prices shape: {output['benchmark_prices'].shape}")
    print(f"  Country benchmarks shape: {output['country_benchmarks'].shape}")
    print(f"  Base deltas shape: {output['base_deltas'].shape}")
    print(f"  Propagated deltas shape: {output['propagated_deltas'].shape}")
    print(f"  Uncertainties shape: {output['uncertainties'].shape}")
    
    print(f"\nSample prediction for Country 0: ${output['predictions'][0, 0].item():.2f}")


def example_5_training():
    """Example 5: Training with GeoRipNetTrainer."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Training Pipeline")
    print("="*80)
    
    # Setup (dummy data for demonstration)
    num_countries = 10
    model = GeoRipNetModel(
        benchmark_input_dim=50,
        num_benchmarks=3,
        local_input_dim=40,
        num_countries=num_countries,
        event_embed_dim=384
    )
    
    # Loss function
    criterion = MultiObjectiveLoss(
        lambda_huber=1.0,
        lambda_directional=0.3,
        lambda_correlation=0.2
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-5
    )
    
    # Trainer
    trainer = GeoRipNetTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device='cpu',  # Use CPU for demo
        use_amp=False,  # Disable AMP for CPU
        use_swa=False
    )
    
    print("Trainer initialized successfully!")
    print(f"  Device: cpu")
    print(f"  Mixed Precision: False")
    print(f"  SWA: False")
    
    # Note: In real usage, call trainer.fit(train_loader, val_loader, num_epochs=100)


def example_6_metrics():
    """Example 6: Computing evaluation metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Evaluation Metrics")
    print("="*80)
    
    num_samples = 100
    num_countries = 10
    
    # Generate dummy predictions and targets
    np.random.seed(42)
    targets = np.random.randn(num_samples, num_countries) * 10 + 80
    predictions = targets + np.random.randn(num_samples, num_countries) * 2
    prev_targets = targets - np.random.randn(num_samples, num_countries) * 3
    
    predicted_deltas = predictions - prev_targets
    true_deltas = targets - prev_targets
    
    # Create metrics calculator
    country_names = [f"Country_{i}" for i in range(num_countries)]
    calculator = MetricsCalculator(country_names)
    
    # Compute all metrics
    metrics = calculator.compute_all_metrics(
        predictions, targets, prev_targets,
        predicted_deltas, true_deltas
    )
    
    # Print summary
    calculator.print_summary(metrics)


def example_7_visualization():
    """Example 7: Creating visualizations."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Visualization")
    print("="*80)
    
    # Generate dummy data
    num_samples = 100
    num_countries = 10
    
    np.random.seed(42)
    targets = np.random.randn(num_samples, num_countries) * 10 + 80
    predictions = targets + np.random.randn(num_samples, num_countries) * 2
    
    country_names = [f"Country_{i}" for i in range(num_countries)]
    
    # Create visualizer
    visualizer = GeoRipNetVisualizer(save_dir='demo_plots')
    
    # Training history plot
    history = {
        'train_loss': np.random.exponential(2, 50).tolist(),
        'val_loss': np.random.exponential(2.2, 50).tolist(),
        'learning_rates': (np.logspace(-3, -5, 50)).tolist()
    }
    
    print("\nGenerating plots...")
    visualizer.plot_training_history(history, 'demo_training_history.png')
    visualizer.plot_predictions_vs_actual(predictions, targets, country_names, 'demo_predictions.png')
    
    print(f"Plots saved to demo_plots/")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("GEORIPNET ADVANCED - USAGE EXAMPLES")
    print("="*80)
    
    example_1_benchmark_model()
    example_2_local_delta_model()
    example_3_ripple_gnn()
    example_4_complete_model()
    example_5_training()
    example_6_metrics()
    example_7_visualization()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Prepare your real data in the required format")
    print("  2. Update config.yaml with your settings")
    print("  3. Run: python train_geo_ripnet.py --config config.yaml")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

