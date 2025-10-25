"""
End-to-End Training Script for GeoRipNet Advanced.

This script demonstrates complete training pipeline:
1. Data loading and preprocessing
2. Model initialization
3. Training with advanced techniques
4. Evaluation and visualization
5. Model checkpointing

Run with:
    python train_geo_ripnet.py --config config.yaml
"""

import argparse
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models import GeoRipNetModel, EnsembleGeoRipNet
from losses.multi_objective_loss import MultiObjectiveLoss
from data.data_loader import create_dataloaders, TimeSeriesDataSplitter
from data.preprocessing import DataPreprocessor
from training.trainer import GeoRipNetTrainer
from utils.metrics import MetricsCalculator
from utils.visualization import GeoRipNetVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GeoRipNet Model')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory for outputs (checkpoints, plots, etc.)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--use_swa', action='store_true', default=True,
                        help='Use stochastic weight averaging')
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='Train ensemble of models')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'benchmark_input_dim': 50,
                'num_benchmarks': 3,
                'local_input_dim': 40,
                'event_embed_dim': 384,
                'd_model': 256,
                'local_hidden_channels': [128, 256, 256, 128],
                'ripple_hidden_dim': 128,
                'ripple_num_heads': 4,
                'ripple_num_layers': 2,
                'dropout': 0.3,
                'seq_len': 30
            },
            'training': {
                'learning_rate': 5e-5,
                'weight_decay': 1e-5,
                'gradient_clip_norm': 1.0,
                'swa_start_epoch': 10,
                'early_stopping_patience': 15,
                'scheduler_type': 'onecycle'
            },
            'loss': {
                'lambda_huber': 1.0,
                'lambda_directional': 0.3,
                'lambda_correlation': 0.2,
                'lambda_quantile': 0.0,
                'huber_delta': 1.0
            }
        }
    
    return config


def create_dummy_data(
    num_timesteps: int = 1000,
    num_countries: int = 20,
    num_benchmarks: int = 3,
    config: dict = None
):
    """
    Create dummy data for testing (replace with real data loader).
    
    In production, load real data from files.
    """
    print("\nGenerating dummy data for demonstration...")
    print("(Replace this with your real data loading logic)")
    
    benchmark_input_dim = config['model']['benchmark_input_dim']
    local_input_dim = config['model']['local_input_dim']
    event_embed_dim = config['model']['event_embed_dim']
    
    data_dict = {
        'benchmark_features': np.random.randn(num_timesteps, benchmark_input_dim),
        'local_features': np.random.randn(num_timesteps, num_countries, local_input_dim),
        'prices': np.random.randn(num_timesteps, num_countries) * 10 + 80,  # Around $80
        'event_embeddings': np.random.randn(num_timesteps, num_countries, event_embed_dim),
        'trade_adjacency': np.random.rand(num_countries, num_countries),
        'trade_weights': np.random.rand(num_countries, num_benchmarks)
    }
    
    # Make adjacency symmetric
    data_dict['trade_adjacency'] = (
        data_dict['trade_adjacency'] + data_dict['trade_adjacency'].T
    ) / 2
    
    # Add self-loops
    data_dict['trade_adjacency'] += np.eye(num_countries)
    
    # Normalize trade weights
    data_dict['trade_weights'] = (
        data_dict['trade_weights'] / 
        data_dict['trade_weights'].sum(axis=1, keepdims=True)
    )
    
    # Country names
    country_names = [f"Country_{i}" for i in range(num_countries)]
    
    return data_dict, country_names


def train_geo_ripnet():
    """Main training function."""
    
    # Parse arguments
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    plots_dir = output_dir / 'plots'
    
    # Load configuration
    config = load_config(args.config)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*80)
    print("GEORIPNET ADVANCED - TRAINING PIPELINE")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Random seed: {args.seed}")
    
    # ============ Step 1: Load and Prepare Data ============
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*80)
    
    # Load data (replace with real data loading)
    data_dict, country_names = create_dummy_data(
        num_timesteps=1000,
        num_countries=20,
        num_benchmarks=config['model']['num_benchmarks'],
        config=config
    )
    
    num_countries = len(country_names)
    
    # Split data
    train_dict, val_dict, test_dict = TimeSeriesDataSplitter.split_data(
        data_dict,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"\nData split:")
    print(f"  Training samples: {len(train_dict['prices'])}")
    print(f"  Validation samples: {len(val_dict['prices'])}")
    print(f"  Test samples: {len(test_dict['prices'])}")
    
    # Preprocessing
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor(
        num_countries=num_countries,
        num_benchmarks=config['model']['num_benchmarks'],
        scaler_type='standard'
    )
    
    train_dict = preprocessor.fit_transform(train_dict)
    val_dict = preprocessor.transform(val_dict)
    test_dict = preprocessor.transform(test_dict)
    
    # Save preprocessor
    preprocessor.save(output_dir / 'preprocessor')
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dict, val_dict, test_dict,
        seq_len=config['model']['seq_len'],
        pred_horizon=1,
        batch_size=args.batch_size,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # ============ Step 2: Initialize Model ============
    print("\n" + "="*80)
    print("STEP 2: MODEL INITIALIZATION")
    print("="*80)
    
    if args.ensemble:
        print("\nInitializing Ensemble GeoRipNet (3 models)...")
        model = EnsembleGeoRipNet(
            num_models=3,
            benchmark_input_dim=config['model']['benchmark_input_dim'],
            num_benchmarks=config['model']['num_benchmarks'],
            local_input_dim=config['model']['local_input_dim'],
            num_countries=num_countries,
            event_embed_dim=config['model']['event_embed_dim'],
            d_model=config['model']['d_model'],
            local_hidden_channels=config['model']['local_hidden_channels'],
            ripple_hidden_dim=config['model']['ripple_hidden_dim'],
            ripple_num_heads=config['model']['ripple_num_heads'],
            ripple_num_layers=config['model']['ripple_num_layers'],
            dropout=config['model']['dropout'],
            seq_len=config['model']['seq_len']
        )
    else:
        print("\nInitializing GeoRipNet model...")
        model = GeoRipNetModel(
            benchmark_input_dim=config['model']['benchmark_input_dim'],
            num_benchmarks=config['model']['num_benchmarks'],
            local_input_dim=config['model']['local_input_dim'],
            num_countries=num_countries,
            event_embed_dim=config['model']['event_embed_dim'],
            d_model=config['model']['d_model'],
            local_hidden_channels=config['model']['local_hidden_channels'],
            ripple_hidden_dim=config['model']['ripple_hidden_dim'],
            ripple_num_heads=config['model']['ripple_num_heads'],
            ripple_num_layers=config['model']['ripple_num_layers'],
            dropout=config['model']['dropout'],
            seq_len=config['model']['seq_len'],
            use_ensemble=False
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ============ Step 3: Initialize Training Components ============
    print("\n" + "="*80)
    print("STEP 3: TRAINING SETUP")
    print("="*80)
    
    # Loss function
    criterion = MultiObjectiveLoss(
        lambda_huber=config['loss']['lambda_huber'],
        lambda_directional=config['loss']['lambda_directional'],
        lambda_correlation=config['loss']['lambda_correlation'],
        lambda_quantile=config['loss']['lambda_quantile'],
        huber_delta=config['loss']['huber_delta']
    )
    
    print("\nLoss function: Multi-Objective Loss")
    print(f"  λ_huber: {config['loss']['lambda_huber']}")
    print(f"  λ_directional: {config['loss']['lambda_directional']}")
    print(f"  λ_correlation: {config['loss']['lambda_correlation']}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training']['weight_decay']}")
    
    # Trainer
    trainer = GeoRipNetTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        use_amp=args.use_amp,
        use_swa=args.use_swa,
        gradient_clip_norm=config['training']['gradient_clip_norm'],
        swa_start_epoch=config['training']['swa_start_epoch'],
        checkpoint_dir=checkpoint_dir
    )
    
    # ============ Step 4: Training ============
    print("\n" + "="*80)
    print("STEP 4: TRAINING")
    print("="*80)
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        scheduler_type=config['training']['scheduler_type'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        save_best_only=True,
        verbose=True
    )
    
    # ============ Step 5: Evaluation ============
    print("\n" + "="*80)
    print("STEP 5: EVALUATION")
    print("="*80)
    
    # Load best model
    print("\nLoading best model for evaluation...")
    best_checkpoint = checkpoint_dir / 'checkpoint_best.pth'
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_prev_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            outputs = model(
                batch['benchmark_features'],
                batch['local_features'],
                batch['country_ids'],
                batch['event_embeddings'],
                batch['trade_adjacency'],
                batch['trade_weights']
            )
            
            all_predictions.append(outputs['predictions'].cpu().numpy())
            all_targets.append(batch['targets'].cpu().numpy())
            all_prev_targets.append(batch['prev_targets'].cpu().numpy())
    
    # Concatenate results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    prev_targets = np.concatenate(all_prev_targets, axis=0)
    
    # Inverse transform to original scale
    predictions = preprocessor.inverse_transform_prices(predictions)
    targets = preprocessor.inverse_transform_prices(targets)
    prev_targets = preprocessor.inverse_transform_prices(prev_targets)
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics_calculator = MetricsCalculator(country_names)
    
    predicted_deltas = predictions - prev_targets
    true_deltas = targets - prev_targets
    
    metrics = metrics_calculator.compute_all_metrics(
        predictions, targets, prev_targets,
        predicted_deltas, true_deltas
    )
    
    # Print summary
    metrics_calculator.print_summary(metrics)
    
    # Save metrics
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed report
    report = metrics_calculator.create_report(metrics)
    report.to_csv(output_dir / 'test_metrics.csv')
    
    # ============ Step 6: Visualization ============
    print("\n" + "="*80)
    print("STEP 6: VISUALIZATION")
    print("="*80)
    
    visualizer = GeoRipNetVisualizer(save_dir=plots_dir)
    
    visualizer.create_comprehensive_report(
        predictions=predictions,
        targets=targets,
        prev_targets=prev_targets,
        history=trainer.history,
        country_names=country_names
    )
    
    # ============ Completion ============
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Plots: {plots_dir}")
    print(f"  - Metrics: {output_dir / 'test_metrics.json'}")
    print(f"  - Config: {output_dir / 'config.json'}")
    
    print("\n" + "="*80)
    
    return model, metrics, trainer.history


if __name__ == "__main__":
    train_geo_ripnet()

