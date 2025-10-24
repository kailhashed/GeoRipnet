#!/usr/bin/env python3
"""
Fixed Evaluation Script for RippleNet-TFT
Addresses scaling, target alignment, and ARIMA integration issues
"""

import torch
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
import yaml
from data.dataset import create_dataset_from_merged_data
from model.ripple_tft import create_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_data():
    """Load model and data with proper scaling"""
    logger.info("Loading model and data...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    data_module = create_dataset_from_merged_data('data/merged.csv', config)
    train_loader, val_loader, test_loader = data_module.get_data_loaders(num_workers=0)
    
    # Create model
    model = create_model(config)
    model.eval()
    
    # Get scalers
    scalers = data_module.get_scalers()
    
    return model, test_loader, scalers, data_module

def inverse_transform_predictions(predictions, targets, scalers):
    """Properly inverse transform predictions and targets"""
    logger.info("Applying inverse transformations...")
    
    # Get target scaler
    target_scaler = scalers.get('target')
    if target_scaler is None:
        logger.warning("No target scaler found, using raw values")
        return predictions, targets
    
    # Reshape for inverse transform
    pred_shape = predictions.shape
    target_shape = targets.shape
    
    # Flatten for inverse transform
    pred_flat = predictions.reshape(-1, 1)
    target_flat = targets.reshape(-1, 1)
    
    # Inverse transform
    pred_inverse = target_scaler.inverse_transform(pred_flat)
    target_inverse = target_scaler.inverse_transform(target_flat)
    
    # Reshape back to original shape
    pred_inverse = pred_inverse.reshape(pred_shape)
    target_inverse = target_inverse.reshape(target_shape)
    
    logger.info(f"Predictions shape after inverse: {pred_inverse.shape}")
    logger.info(f"Targets shape after inverse: {target_inverse.shape}")
    
    return pred_inverse, target_inverse

def evaluate_model_fixed():
    """Fixed evaluation with proper scaling and metrics"""
    logger.info("Starting fixed evaluation...")
    
    # Load model and data
    model, test_loader, scalers, data_module = load_model_and_data()
    
    # Load best model
    checkpoint_path = 'checkpoints/best_model.pt'
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded best model checkpoint")
    else:
        logger.warning("No checkpoint found, using untrained model")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions = []
    test_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            past_target = batch['past_target']
            observed_covariates = batch['observed_covariates']
            known_future = batch['known_future']
            target = batch['target']
            
            predictions = model(past_target, observed_covariates, known_future)
            
            test_predictions.append(predictions.cpu().numpy())
            test_targets.append(target.cpu().numpy())
    
    # Combine all predictions and targets
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    logger.info(f"Raw predictions shape: {test_predictions.shape}")
    logger.info(f"Raw targets shape: {test_targets.shape}")
    logger.info(f"Raw predictions range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
    logger.info(f"Raw targets range: [{test_targets.min():.4f}, {test_targets.max():.4f}]")
    
    # Handle multi-target case - use first target for evaluation
    if test_targets.shape[1] > 1:
        test_targets = test_targets[:, 0:1]  # Take first target only
    
    # Ensure both arrays have the same shape
    if test_targets.shape != test_predictions.shape:
        if test_targets.shape[1] > test_predictions.shape[1]:
            test_targets = test_targets[:, :test_predictions.shape[1]]
        elif test_predictions.shape[1] > test_targets.shape[1]:
            test_targets = np.repeat(test_targets, predictions.shape[1], axis=1)
    
    # CRITICAL FIX: Apply inverse transformations
    test_predictions_inv, test_targets_inv = inverse_transform_predictions(
        test_predictions, test_targets, scalers
    )
    
    # Flatten for metrics calculation
    pred_flat = test_predictions_inv.flatten()
    target_flat = test_targets_inv.flatten()
    
    logger.info(f"Inverse transformed predictions range: [{pred_flat.min():.4f}, {pred_flat.max():.4f}]")
    logger.info(f"Inverse transformed targets range: [{target_flat.min():.4f}, {target_flat.max():.4f}]")
    
    # Calculate metrics on inverse-transformed data
    mse = mean_squared_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_flat, pred_flat)
    r2 = r2_score(target_flat, pred_flat)
    
    # Calculate MAPE (avoid division by zero)
    mape = np.mean(np.abs((target_flat - pred_flat) / (np.abs(target_flat) + 1e-8))) * 100
    
    # Calculate directional accuracy
    try:
        # Use price changes for directional accuracy
        pred_changes = np.diff(pred_flat)
        target_changes = np.diff(target_flat)
        
        # Calculate directional accuracy
        directional_accuracy = np.mean(np.sign(pred_changes) == np.sign(target_changes)) * 100
    except Exception as e:
        logger.warning(f"Could not calculate directional accuracy: {e}")
        directional_accuracy = 0.0
    
    # Print results
    logger.info("=== FIXED EVALUATION RESULTS ===")
    logger.info(f"Test Samples: {len(pred_flat)}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")
    logger.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
    
    # Additional diagnostic metrics
    logger.info("\n=== DIAGNOSTIC METRICS ===")
    logger.info(f"Prediction std: {np.std(pred_flat):.4f}")
    logger.info(f"Target std: {np.std(target_flat):.4f}")
    logger.info(f"Prediction mean: {np.mean(pred_flat):.4f}")
    logger.info(f"Target mean: {np.mean(target_flat):.4f}")
    
    # Check for constant predictions
    if np.std(pred_flat) < 1e-6:
        logger.warning("⚠️  Model is outputting constant predictions!")
    
    # Check for scale mismatch
    scale_ratio = np.std(target_flat) / (np.std(pred_flat) + 1e-8)
    if scale_ratio > 10 or scale_ratio < 0.1:
        logger.warning(f"⚠️  Scale mismatch detected! Ratio: {scale_ratio:.2f}")
    
    # Save results
    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'directional_accuracy': float(directional_accuracy),
        'test_samples': len(pred_flat),
        'prediction_std': float(np.std(pred_flat)),
        'target_std': float(np.std(target_flat)),
        'scale_ratio': float(scale_ratio)
    }
    
    with open('fixed_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to fixed_evaluation_results.json")
    
    # Create diagnostic plots
    create_diagnostic_plots(pred_flat, target_flat)
    
    return results

def create_diagnostic_plots(predictions, targets):
    """Create diagnostic plots for model evaluation"""
    logger.info("Creating diagnostic plots...")
    
    # Create plots directory
    Path('diagnostic_plots').mkdir(exist_ok=True)
    
    # Plot 1: Predictions vs Actual
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Fixed Evaluation: Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('diagnostic_plots/predictions_vs_actual_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Time series comparison
    plt.figure(figsize=(15, 6))
    time_index = range(len(predictions))
    plt.plot(time_index, targets, 'b-', label='Actual', linewidth=2, alpha=0.8)
    plt.plot(time_index, predictions, 'r-', label='Predicted', linewidth=2, alpha=0.8)
    plt.xlabel('Time Steps')
    plt.ylabel('Price Values')
    plt.title('Fixed Evaluation: Time Series Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('diagnostic_plots/time_series_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Error distribution
    errors = targets - predictions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Errors')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diagnostic_plots/error_analysis_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Diagnostic plots saved to diagnostic_plots/")

if __name__ == "__main__":
    evaluate_model_fixed()
