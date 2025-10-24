#!/usr/bin/env python3
"""
Simple Evaluation Script for RippleNet-TFT
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from data.dataset import create_dataset_from_merged_data
from model.ripple_tft import create_model
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model():
    """Simple evaluation of the trained model"""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    logger.info("Loading data...")
    data_module = create_dataset_from_merged_data('data/merged.csv', config)
    train_loader, val_loader, test_loader = data_module.get_data_loaders(num_workers=0)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model.eval()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            past_target = batch['past_target']
            observed_covariates = batch['observed_covariates']
            known_future = batch['known_future']
            target = batch['target']
            
            # Get predictions
            predictions = model(past_target, observed_covariates, known_future)
            
            test_predictions.append(predictions.cpu().numpy())
            test_targets.append(target.cpu().numpy())
    
    # Combine all predictions and targets
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    # Handle multi-target case - use first target for evaluation
    if test_targets.shape[1] > 1:
        test_targets = test_targets[:, 0:1]  # Take first target only
    
    # Ensure both arrays have the same shape
    if test_targets.shape != test_predictions.shape:
        # If targets have more columns, take the first one
        if test_targets.shape[1] > test_predictions.shape[1]:
            test_targets = test_targets[:, :test_predictions.shape[1]]
        # If predictions have more columns, repeat the first target
        elif test_predictions.shape[1] > test_targets.shape[1]:
            test_targets = np.repeat(test_targets, test_predictions.shape[1], axis=1)
    
    # Get scalers for inverse transformation
    scalers = data_module.get_scalers()
    target_scaler = scalers['target']
    
    # Debug shapes
    logger.info(f"Test predictions shape: {test_predictions.shape}")
    logger.info(f"Test targets shape: {test_targets.shape}")
    logger.info(f"Target scaler scale shape: {target_scaler.scale_.shape}")
    
    # Fix 3D targets - flatten to 2D
    if len(test_targets.shape) == 3:
        test_targets = test_targets.reshape(test_targets.shape[0], -1)
        logger.info(f"Reshaped targets to: {test_targets.shape}")
    
    # Handle multi-target case by reshaping if necessary
    if test_predictions.shape[1] != test_targets.shape[1]:
        if test_predictions.shape[1] == 1 and test_targets.shape[1] > 1:
            # Model predicts single target, repeat for all targets
            test_predictions = np.repeat(test_predictions, test_targets.shape[1], axis=1)
        elif test_predictions.shape[1] > test_targets.shape[1]:
            # Model predicts multiple targets, take first one
            test_predictions = test_predictions[:, :test_targets.shape[1]]
    
    # Ensure shapes match for inverse transform
    if test_predictions.shape[1] != target_scaler.scale_.shape[0]:
        # If scaler expects different number of features, adjust
        if test_predictions.shape[1] == 1 and target_scaler.scale_.shape[0] > 1:
            # Repeat single prediction for all targets
            test_predictions = np.repeat(test_predictions, target_scaler.scale_.shape[0], axis=1)
        elif test_predictions.shape[1] > target_scaler.scale_.shape[0]:
            # Take first N predictions
            test_predictions = test_predictions[:, :target_scaler.scale_.shape[0]]
    
    test_predictions_inv = target_scaler.inverse_transform(test_predictions)
    test_targets_inv = target_scaler.inverse_transform(test_targets)
    
    # Calculate metrics on inverse-transformed data
    mse = np.mean((test_predictions_inv - test_targets_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_predictions_inv - test_targets_inv))
    
    # Calculate R² on inverse-transformed data
    ss_res = np.sum((test_targets_inv - test_predictions_inv) ** 2)
    ss_tot = np.sum((test_targets_inv - np.mean(test_targets_inv)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate MAPE on inverse-transformed data
    mape = np.mean(np.abs((test_targets_inv - test_predictions_inv) / test_targets_inv)) * 100
    
    # Print results
    logger.info("=== EVALUATION RESULTS ===")
    logger.info(f"Test Samples: {len(test_predictions)}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    # Calculate directional accuracy on inverse-transformed data
    try:
        # Use only first target for directional accuracy
        pred_first = test_predictions_inv.flatten()
        target_first = test_targets_inv[:, 0] if test_targets_inv.shape[1] > 0 else test_targets_inv.flatten()
        
        # Ensure same length
        min_len = min(len(pred_first), len(target_first))
        pred_first = pred_first[:min_len]
        target_first = target_first[:min_len]
        
        # Calculate price changes for directional accuracy
        pred_changes = np.diff(pred_first)
        actual_changes = np.diff(target_first)
        
        # Calculate directional accuracy
        pred_direction = np.sign(pred_changes)
        actual_direction = np.sign(actual_changes)
        directional_accuracy = np.mean(pred_direction == actual_direction) * 100
        logger.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
    except Exception as e:
        logger.warning(f"Could not calculate directional accuracy: {e}")
        directional_accuracy = 0.0
    
    # Save results
    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'directional_accuracy': float(directional_accuracy),
        'test_samples': len(test_predictions)
    }
    
    with open('evaluation_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to evaluation_results.json")
    
    return results

if __name__ == "__main__":
    evaluate_model()
