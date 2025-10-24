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
    
    # Calculate metrics
    mse = np.mean((test_predictions - test_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_predictions - test_targets))
    
    # Calculate R²
    ss_res = np.sum((test_targets - test_predictions) ** 2)
    ss_tot = np.sum((test_targets - np.mean(test_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate MAPE
    mape = np.mean(np.abs((test_targets - test_predictions) / test_targets)) * 100
    
    # Print results
    logger.info("=== EVALUATION RESULTS ===")
    logger.info(f"Test Samples: {len(test_predictions)}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    # Calculate directional accuracy (simplified)
    try:
        # Use only first target for directional accuracy
        pred_first = test_predictions.flatten()
        target_first = test_targets[:, 0] if test_targets.shape[1] > 0 else test_targets.flatten()
        
        # Ensure same length
        min_len = min(len(pred_first), len(target_first))
        pred_first = pred_first[:min_len]
        target_first = target_first[:min_len]
        
        pred_direction = np.sign(pred_first - pred_first.mean())
        actual_direction = np.sign(target_first - target_first.mean())
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
