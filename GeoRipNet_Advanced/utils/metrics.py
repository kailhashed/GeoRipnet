"""
Evaluation Metrics for GeoRipNet.

Implements comprehensive evaluation metrics:
- RMSE, MAE, R² (per country and aggregated)
- Directional Accuracy
- Ripple Correlation Score
- MAPE (Mean Absolute Percentage Error)
- Coverage metrics for uncertainty quantification
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd


def compute_rmse(predictions: np.ndarray, targets: np.ndarray, per_country: bool = False) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        predictions: (num_samples, num_countries) or (num_samples,)
        targets: Same shape as predictions
        per_country: If True, return RMSE for each country
    
    Returns:
        RMSE value(s)
    """
    if per_country and predictions.ndim == 2:
        rmse = np.sqrt(((predictions - targets) ** 2).mean(axis=0))
    else:
        rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    return rmse


def compute_mae(predictions: np.ndarray, targets: np.ndarray, per_country: bool = False) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: (num_samples, num_countries) or (num_samples,)
        targets: Same shape as predictions
        per_country: If True, return MAE for each country
    
    Returns:
        MAE value(s)
    """
    if per_country and predictions.ndim == 2:
        mae = np.abs(predictions - targets).mean(axis=0)
    else:
        mae = mean_absolute_error(targets, predictions)
    
    return mae


def compute_r2(predictions: np.ndarray, targets: np.ndarray, per_country: bool = False) -> float:
    """
    Compute R² (coefficient of determination).
    
    Args:
        predictions: (num_samples, num_countries) or (num_samples,)
        targets: Same shape as predictions
        per_country: If True, return R² for each country
    
    Returns:
        R² value(s)
    """
    if per_country and predictions.ndim == 2:
        r2 = np.array([r2_score(targets[:, i], predictions[:, i]) 
                       for i in range(predictions.shape[1])])
    else:
        r2 = r2_score(targets, predictions)
    
    return r2


def compute_mape(predictions: np.ndarray, targets: np.ndarray, per_country: bool = False) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    Args:
        predictions: (num_samples, num_countries) or (num_samples,)
        targets: Same shape as predictions
        per_country: If True, return MAPE for each country
    
    Returns:
        MAPE value(s) in percentage
    """
    epsilon = 1e-8  # Avoid division by zero
    
    if per_country and predictions.ndim == 2:
        mape = (np.abs((targets - predictions) / (targets + epsilon)).mean(axis=0)) * 100
    else:
        mape = (np.abs((targets - predictions) / (targets + epsilon)).mean()) * 100
    
    return mape


def compute_directional_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    prev_targets: np.ndarray,
    per_country: bool = False
) -> float:
    """
    Compute directional accuracy (percentage of correct sign predictions).
    
    Args:
        predictions: (num_samples, num_countries) - Current predictions
        targets: (num_samples, num_countries) - Actual current values
        prev_targets: (num_samples, num_countries) - Previous time step values
        per_country: If True, return accuracy for each country
    
    Returns:
        Directional accuracy in [0, 1]
    """
    pred_change = predictions - prev_targets
    true_change = targets - prev_targets
    
    # Sign agreement
    correct = (np.sign(pred_change) == np.sign(true_change))
    
    if per_country and predictions.ndim == 2:
        accuracy = correct.mean(axis=0)
    else:
        accuracy = correct.mean()
    
    return accuracy


def compute_ripple_correlation(
    predicted_deltas: np.ndarray,
    true_deltas: np.ndarray
) -> float:
    """
    Compute ripple correlation score.
    
    Measures how well predicted deltas preserve correlation structure
    across countries compared to true deltas.
    
    Args:
        predicted_deltas: (num_samples, num_countries) - Predicted deltas
        true_deltas: (num_samples, num_countries) - True deltas
    
    Returns:
        Correlation score in [-1, 1] (higher is better)
    """
    # Compute correlation matrices
    pred_corr = np.corrcoef(predicted_deltas.T)
    true_corr = np.corrcoef(true_deltas.T)
    
    # Extract upper triangular (exclude diagonal)
    num_countries = predicted_deltas.shape[1]
    mask = np.triu(np.ones((num_countries, num_countries)), k=1).astype(bool)
    
    pred_corr_flat = pred_corr[mask]
    true_corr_flat = true_corr[mask]
    
    # Correlation between correlation vectors
    if len(pred_corr_flat) > 1:
        correlation = np.corrcoef(pred_corr_flat, true_corr_flat)[0, 1]
    else:
        correlation = 0.0
    
    return correlation


def compute_coverage(
    targets: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    expected_coverage: float = 0.8
) -> Dict[str, float]:
    """
    Compute coverage metrics for uncertainty quantification.
    
    Args:
        targets: (num_samples, num_countries) - True values
        lower_bounds: (num_samples, num_countries) - Lower prediction bounds
        upper_bounds: (num_samples, num_countries) - Upper prediction bounds
        expected_coverage: Expected coverage level (default: 0.8 for 80%)
    
    Returns:
        Dictionary with coverage metrics
    """
    # Check if targets are within bounds
    within_bounds = (targets >= lower_bounds) & (targets <= upper_bounds)
    
    # Overall coverage
    coverage = within_bounds.mean()
    
    # Per-country coverage
    per_country_coverage = within_bounds.mean(axis=0)
    
    # Average interval width
    interval_width = (upper_bounds - lower_bounds).mean()
    
    # Coverage calibration error
    calibration_error = abs(coverage - expected_coverage)
    
    return {
        'coverage': coverage,
        'per_country_coverage': per_country_coverage,
        'interval_width': interval_width,
        'calibration_error': calibration_error
    }


class MetricsCalculator:
    """
    Comprehensive metrics calculator for GeoRipNet evaluation.
    
    Computes all relevant metrics and organizes them into a report.
    """
    
    def __init__(self, country_names: Optional[List[str]] = None):
        """
        Args:
            country_names: List of country names for per-country reporting
        """
        self.country_names = country_names
    
    def compute_all_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        prev_targets: np.ndarray,
        predicted_deltas: Optional[np.ndarray] = None,
        true_deltas: Optional[np.ndarray] = None,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: (num_samples, num_countries)
            targets: (num_samples, num_countries)
            prev_targets: (num_samples, num_countries)
            predicted_deltas: Optional (num_samples, num_countries)
            true_deltas: Optional (num_samples, num_countries)
            lower_bounds: Optional (num_samples, num_countries)
            upper_bounds: Optional (num_samples, num_countries)
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Aggregate metrics
        metrics['aggregate'] = {
            'rmse': float(compute_rmse(predictions, targets)),
            'mae': float(compute_mae(predictions, targets)),
            'r2': float(compute_r2(predictions, targets)),
            'mape': float(compute_mape(predictions, targets)),
            'directional_accuracy': float(compute_directional_accuracy(
                predictions, targets, prev_targets
            ))
        }
        
        # Per-country metrics
        metrics['per_country'] = {
            'rmse': compute_rmse(predictions, targets, per_country=True).tolist(),
            'mae': compute_mae(predictions, targets, per_country=True).tolist(),
            'r2': compute_r2(predictions, targets, per_country=True).tolist(),
            'mape': compute_mape(predictions, targets, per_country=True).tolist(),
            'directional_accuracy': compute_directional_accuracy(
                predictions, targets, prev_targets, per_country=True
            ).tolist()
        }
        
        # Ripple correlation (if deltas provided)
        if predicted_deltas is not None and true_deltas is not None:
            metrics['aggregate']['ripple_correlation'] = float(
                compute_ripple_correlation(predicted_deltas, true_deltas)
            )
        
        # Coverage metrics (if bounds provided)
        if lower_bounds is not None and upper_bounds is not None:
            coverage_metrics = compute_coverage(targets, lower_bounds, upper_bounds)
            metrics['uncertainty'] = {
                'coverage': float(coverage_metrics['coverage']),
                'interval_width': float(coverage_metrics['interval_width']),
                'calibration_error': float(coverage_metrics['calibration_error'])
            }
        
        return metrics
    
    def create_report(self, metrics: Dict) -> pd.DataFrame:
        """
        Create a pandas DataFrame report from metrics.
        
        Args:
            metrics: Dictionary from compute_all_metrics
        
        Returns:
            DataFrame with organized metrics
        """
        # Aggregate metrics
        agg_df = pd.DataFrame([metrics['aggregate']], index=['Aggregate'])
        
        # Per-country metrics
        if self.country_names is not None:
            index = self.country_names
        else:
            num_countries = len(metrics['per_country']['rmse'])
            index = [f'Country_{i}' for i in range(num_countries)]
        
        per_country_df = pd.DataFrame(metrics['per_country'], index=index)
        
        # Combine
        report = pd.concat([agg_df, per_country_df])
        
        return report
    
    def print_summary(self, metrics: Dict):
        """Print a summary of key metrics."""
        print("\n" + "="*60)
        print("GEORIPNET EVALUATION SUMMARY")
        print("="*60)
        
        print("\nAggregate Metrics:")
        print(f"  RMSE:                  {metrics['aggregate']['rmse']:.4f}")
        print(f"  MAE:                   {metrics['aggregate']['mae']:.4f}")
        print(f"  R²:                    {metrics['aggregate']['r2']:.4f}")
        print(f"  MAPE:                  {metrics['aggregate']['mape']:.2f}%")
        print(f"  Directional Accuracy:  {metrics['aggregate']['directional_accuracy']:.2%}")
        
        if 'ripple_correlation' in metrics['aggregate']:
            print(f"  Ripple Correlation:    {metrics['aggregate']['ripple_correlation']:.4f}")
        
        if 'uncertainty' in metrics:
            print("\nUncertainty Quantification:")
            print(f"  Coverage:              {metrics['uncertainty']['coverage']:.2%}")
            print(f"  Interval Width:        {metrics['uncertainty']['interval_width']:.4f}")
            print(f"  Calibration Error:     {metrics['uncertainty']['calibration_error']:.4f}")
        
        # Top and bottom countries by R²
        r2_scores = np.array(metrics['per_country']['r2'])
        top_indices = np.argsort(r2_scores)[-5:][::-1]
        bottom_indices = np.argsort(r2_scores)[:5]
        
        print("\nTop 5 Countries (by R²):")
        for i in top_indices:
            country_name = self.country_names[i] if self.country_names else f"Country_{i}"
            print(f"  {country_name:20s} R²={r2_scores[i]:.4f}")
        
        print("\nBottom 5 Countries (by R²):")
        for i in bottom_indices:
            country_name = self.country_names[i] if self.country_names else f"Country_{i}"
            print(f"  {country_name:20s} R²={r2_scores[i]:.4f}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # Test metrics
    print("Testing GeoRipNet Metrics...")
    
    num_samples = 100
    num_countries = 10
    
    # Generate dummy data
    np.random.seed(42)
    targets = np.random.randn(num_samples, num_countries) * 10 + 80
    predictions = targets + np.random.randn(num_samples, num_countries) * 2  # Add noise
    prev_targets = targets - np.random.randn(num_samples, num_countries) * 3
    
    predicted_deltas = predictions - prev_targets
    true_deltas = targets - prev_targets
    
    lower_bounds = predictions - 5
    upper_bounds = predictions + 5
    
    # Test individual metrics
    print("\n1. Testing individual metrics...")
    rmse = compute_rmse(predictions, targets)
    mae = compute_mae(predictions, targets)
    r2 = compute_r2(predictions, targets)
    mape = compute_mape(predictions, targets)
    dir_acc = compute_directional_accuracy(predictions, targets, prev_targets)
    ripple_corr = compute_ripple_correlation(predicted_deltas, true_deltas)
    
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Directional Accuracy: {dir_acc:.2%}")
    print(f"   Ripple Correlation: {ripple_corr:.4f}")
    
    # Test per-country metrics
    print("\n2. Testing per-country metrics...")
    rmse_per_country = compute_rmse(predictions, targets, per_country=True)
    print(f"   Per-country RMSE shape: {rmse_per_country.shape}")
    print(f"   Per-country RMSE range: [{rmse_per_country.min():.4f}, {rmse_per_country.max():.4f}]")
    
    # Test coverage metrics
    print("\n3. Testing coverage metrics...")
    coverage = compute_coverage(targets, lower_bounds, upper_bounds)
    print(f"   Coverage: {coverage['coverage']:.2%}")
    print(f"   Interval width: {coverage['interval_width']:.4f}")
    
    # Test MetricsCalculator
    print("\n4. Testing MetricsCalculator...")
    country_names = [f"Country_{i}" for i in range(num_countries)]
    calculator = MetricsCalculator(country_names)
    
    all_metrics = calculator.compute_all_metrics(
        predictions, targets, prev_targets,
        predicted_deltas, true_deltas,
        lower_bounds, upper_bounds
    )
    
    calculator.print_summary(all_metrics)
    
    # Create report
    print("\n5. Creating DataFrame report...")
    report = calculator.create_report(all_metrics)
    print(report.head())
    
    print("\n✓ Metrics tests passed!")

