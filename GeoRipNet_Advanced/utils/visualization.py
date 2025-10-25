"""
Visualization Utilities for GeoRipNet.

Provides comprehensive visualization tools for:
- Training history
- Prediction vs actual comparisons
- Ripple effect heatmaps
- Attention weights
- Uncertainty quantification
- Per-country performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class GeoRipNetVisualizer:
    """
    Comprehensive visualization suite for GeoRipNet.
    """
    
    def __init__(self, save_dir: str = 'plots'):
        """
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(
        self,
        history: Dict[str, List],
        save_name: str = 'training_history.png'
    ):
        """
        Plot training and validation loss over epochs.
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'learning_rates'
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training history plot to {self.save_dir / save_name}")
    
    def plot_predictions_vs_actual(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        country_names: Optional[List[str]] = None,
        save_name: str = 'predictions_vs_actual.png'
    ):
        """
        Plot predictions vs actual values with scatter and time series.
        
        Args:
            predictions: (num_samples, num_countries)
            targets: (num_samples, num_countries)
            country_names: Optional list of country names
            save_name: Filename to save plot
        """
        num_countries = predictions.shape[1]
        
        # Select up to 6 countries to display
        display_countries = min(6, num_countries)
        indices = np.linspace(0, num_countries-1, display_countries, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
            
            pred = predictions[:, idx]
            true = targets[:, idx]
            
            country_name = country_names[idx] if country_names else f"Country {idx}"
            
            # Scatter plot
            axes[i].scatter(true, pred, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
            
            # R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(true, pred)
            
            axes[i].set_xlabel('Actual Price')
            axes[i].set_ylabel('Predicted Price')
            axes[i].set_title(f'{country_name}\n(R² = {r2:.3f})', fontsize=11, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Predictions vs Actual (Scatter)', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved predictions vs actual plot to {self.save_dir / save_name}")
    
    def plot_time_series_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: Optional[List] = None,
        country_names: Optional[List[str]] = None,
        num_countries_to_plot: int = 4,
        save_name: str = 'time_series_predictions.png'
    ):
        """
        Plot time series of predictions vs actual for selected countries.
        
        Args:
            predictions: (num_samples, num_countries)
            targets: (num_samples, num_countries)
            timestamps: Optional list of timestamps
            country_names: Optional list of country names
            num_countries_to_plot: Number of countries to plot
            save_name: Filename to save plot
        """
        num_countries = predictions.shape[1]
        display_countries = min(num_countries_to_plot, num_countries)
        indices = np.linspace(0, num_countries-1, display_countries, dtype=int)
        
        if timestamps is None:
            timestamps = np.arange(len(predictions))
        
        fig, axes = plt.subplots(display_countries, 1, figsize=(14, 3 * display_countries))
        if display_countries == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            pred = predictions[:, idx]
            true = targets[:, idx]
            
            country_name = country_names[idx] if country_names else f"Country {idx}"
            
            axes[i].plot(timestamps, true, 'b-', label='Actual', linewidth=2, alpha=0.7)
            axes[i].plot(timestamps, pred, 'r--', label='Predicted', linewidth=2, alpha=0.7)
            axes[i].fill_between(timestamps, true, pred, alpha=0.2)
            
            axes[i].set_ylabel('Price')
            axes[i].set_title(country_name, fontsize=12, fontweight='bold')
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time')
        
        plt.suptitle('Time Series: Predictions vs Actual', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved time series predictions plot to {self.save_dir / save_name}")
    
    def plot_ripple_heatmap(
        self,
        ripple_matrix: np.ndarray,
        country_names: Optional[List[str]] = None,
        save_name: str = 'ripple_heatmap.png'
    ):
        """
        Plot heatmap of ripple effects between countries.
        
        Args:
            ripple_matrix: (num_countries, num_countries) - Influence matrix
            country_names: Optional list of country names
            save_name: Filename to save plot
        """
        plt.figure(figsize=(14, 12))
        
        if country_names is None:
            country_names = [f"C{i}" for i in range(ripple_matrix.shape[0])]
        
        # Plot heatmap
        sns.heatmap(
            ripple_matrix,
            xticklabels=country_names,
            yticklabels=country_names,
            cmap='RdYlBu_r',
            center=0,
            annot=False,
            fmt='.2f',
            cbar_kws={'label': 'Ripple Influence'},
            linewidths=0.5
        )
        
        plt.title('Ripple Effect Influence Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Target Country', fontsize=12)
        plt.ylabel('Source Country', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ripple heatmap to {self.save_dir / save_name}")
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        layer_name: str = 'Attention',
        save_name: str = 'attention_weights.png'
    ):
        """
        Plot attention weight heatmap.
        
        Args:
            attention_weights: (num_heads, seq_len, seq_len) or (seq_len, seq_len)
            layer_name: Name of the attention layer
            save_name: Filename to save plot
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # If multi-head, average across heads
        if attention_weights.ndim == 3:
            attention_weights = attention_weights.mean(axis=0)
        
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            attention_weights,
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'},
            square=True
        )
        
        plt.title(f'{layer_name} Attention Weights', fontsize=14, fontweight='bold')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved attention weights plot to {self.save_dir / save_name}")
    
    def plot_uncertainty_quantification(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        country_idx: int = 0,
        country_name: Optional[str] = None,
        save_name: str = 'uncertainty_quantification.png'
    ):
        """
        Plot predictions with uncertainty intervals.
        
        Args:
            predictions: (num_samples,)
            targets: (num_samples,)
            lower_bounds: (num_samples,)
            upper_bounds: (num_samples,)
            country_idx: Country index for title
            country_name: Optional country name
            save_name: Filename to save plot
        """
        timestamps = np.arange(len(predictions))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot actual values
        ax.plot(timestamps, targets, 'b-', label='Actual', linewidth=2, zorder=3)
        
        # Plot predictions
        ax.plot(timestamps, predictions, 'r--', label='Predicted', linewidth=2, zorder=2)
        
        # Plot uncertainty intervals
        ax.fill_between(
            timestamps,
            lower_bounds,
            upper_bounds,
            alpha=0.3,
            color='orange',
            label='80% Confidence Interval',
            zorder=1
        )
        
        # Calculate coverage
        within_bounds = (targets >= lower_bounds) & (targets <= upper_bounds)
        coverage = within_bounds.mean()
        
        country_label = country_name if country_name else f"Country {country_idx}"
        ax.set_title(
            f'Uncertainty Quantification: {country_label}\n(Coverage: {coverage:.1%})',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved uncertainty quantification plot to {self.save_dir / save_name}")
    
    def plot_per_country_metrics(
        self,
        metrics: Dict[str, np.ndarray],
        country_names: Optional[List[str]] = None,
        save_name: str = 'per_country_metrics.png'
    ):
        """
        Plot bar charts of per-country metrics.
        
        Args:
            metrics: Dictionary with metric arrays (e.g., {'rmse': [...], 'r2': [...]})
            country_names: Optional list of country names
            save_name: Filename to save plot
        """
        num_metrics = len(metrics)
        num_countries = len(list(metrics.values())[0])
        
        if country_names is None:
            country_names = [f"C{i}" for i in range(num_countries)]
        
        fig, axes = plt.subplots(num_metrics, 1, figsize=(14, 4 * num_metrics))
        if num_metrics == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, num_countries))
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            axes[i].bar(range(num_countries), values, color=colors, edgecolor='black', linewidth=0.5)
            axes[i].set_xticks(range(num_countries))
            axes[i].set_xticklabels(country_names, rotation=45, ha='right')
            axes[i].set_ylabel(metric_name.upper())
            axes[i].set_title(f'Per-Country {metric_name.upper()}', fontsize=12, fontweight='bold')
            axes[i].grid(True, axis='y', alpha=0.3)
            
            # Add mean line
            mean_val = np.mean(values)
            axes[i].axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            axes[i].legend()
        
        plt.suptitle('Per-Country Performance Metrics', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved per-country metrics plot to {self.save_dir / save_name}")
    
    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_name: str = 'error_distribution.png'
    ):
        """
        Plot distribution of prediction errors.
        
        Args:
            predictions: (num_samples, num_countries)
            targets: (num_samples, num_countries)
            save_name: Filename to save plot
        """
        errors = predictions - targets
        errors_flat = errors.flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(errors_flat, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Prediction Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution (Histogram)', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(errors_flat, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Prediction Error Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved error distribution plot to {self.save_dir / save_name}")
    
    def create_comprehensive_report(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        prev_targets: np.ndarray,
        history: Dict,
        country_names: Optional[List[str]] = None,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None
    ):
        """
        Create a comprehensive visualization report with all plots.
        
        Args:
            predictions: Model predictions
            targets: Actual values
            prev_targets: Previous time step values
            history: Training history
            country_names: Optional country names
            lower_bounds: Optional lower confidence bounds
            upper_bounds: Optional upper confidence bounds
        """
        print("\nGenerating comprehensive visualization report...")
        
        # 1. Training history
        self.plot_training_history(history, 'training_history.png')
        
        # 2. Predictions vs actual
        self.plot_predictions_vs_actual(predictions, targets, country_names, 'predictions_vs_actual.png')
        
        # 3. Time series
        self.plot_time_series_predictions(predictions, targets, country_names=country_names)
        
        # 4. Error distribution
        self.plot_error_distribution(predictions, targets)
        
        # 5. Per-country metrics
        from .metrics import compute_rmse, compute_mae, compute_r2, compute_directional_accuracy
        
        metrics_dict = {
            'rmse': compute_rmse(predictions, targets, per_country=True),
            'mae': compute_mae(predictions, targets, per_country=True),
            'r2': compute_r2(predictions, targets, per_country=True),
            'dir_acc': compute_directional_accuracy(predictions, targets, prev_targets, per_country=True)
        }
        self.plot_per_country_metrics(metrics_dict, country_names)
        
        # 6. Uncertainty (if available)
        if lower_bounds is not None and upper_bounds is not None:
            self.plot_uncertainty_quantification(
                predictions[:, 0], targets[:, 0],
                lower_bounds[:, 0], upper_bounds[:, 0],
                country_idx=0,
                country_name=country_names[0] if country_names else None
            )
        
        print(f"\n✓ Comprehensive report generated in {self.save_dir}/")


if __name__ == "__main__":
    # Test visualization
    print("Testing GeoRipNet Visualization...")
    
    num_samples = 100
    num_countries = 10
    
    # Generate dummy data
    np.random.seed(42)
    targets = np.random.randn(num_samples, num_countries) * 10 + 80
    predictions = targets + np.random.randn(num_samples, num_countries) * 2
    prev_targets = targets - np.random.randn(num_samples, num_countries) * 3
    
    lower_bounds = predictions - 5
    upper_bounds = predictions + 5
    
    history = {
        'train_loss': np.random.exponential(2, 50).tolist(),
        'val_loss': np.random.exponential(2.2, 50).tolist(),
        'learning_rates': (np.logspace(-3, -5, 50)).tolist()
    }
    
    country_names = [f"Country_{i}" for i in range(num_countries)]
    
    # Create visualizer
    visualizer = GeoRipNetVisualizer(save_dir='test_plots')
    
    # Test plots
    print("\n1. Testing training history plot...")
    visualizer.plot_training_history(history)
    
    print("\n2. Testing predictions vs actual plot...")
    visualizer.plot_predictions_vs_actual(predictions, targets, country_names)
    
    print("\n3. Testing time series plot...")
    visualizer.plot_time_series_predictions(predictions, targets, country_names=country_names)
    
    print("\n4. Testing ripple heatmap...")
    ripple_matrix = np.random.rand(num_countries, num_countries)
    visualizer.plot_ripple_heatmap(ripple_matrix, country_names)
    
    print("\n5. Testing uncertainty plot...")
    visualizer.plot_uncertainty_quantification(
        predictions[:, 0], targets[:, 0],
        lower_bounds[:, 0], upper_bounds[:, 0],
        country_name=country_names[0]
    )
    
    print("\n6. Testing error distribution...")
    visualizer.plot_error_distribution(predictions, targets)
    
    print("\n✓ Visualization tests passed!")

