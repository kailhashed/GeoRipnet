"""
Evaluation Script for RippleNet-TFT
Comprehensive evaluation with multiple metrics and visualizations
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import argparse
import json
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.dataset import RippleNetDataModule, create_dataset_from_merged_data
from model.ripple_tft import create_model
from data.arima_baseline import ARIMABaseline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RippleNetEvaluator:
    """Evaluation class for RippleNet-TFT"""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = create_model(config)
        self.model.to(device)
        
        # Initialize data module
        self.data_module = None
        
        # Results storage
        self.results = {}
        
        # Create output directories
        self.results_dir = Path(config['paths']['results'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, checkpoint_path: str):
        """Load trained model"""
        logger.info(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def load_data(self, merged_data_path: str):
        """Load and prepare data"""
        logger.info(f"Loading data from {merged_data_path}")
        
        self.data_module = create_dataset_from_merged_data(merged_data_path, self.config)
        
        # Get data loaders
        self.train_loader, self.val_loader, self.test_loader = self.data_module.get_data_loaders()
        
        logger.info(f"Data loaded successfully")
        logger.info(f"Test batches: {len(self.test_loader)}")
    
    def evaluate_model(self, data_loader: DataLoader, model_name: str = "Model") -> Dict[str, float]:
        """Evaluate model on given data loader"""
        logger.info(f"Evaluating {model_name}")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                past_target = batch['past_target'].to(self.device)
                observed_covariates = batch['observed_covariates'].to(self.device)
                known_future = batch['known_future'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                predictions = self.model(past_target, observed_covariates, known_future)
                
                # Store predictions and targets
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Flatten for metrics calculation
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        
        # Calculate metrics
        metrics = self._calculate_metrics(targets_flat, predictions_flat)
        
        # Store results
        self.results[model_name] = {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics
        }
        
        logger.info(f"{model_name} Metrics: {metrics}")
        return metrics
    
    def _calculate_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Remove NaN values
        mask = ~(np.isnan(targets) | np.isnan(predictions))
        targets = targets[mask]
        predictions = predictions[mask]
        
        if len(targets) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'pearson': np.nan, 'directional_accuracy': np.nan}
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        
        # MAPE (handle division by zero)
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # R² score
        r2 = r2_score(targets, predictions)
        
        # Pearson correlation
        pearson, _ = pearsonr(targets, predictions)
        
        # Directional accuracy
        target_direction = np.sign(targets)
        pred_direction = np.sign(predictions)
        directional_accuracy = np.mean(target_direction == pred_direction) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'pearson': pearson,
            'directional_accuracy': directional_accuracy
        }
    
    def evaluate_arima_baseline(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate ARIMA baseline"""
        logger.info("Evaluating ARIMA baseline")
        
        # Initialize ARIMA baseline
        arima_baseline = ARIMABaseline(self.config)
        
        # Evaluate ARIMA models
        arima_results = arima_baseline.evaluate_arima_models(data)
        
        # Calculate average metrics
        if arima_results:
            avg_metrics = {
                'rmse': np.mean([r['rmse'] for r in arima_results.values()]),
                'mae': np.mean([r['mae'] for r in arima_results.values()]),
                'mape': np.mean([r['mape'] for r in arima_results.values()]),
                'r2': np.nan,  # ARIMA doesn't provide R²
                'pearson': np.nan,  # ARIMA doesn't provide correlation
                'directional_accuracy': np.nan  # ARIMA doesn't provide directional accuracy
            }
        else:
            avg_metrics = {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'pearson': np.nan, 'directional_accuracy': np.nan}
        
        self.results['ARIMA'] = {
            'metrics': avg_metrics,
            'detailed_results': arima_results
        }
        
        logger.info(f"ARIMA Baseline Metrics: {avg_metrics}")
        return avg_metrics
    
    def create_visualizations(self):
        """Create evaluation visualizations"""
        logger.info("Creating visualizations")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Predictions vs Actual
        self._plot_predictions_vs_actual()
        
        # 2. Time series predictions
        self._plot_time_series_predictions()
        
        # 3. Metrics comparison
        self._plot_metrics_comparison()
        
        # 4. Error distribution
        self._plot_error_distribution()
        
        # 5. Attention visualization (if available)
        self._plot_attention_weights()
        
        logger.info("Visualizations created successfully")
    
    def _plot_predictions_vs_actual(self):
        """Plot predictions vs actual values"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        model_names = list(self.results.keys())
        
        for i, model_name in enumerate(model_names[:4]):
            if model_name in self.results and 'predictions' in self.results[model_name]:
                predictions = self.results[model_name]['predictions'].flatten()
                targets = self.results[model_name]['targets'].flatten()
                
                # Remove NaN values
                mask = ~(np.isnan(predictions) | np.isnan(targets))
                predictions = predictions[mask]
                targets = targets[mask]
                
                if len(predictions) > 0:
                    axes[i].scatter(targets, predictions, alpha=0.6, s=20)
                    axes[i].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
                    axes[i].set_xlabel('Actual')
                    axes[i].set_ylabel('Predicted')
                    axes[i].set_title(f'{model_name} - Predictions vs Actual')
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series_predictions(self):
        """Plot time series predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        model_names = list(self.results.keys())
        
        for i, model_name in enumerate(model_names[:4]):
            if model_name in self.results and 'predictions' in self.results[model_name]:
                predictions = self.results[model_name]['predictions'].flatten()
                targets = self.results[model_name]['targets'].flatten()
                
                # Remove NaN values
                mask = ~(np.isnan(predictions) | np.isnan(targets))
                predictions = predictions[mask]
                targets = targets[mask]
                
                if len(predictions) > 0:
                    # Plot last 100 points for clarity
                    n_points = min(100, len(predictions))
                    x = range(n_points)
                    
                    axes[i].plot(x, targets[-n_points:], label='Actual', linewidth=2)
                    axes[i].plot(x, predictions[-n_points:], label='Predicted', linewidth=2)
                    axes[i].set_xlabel('Time Steps')
                    axes[i].set_ylabel('Price')
                    axes[i].set_title(f'{model_name} - Time Series Predictions')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'time_series_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self):
        """Plot metrics comparison"""
        # Prepare data
        metrics_data = []
        for model_name, result in self.results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                for metric_name, value in metrics.items():
                    if not np.isnan(value):
                        metrics_data.append({
                            'Model': model_name,
                            'Metric': metric_name.upper(),
                            'Value': value
                        })
        
        if not metrics_data:
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Create subplots for different metrics
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE', 'R2', 'PEARSON', 'DIRECTIONAL_ACCURACY']
        n_metrics = len([m for m in metrics_to_plot if m in df['Metric'].values])
        
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if i >= len(axes):
                break
            
            metric_data = df[df['Metric'] == metric]
            if len(metric_data) > 0:
                sns.barplot(data=metric_data, x='Model', y='Value', ax=axes[i])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self):
        """Plot error distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        model_names = list(self.results.keys())
        
        for i, model_name in enumerate(model_names[:4]):
            if model_name in self.results and 'predictions' in self.results[model_name]:
                predictions = self.results[model_name]['predictions'].flatten()
                targets = self.results[model_name]['targets'].flatten()
                
                # Remove NaN values
                mask = ~(np.isnan(predictions) | np.isnan(targets))
                predictions = predictions[mask]
                targets = targets[mask]
                
                if len(predictions) > 0:
                    errors = predictions - targets
                    
                    axes[i].hist(errors, bins=50, alpha=0.7, density=True)
                    axes[i].axvline(0, color='red', linestyle='--', linewidth=2)
                    axes[i].set_xlabel('Prediction Error')
                    axes[i].set_ylabel('Density')
                    axes[i].set_title(f'{model_name} - Error Distribution')
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_weights(self):
        """Plot attention weights (placeholder)"""
        # This would require access to attention weights from the model
        # For now, create a placeholder plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create dummy attention weights
        time_steps = 60
        variables = 10
        attention_weights = np.random.rand(time_steps, variables)
        
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        ax.set_xlabel('Variables')
        ax.set_ylabel('Time Steps')
        ax.set_title('Attention Weights (Example)')
        
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'attention_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report")
        
        # Create summary table
        summary_data = []
        for model_name, result in self.results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                summary_data.append({
                    'Model': model_name,
                    'RMSE': f"{metrics.get('rmse', 0):.4f}",
                    'MAE': f"{metrics.get('mae', 0):.4f}",
                    'MAPE': f"{metrics.get('mape', 0):.2f}%",
                    'R²': f"{metrics.get('r2', 0):.4f}",
                    'Pearson': f"{metrics.get('pearson', 0):.4f}",
                    'Directional Accuracy': f"{metrics.get('directional_accuracy', 0):.2f}%"
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = self.results_dir / 'evaluation_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Create detailed report
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'config': self.config,
            'summary': summary_data,
            'detailed_results': self.results
        }
        
        report_path = self.results_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {report_path}")
        logger.info(f"Summary saved to {summary_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
    
    def run_evaluation(self, checkpoint_path: str, merged_data_path: str):
        """Run complete evaluation"""
        logger.info("Starting evaluation")
        
        # Load model
        self.load_model(checkpoint_path)
        
        # Load data
        self.load_data(merged_data_path)
        
        # Evaluate RippleNet-TFT
        self.evaluate_model(self.test_loader, "RippleNet-TFT")
        
        # Evaluate ARIMA baseline
        data = pd.read_csv(merged_data_path)
        self.evaluate_arima_baseline(data)
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        logger.info("Evaluation completed successfully!")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate RippleNet-TFT')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data', type=str, default='data/merged.csv', help='Merged data file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = RippleNetEvaluator(config, device=args.device)
    
    # Run evaluation
    evaluator.run_evaluation(args.checkpoint, args.data)

if __name__ == "__main__":
    main()
