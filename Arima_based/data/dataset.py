"""
Dataset Module for RippleNet-TFT
Creates sliding windows and prepares data for TFT model
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RippleNetDataset(Dataset):
    """Dataset class for RippleNet-TFT model"""
    
    def __init__(self, data: pd.DataFrame, config: Dict, 
                 lookback_window: int = 60, forecast_horizon: int = 1,
                 target_columns: List[str] = None, 
                 past_target_columns: List[str] = None,
                 observed_covariates: List[str] = None,
                 known_future_columns: List[str] = None):
        
        self.data = data.copy()
        self.config = config
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        
        # Define column groups
        self.target_columns = target_columns or self._get_target_columns()
        self.past_target_columns = past_target_columns or self._get_past_target_columns()
        self.observed_covariates = observed_covariates or self._get_observed_covariates()
        self.known_future_columns = known_future_columns or self._get_known_future_columns()
        
        # Initialize scalers
        self.scalers = {}
        self._fit_scalers()
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences")
        logger.info(f"Target columns: {self.target_columns}")
        logger.info(f"Past target columns: {self.past_target_columns}")
        logger.info(f"Observed covariates: {self.observed_covariates}")
        logger.info(f"Known future columns: {self.known_future_columns}")
    
    def _get_target_columns(self) -> List[str]:
        """Get target columns (next day prices)"""
        target_cols = []
        for col in self.data.columns:
            if col.endswith('_next_day') and not col.endswith('_change'):
                target_cols.append(col)
        
        if not target_cols:
            # Fallback to price columns
            target_cols = [col for col in self.data.columns if col.endswith('_close')]
        
        return target_cols
    
    def _get_past_target_columns(self) -> List[str]:
        """Get past target columns (historical prices and residuals)"""
        past_target_cols = []
        
        # Price columns
        for col in self.data.columns:
            if col.endswith('_close') or col.endswith('_open') or col.endswith('_high') or col.endswith('_low'):
                past_target_cols.append(col)
        
        # ARIMA residuals
        for col in self.data.columns:
            if col.endswith('_arima_residual'):
                past_target_cols.append(col)
        
        # Technical indicators
        for col in self.data.columns:
            if any(x in col for x in ['rsi', 'macd', 'volatility', 'sma']):
                past_target_cols.append(col)
        
        return past_target_cols
    
    def _get_observed_covariates(self) -> List[str]:
        """Get observed covariates (news, ripple, macro indicators)"""
        observed_cols = []
        
        # News embeddings
        for col in self.data.columns:
            if col.startswith('news_'):
                observed_cols.append(col)
        
        # Ripple embeddings
        for col in self.data.columns:
            if col.endswith('_ripple'):
                observed_cols.append(col)
        
        # Macro indicators
        macro_cols = ['epu', 'gpr', 'vix', 'dxy', 'fed_funds_rate']
        for col in macro_cols:
            if col in self.data.columns:
                observed_cols.append(col)
        
        # GDELT features
        for col in self.data.columns:
            if col.startswith('gdelt_'):
                observed_cols.append(col)
        
        return observed_cols
    
    def _get_known_future_columns(self) -> List[str]:
        """Get known future columns (calendar features)"""
        known_cols = []
        
        # Calendar features
        calendar_cols = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter', 'is_weekend', 'is_holiday']
        for col in calendar_cols:
            if col in self.data.columns:
                known_cols.append(col)
        
        return known_cols
    
    def _fit_scalers(self):
        """Fit scalers for different column groups"""
        logger.info("Fitting scalers")
        
        # Scale past target columns
        if self.past_target_columns:
            self.scalers['past_target'] = StandardScaler()
            self.scalers['past_target'].fit(self.data[self.past_target_columns].fillna(0))
        
        # Scale observed covariates
        if self.observed_covariates:
            self.scalers['observed'] = StandardScaler()
            self.scalers['observed'].fit(self.data[self.observed_covariates].fillna(0))
        
        # Scale known future columns
        if self.known_future_columns:
            self.scalers['known_future'] = StandardScaler()
            self.scalers['known_future'].fit(self.data[self.known_future_columns].fillna(0))
        
        # Scale target columns
        if self.target_columns:
            self.scalers['target'] = StandardScaler()
            self.scalers['target'].fit(self.data[self.target_columns].fillna(0))
    
    def _create_sequences(self) -> List[Dict]:
        """Create sliding window sequences"""
        logger.info("Creating sequences")
        
        sequences = []
        
        for i in range(self.lookback_window, len(self.data) - self.forecast_horizon + 1):
            # Get sequence data
            sequence_data = self.data.iloc[i - self.lookback_window:i + self.forecast_horizon]
            
            # Past target (historical data)
            past_target = sequence_data.iloc[:-self.forecast_horizon][self.past_target_columns].values
            if self.past_target_columns:
                past_target = self.scalers['past_target'].transform(past_target)
            
            # Observed covariates (historical)
            observed_cov = sequence_data.iloc[:-self.forecast_horizon][self.observed_covariates].values
            if self.observed_covariates:
                observed_cov = self.scalers['observed'].transform(observed_cov)
            
            # Known future (calendar features for forecast period)
            known_future = sequence_data.iloc[-self.forecast_horizon:][self.known_future_columns].values
            if self.known_future_columns:
                known_future = self.scalers['known_future'].transform(known_future)
            
            # Target (future prices)
            target = sequence_data.iloc[-self.forecast_horizon:][self.target_columns].values
            if self.target_columns:
                target = self.scalers['target'].transform(target)
            
            sequences.append({
                'past_target': torch.FloatTensor(past_target),
                'observed_covariates': torch.FloatTensor(observed_cov),
                'known_future': torch.FloatTensor(known_future),
                'target': torch.FloatTensor(target)
            })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def get_scalers(self):
        """Get fitted scalers"""
        return self.scalers

class RippleNetDataModule:
    """Data module for managing train/val/test splits"""
    
    def __init__(self, data: pd.DataFrame, config: Dict):
        self.data = data
        self.config = config
        
        # Get split ratios
        self.train_split = config.get('train_split', 0.7)
        self.val_split = config.get('val_split', 0.15)
        self.test_split = config.get('test_split', 0.15)
        
        # Get training parameters
        self.lookback_window = config.get('lookback_window', 60)
        self.forecast_horizon = config.get('forecast_horizon', 1)
        self.batch_size = config.get('batch_size', 32)
        
        # Create splits
        self._create_splits()
    
    def _create_splits(self):
        """Create train/validation/test splits"""
        logger.info("Creating data splits")
        
        n_samples = len(self.data)
        train_end = int(n_samples * self.train_split)
        val_end = int(n_samples * (self.train_split + self.val_split))
        
        self.train_data = self.data.iloc[:train_end]
        self.val_data = self.data.iloc[train_end:val_end]
        self.test_data = self.data.iloc[val_end:]
        
        logger.info(f"Train: {len(self.train_data)} samples")
        logger.info(f"Validation: {len(self.val_data)} samples")
        logger.info(f"Test: {len(self.test_data)} samples")
    
    def get_train_dataset(self) -> RippleNetDataset:
        """Get training dataset"""
        return RippleNetDataset(
            self.train_data, 
            self.config,
            lookback_window=self.lookback_window,
            forecast_horizon=self.forecast_horizon
        )
    
    def get_val_dataset(self) -> RippleNetDataset:
        """Get validation dataset"""
        return RippleNetDataset(
            self.val_data, 
            self.config,
            lookback_window=self.lookback_window,
            forecast_horizon=self.forecast_horizon
        )
    
    def get_test_dataset(self) -> RippleNetDataset:
        """Get test dataset"""
        return RippleNetDataset(
            self.test_data, 
            self.config,
            lookback_window=self.lookback_window,
            forecast_horizon=self.forecast_horizon
        )
    
    def get_data_loaders(self, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get data loaders for train/val/test"""
        train_dataset = self.get_train_dataset()
        val_dataset = self.get_val_dataset()
        test_dataset = self.get_test_dataset()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_scalers(self):
        """Get fitted scalers"""
        return self.get_train_dataset().get_scalers()

def create_dataset_from_merged_data(merged_data_path: str, config: Dict) -> RippleNetDataModule:
    """Create dataset from merged data file"""
    logger.info(f"Loading merged data from {merged_data_path}")
    
    # Load merged data
    data = pd.read_csv(merged_data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort by date
    data = data.sort_values('date').reset_index(drop=True)
    
    # Create data module
    data_module = RippleNetDataModule(data, config)
    
    logger.info(f"Created dataset with {len(data)} samples")
    return data_module

def main():
    """Main function for dataset creation"""
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'CL=F_close': 100 + np.cumsum(np.random.randn(1000) * 0.02),
        'NG=F_close': 50 + np.cumsum(np.random.randn(1000) * 0.03),
        'CL=F_next_day': 100 + np.cumsum(np.random.randn(1000) * 0.02),
        'NG=F_next_day': 50 + np.cumsum(np.random.randn(1000) * 0.03),
        'epu': np.random.normal(100, 20, 1000),
        'gpr': np.random.normal(50, 15, 1000),
        'year': dates.year,
        'month': dates.month,
        'dayofweek': dates.dayofweek
    })
    
    # Create data module
    data_module = RippleNetDataModule(sample_data, config)
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_module.get_data_loaders()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch shapes:")
        print(f"  Past target: {batch['past_target'].shape}")
        print(f"  Observed covariates: {batch['observed_covariates'].shape}")
        print(f"  Known future: {batch['known_future'].shape}")
        print(f"  Target: {batch['target'].shape}")
        break
    
    print("Dataset creation completed successfully!")

if __name__ == "__main__":
    main()
