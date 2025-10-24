#!/usr/bin/env python3
"""
Fixed Dataset Implementation for RippleNet-TFT
Addresses target alignment, scaling, and sequence creation issues
"""

import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedRippleNetDataset(Dataset):
    """Fixed dataset with proper target alignment and scaling"""
    
    def __init__(self, data: pd.DataFrame, config: Dict, split: str = 'train'):
        self.data = data
        self.config = config
        self.split = split
        
        # Configuration
        self.lookback_window = config['training']['lookback_window']
        self.forecast_horizon = config['training']['forecast_horizon']
        self.target_columns = config['data']['target_columns']
        self.past_target_columns = config['data']['past_target_columns']
        self.observed_cov_columns = config['data']['observed_cov_columns']
        self.known_future_columns = config['data']['known_future_columns']
        
        # Initialize scalers
        self.scalers = {}
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        logger.info(f"Created {len(self.sequences)} sequences for {split} split")
    
    def _create_sequences(self) -> List[Dict]:
        """Create sequences with proper target alignment"""
        sequences = []
        
        # Ensure we have enough data
        min_length = self.lookback_window + self.forecast_horizon
        if len(self.data) < min_length:
            logger.warning(f"Not enough data: {len(self.data)} < {min_length}")
            return sequences
        
        # Create sequences with proper alignment
        for i in range(len(self.data) - min_length + 1):
            # Input window: [i, i+lookback_window)
            input_end = i + self.lookback_window
            
            # Target window: [i+lookback_window, i+lookback_window+forecast_horizon)
            target_start = i + self.lookback_window
            target_end = target_start + self.forecast_horizon
            
            # Ensure we don't go out of bounds
            if target_end > len(self.data):
                continue
            
            # Extract sequences
            sequence_data = self.data.iloc[i:input_end]
            target_data = self.data.iloc[target_start:target_end]
            
            # Past target features (OHLCV + technical indicators)
            past_target = sequence_data[self.past_target_columns].values
            
            # Observed covariates (news, economic indicators)
            observed_cov = sequence_data[self.observed_cov_columns].values if self.observed_cov_columns else np.zeros((self.lookback_window, 1))
            
            # Known future features (calendar features)
            known_future = target_data[self.known_future_columns].values if self.known_future_columns else np.zeros((self.forecast_horizon, 1))
            
            # Target values (next day prices)
            target = target_data[self.target_columns].values
            
            # CRITICAL FIX: Ensure target is properly aligned
            if target.shape[0] != self.forecast_horizon:
                logger.warning(f"Target shape mismatch: {target.shape[0]} != {self.forecast_horizon}")
                continue
            
            # Store raw values for inverse transformation
            sequences.append({
                'past_target': torch.FloatTensor(past_target),
                'observed_covariates': torch.FloatTensor(observed_cov),
                'known_future': torch.FloatTensor(known_future),
                'target': torch.FloatTensor(target),
                'raw_target': target.copy(),  # Store raw values
                'sequence_index': i  # For debugging
            })
        
        return sequences
    
    def fit_scalers(self, train_sequences: List[Dict]):
        """Fit scalers on training data only"""
        logger.info("Fitting scalers on training data...")
        
        # Collect all data for scaling
        all_past_target = []
        all_observed_cov = []
        all_known_future = []
        all_target = []
        
        for seq in train_sequences:
            all_past_target.append(seq['past_target'].numpy())
            all_observed_cov.append(seq['observed_covariates'].numpy())
            all_known_future.append(seq['known_future'].numpy())
            all_target.append(seq['target'].numpy())
        
        # Fit scalers
        self.scalers['past_target'] = StandardScaler()
        self.scalers['past_target'].fit(np.vstack(all_past_target))
        
        if all_observed_cov[0].shape[1] > 0:
            self.scalers['observed_cov'] = StandardScaler()
            self.scalers['observed_cov'].fit(np.vstack(all_observed_cov))
        
        if all_known_future[0].shape[1] > 0:
            self.scalers['known_future'] = StandardScaler()
            self.scalers['known_future'].fit(np.vstack(all_known_future))
        
        self.scalers['target'] = StandardScaler()
        self.scalers['target'].fit(np.vstack(all_target))
        
        logger.info("Scalers fitted successfully")
    
    def apply_scaling(self):
        """Apply fitted scalers to sequences"""
        logger.info("Applying scaling to sequences...")
        
        for seq in self.sequences:
            # Scale past target
            past_target = seq['past_target'].numpy()
            past_target_scaled = self.scalers['past_target'].transform(past_target)
            seq['past_target'] = torch.FloatTensor(past_target_scaled)
            
            # Scale observed covariates
            if 'observed_cov' in self.scalers:
                observed_cov = seq['observed_covariates'].numpy()
                observed_cov_scaled = self.scalers['observed_cov'].transform(observed_cov)
                seq['observed_covariates'] = torch.FloatTensor(observed_cov_scaled)
            
            # Scale known future
            if 'known_future' in self.scalers:
                known_future = seq['known_future'].numpy()
                known_future_scaled = self.scalers['known_future'].transform(known_future)
                seq['known_future'] = torch.FloatTensor(known_future_scaled)
            
            # Scale target
            target = seq['target'].numpy()
            target_scaled = self.scalers['target'].transform(target)
            seq['target'] = torch.FloatTensor(target_scaled)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def get_scalers(self):
        """Get fitted scalers"""
        return self.scalers

class FixedRippleNetDataModule:
    """Fixed data module with proper train/val/test splits"""
    
    def __init__(self, data: pd.DataFrame, config: Dict):
        self.data = data
        self.config = config
        
        # Get split ratios
        train_ratio = config['data']['train_ratio']
        val_ratio = config['data']['val_ratio']
        
        # Calculate split indices
        total_len = len(data)
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))
        
        # Create splits
        self.train_data = data.iloc[:train_end]
        self.val_data = data.iloc[train_end:val_end]
        self.test_data = data.iloc[val_end:]
        
        logger.info(f"Data splits - Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
        
        # Create datasets
        self.train_dataset = FixedRippleNetDataset(self.train_data, config, 'train')
        self.val_dataset = FixedRippleNetDataset(self.val_data, config, 'val')
        self.test_dataset = FixedRippleNetDataset(self.test_data, config, 'test')
        
        # Fit scalers on training data
        self.train_dataset.fit_scalers(self.train_dataset.sequences)
        
        # Apply scaling to all datasets
        self.train_dataset.apply_scaling()
        self.val_dataset.apply_scaling()
        self.test_dataset.apply_scaling()
        
        # Copy scalers to other datasets
        self.val_dataset.scalers = self.train_dataset.scalers
        self.test_dataset.scalers = self.train_dataset.scalers
    
    def get_data_loaders(self, batch_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get data loaders for train/val/test"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def get_scalers(self):
        """Get fitted scalers"""
        return self.train_dataset.get_scalers()

def create_fixed_dataset_from_merged_data(data_path: str, config: Dict) -> FixedRippleNetDataModule:
    """Create fixed dataset from merged data"""
    logger.info(f"Loading merged data from {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    
    # Fill missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"Loaded data with shape: {data.shape}")
    logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Create data module
    data_module = FixedRippleNetDataModule(data, config)
    
    return data_module

if __name__ == "__main__":
    # Test the fixed dataset
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_module = create_fixed_dataset_from_merged_data('data/merged.csv', config)
    train_loader, val_loader, test_loader = data_module.get_data_loaders()
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for batch in train_loader:
        logger.info(f"Batch shapes:")
        logger.info(f"  past_target: {batch['past_target'].shape}")
        logger.info(f"  observed_covariates: {batch['observed_covariates'].shape}")
        logger.info(f"  known_future: {batch['known_future'].shape}")
        logger.info(f"  target: {batch['target'].shape}")
        break
