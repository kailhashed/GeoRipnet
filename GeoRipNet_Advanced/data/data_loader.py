"""
Data Loading and Batching for GeoRipNet.

Handles time-series data loading, batching, and preprocessing for
country-level oil price prediction with ripple effects.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class GeoRipNetDataset(Dataset):
    """
    PyTorch Dataset for GeoRipNet training.
    
    Loads and serves time-series data for benchmark features, country-specific
    features, event embeddings, and trade relationships.
    
    Args:
        data_dict: Dictionary containing all required data arrays
        seq_len: Input sequence length (default: 30)
        pred_horizon: Prediction horizon (default: 1)
        mode: 'train', 'val', or 'test'
    """
    
    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        seq_len: int = 30,
        pred_horizon: int = 1,
        mode: str = 'train'
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.mode = mode
        
        # Unpack data
        self.benchmark_features = data_dict['benchmark_features']  # (T, benchmark_feat_dim)
        self.local_features = data_dict['local_features']  # (T, num_countries, local_feat_dim)
        self.prices = data_dict['prices']  # (T, num_countries)
        self.event_embeddings = data_dict['event_embeddings']  # (T, num_countries, embed_dim)
        self.trade_adjacency = data_dict['trade_adjacency']  # (num_countries, num_countries) or (T, num_countries, num_countries)
        self.trade_weights = data_dict['trade_weights']  # (num_countries, num_benchmarks)
        
        self.num_timesteps = len(self.benchmark_features)
        self.num_countries = self.prices.shape[1]
        
        # Country IDs (0 to num_countries-1)
        self.country_ids = np.arange(self.num_countries)
        
        # Compute valid indices (need seq_len history + pred_horizon future)
        self.valid_indices = list(range(
            seq_len,
            self.num_timesteps - pred_horizon
        ))
        
        print(f"Dataset initialized: {len(self.valid_indices)} samples, {self.num_countries} countries")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Returns:
            Dictionary containing:
                - benchmark_features: (seq_len, benchmark_feat_dim)
                - local_features: (num_countries, seq_len, local_feat_dim)
                - event_embeddings: (num_countries, embed_dim)
                - trade_adjacency: (num_countries, num_countries)
                - trade_weights: (num_countries, num_benchmarks)
                - country_ids: (num_countries,)
                - targets: (num_countries,) - Prices at t+pred_horizon
                - prev_targets: (num_countries,) - Prices at t
        """
        t = self.valid_indices[idx]
        
        # Historical sequences: [t-seq_len+1 : t+1]
        benchmark_seq = self.benchmark_features[t-self.seq_len+1 : t+1]  # (seq_len, feat_dim)
        local_seq = self.local_features[t-self.seq_len+1 : t+1]  # (seq_len, num_countries, feat_dim)
        
        # Transpose local_seq: (num_countries, seq_len, feat_dim)
        local_seq = np.transpose(local_seq, (1, 0, 2))
        
        # Current event embeddings (at time t)
        events = self.event_embeddings[t]  # (num_countries, embed_dim)
        
        # Trade adjacency (static or time-varying)
        if self.trade_adjacency.ndim == 2:
            trade_adj = self.trade_adjacency  # Static
        else:
            trade_adj = self.trade_adjacency[t]  # Time-varying
        
        # Targets
        targets = self.prices[t + self.pred_horizon]  # (num_countries,)
        prev_targets = self.prices[t]  # (num_countries,)
        
        # Convert to tensors
        sample = {
            'benchmark_features': torch.FloatTensor(benchmark_seq),
            'local_features': torch.FloatTensor(local_seq),
            'event_embeddings': torch.FloatTensor(events),
            'trade_adjacency': torch.FloatTensor(trade_adj),
            'trade_weights': torch.FloatTensor(self.trade_weights),
            'country_ids': torch.LongTensor(self.country_ids),
            'targets': torch.FloatTensor(targets),
            'prev_targets': torch.FloatTensor(prev_targets),
            'timestamp': t
        }
        
        return sample


class TimeSeriesDataSplitter:
    """
    Time-series aware data splitting (no shuffle, chronological order).
    
    Splits data into train/val/test maintaining temporal order.
    """
    
    @staticmethod
    def split_data(
        data_dict: Dict[str, np.ndarray],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Split data chronologically.
        
        Args:
            data_dict: Full dataset dictionary
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
        
        Returns:
            train_dict, val_dict, test_dict
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        T = len(data_dict['benchmark_features'])
        
        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))
        
        def split_dict(d, start, end):
            """Helper to split dictionary arrays."""
            result = {}
            for key, value in d.items():
                if key in ['trade_weights']:
                    # Static data, no split
                    result[key] = value
                elif key == 'trade_adjacency' and value.ndim == 2:
                    # Static adjacency, no split
                    result[key] = value
                else:
                    # Time-varying data, split along time dimension
                    result[key] = value[start:end]
            return result
        
        train_dict = split_dict(data_dict, 0, train_end)
        val_dict = split_dict(data_dict, train_end, val_end)
        test_dict = split_dict(data_dict, val_end, T)
        
        return train_dict, val_dict, test_dict


def create_dataloaders(
    train_dict: Dict[str, np.ndarray],
    val_dict: Dict[str, np.ndarray],
    test_dict: Dict[str, np.ndarray],
    seq_len: int = 30,
    pred_horizon: int = 1,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test.
    
    Args:
        train_dict, val_dict, test_dict: Data dictionaries
        seq_len: Sequence length
        pred_horizon: Prediction horizon
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = GeoRipNetDataset(train_dict, seq_len, pred_horizon, mode='train')
    val_dataset = GeoRipNetDataset(val_dict, seq_len, pred_horizon, mode='val')
    test_dataset = GeoRipNetDataset(test_dict, seq_len, pred_horizon, mode='test')
    
    # Create dataloaders (no shuffle for time series!)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: preserve temporal order
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class RollingWindowCV:
    """
    Rolling window cross-validation for time series.
    
    Creates multiple train/val splits by sliding a window through time.
    Useful for robust model evaluation.
    """
    
    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        n_splits: int = 5,
        train_size: int = 365,  # Days
        val_size: int = 90,
        gap: int = 0  # Gap between train and val to prevent leakage
    ):
        self.data_dict = data_dict
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.gap = gap
        
        self.T = len(data_dict['benchmark_features'])
        
    def split(self):
        """
        Generate train/val indices for each fold.
        
        Yields:
            train_dict, val_dict for each fold
        """
        max_train_start = self.T - self.train_size - self.gap - self.val_size
        step = max_train_start // (self.n_splits - 1) if self.n_splits > 1 else max_train_start
        
        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + self.train_size
            val_start = train_end + self.gap
            val_end = val_start + self.val_size
            
            if val_end > self.T:
                break
            
            def extract_dict(start, end):
                result = {}
                for key, value in self.data_dict.items():
                    if key in ['trade_weights']:
                        result[key] = value
                    elif key == 'trade_adjacency' and value.ndim == 2:
                        result[key] = value
                    else:
                        result[key] = value[start:end]
                return result
            
            train_dict = extract_dict(train_start, train_end)
            val_dict = extract_dict(val_start, val_end)
            
            yield train_dict, val_dict


def collate_batch(batch):
    """
    Custom collate function for batching samples.
    
    Handles variable-length sequences and special data structures.
    """
    # Stack all tensors in the batch
    collated = {}
    
    for key in batch[0].keys():
        if key == 'timestamp':
            # Keep as list
            collated[key] = [sample[key] for sample in batch]
        else:
            # Stack tensors
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
    
    return collated


if __name__ == "__main__":
    # Test data loading
    print("Testing GeoRipNet Data Loading...")
    
    # Create dummy data
    T = 1000  # Time steps
    num_countries = 20
    num_benchmarks = 3
    benchmark_feat_dim = 50
    local_feat_dim = 40
    embed_dim = 384
    
    data_dict = {
        'benchmark_features': np.random.randn(T, benchmark_feat_dim),
        'local_features': np.random.randn(T, num_countries, local_feat_dim),
        'prices': np.random.randn(T, num_countries) * 10 + 80,  # Around $80
        'event_embeddings': np.random.randn(T, num_countries, embed_dim),
        'trade_adjacency': np.random.rand(num_countries, num_countries),
        'trade_weights': np.random.rand(num_countries, num_benchmarks)
    }
    
    # Normalize trade weights
    data_dict['trade_weights'] = data_dict['trade_weights'] / data_dict['trade_weights'].sum(axis=1, keepdims=True)
    
    # Test dataset
    print("\n1. Testing GeoRipNetDataset...")
    dataset = GeoRipNetDataset(data_dict, seq_len=30, pred_horizon=1)
    print(f"   Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"   Sample keys: {sample.keys()}")
    print(f"   Benchmark features shape: {sample['benchmark_features'].shape}")
    print(f"   Local features shape: {sample['local_features'].shape}")
    print(f"   Event embeddings shape: {sample['event_embeddings'].shape}")
    print(f"   Targets shape: {sample['targets'].shape}")
    
    # Test data splitting
    print("\n2. Testing TimeSeriesDataSplitter...")
    train_dict, val_dict, test_dict = TimeSeriesDataSplitter.split_data(
        data_dict, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    print(f"   Train samples: {len(train_dict['prices'])}")
    print(f"   Val samples: {len(val_dict['prices'])}")
    print(f"   Test samples: {len(test_dict['prices'])}")
    
    # Test dataloaders
    print("\n3. Testing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dict, val_dict, test_dict,
        seq_len=30, batch_size=16, num_workers=0
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\n   Batch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"   - {key}: {value.shape}")
    
    # Test rolling window CV
    print("\n4. Testing RollingWindowCV...")
    cv = RollingWindowCV(data_dict, n_splits=3, train_size=300, val_size=100)
    
    for fold_idx, (train_fold, val_fold) in enumerate(cv.split()):
        print(f"   Fold {fold_idx + 1}:")
        print(f"     Train: {len(train_fold['prices'])} samples")
        print(f"     Val: {len(val_fold['prices'])} samples")
    
    print("\n✓ Data loading tests passed!")

