"""
GeoRipNet Advanced - Maximum-Accuracy Deep Learning Framework

A production-ready PyTorch implementation for country-level oil price prediction
with ripple effect propagation through graph neural networks.

Author: GeoRipNet Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GeoRipNet Research Team"

# Import key components for easy access
from .models import (
    BenchmarkModel,
    LocalDeltaModel,
    RippleGNNLayer,
    GeoRipNetModel,
    EnsembleGeoRipNet
)

from .losses import MultiObjectiveLoss

from .data import (
    GeoRipNetDataset,
    DataPreprocessor,
    create_dataloaders
)

from .training import GeoRipNetTrainer

from .utils import (
    MetricsCalculator,
    GeoRipNetVisualizer
)

__all__ = [
    # Models
    'BenchmarkModel',
    'LocalDeltaModel',
    'RippleGNNLayer',
    'GeoRipNetModel',
    'EnsembleGeoRipNet',
    
    # Loss
    'MultiObjectiveLoss',
    
    # Data
    'GeoRipNetDataset',
    'DataPreprocessor',
    'create_dataloaders',
    
    # Training
    'GeoRipNetTrainer',
    
    # Utils
    'MetricsCalculator',
    'GeoRipNetVisualizer',
]

