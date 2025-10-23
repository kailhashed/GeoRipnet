"""
Training Script for RippleNet-TFT
Multi-GPU training with mixed precision and comprehensive logging
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import argparse
import json
from typing import Dict, List, Optional, Tuple
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RippleNetTrainer:
    """Training class for RippleNet-TFT"""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize model
        self.model = create_model(config)
        self.model.to(device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config['training']['use_mixed_precision'] else None
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Create output directories
        self._create_directories()
        
        # Initialize data module
        self.data_module = None
        
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config['training']['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        if self.config['training']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_name = self.config['training']['optimizer']
        learning_rate = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Created {optimizer_name} optimizer with lr={learning_rate}")
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config['training']['scheduler']
        max_lr = self.config['training']['max_lr']
        epochs = self.config['training']['epochs']
        
        if scheduler_name.lower() == 'onecyclelr':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                total_steps=epochs,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif scheduler_name.lower() == 'cosineannealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        else:
            scheduler = None
        
        logger.info(f"Created {scheduler_name} scheduler")
        return scheduler
    
    def _create_loss_function(self):
        """Create loss function"""
        loss_type = self.config['training'].get('loss_function', 'mse')
        
        if loss_type.lower() == 'mse':
            criterion = nn.MSELoss()
        elif loss_type.lower() == 'huber':
            criterion = nn.HuberLoss()
        elif loss_type.lower() == 'mae':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        
        logger.info(f"Created {loss_type} loss function")
        return criterion
    
    def _create_directories(self):
        """Create output directories"""
        self.checkpoint_dir = Path(self.config['paths']['model_checkpoints'])
        self.results_dir = Path(self.config['paths']['results'])
        self.logs_dir = Path(self.config['paths']['logs'])
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, merged_data_path: str):
        """Load and prepare data"""
        logger.info(f"Loading data from {merged_data_path}")
        
        self.data_module = create_dataset_from_merged_data(merged_data_path, self.config)
        
        # Get data loaders
        self.train_loader, self.val_loader, self.test_loader = self.data_module.get_data_loaders()
        
        logger.info(f"Data loaded successfully")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
        logger.info(f"Test batches: {len(self.test_loader)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            past_target = batch['past_target'].to(self.device)
            observed_covariates = batch['observed_covariates'].to(self.device)
            known_future = batch['known_future'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(past_target, observed_covariates, known_future)
                    loss = self.criterion(predictions, target)
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(past_target, observed_covariates, known_future)
                loss = self.criterion(predictions, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}")
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                past_target = batch['past_target'].to(self.device)
                observed_covariates = batch['observed_covariates'].to(self.device)
                known_future = batch['known_future'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        predictions = self.model(past_target, observed_covariates, known_future)
                        loss = self.criterion(predictions, target)
                else:
                    predictions = self.model(past_target, observed_covariates, known_future)
                    loss = self.criterion(predictions, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self, merged_data_path: str):
        """Main training loop"""
        logger.info("Starting training")
        
        # Load data
        self.load_data(merged_data_path)
        
        # Training loop
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.training_history.append(epoch_metrics)
            
            # Log metrics
            logger.info(f"Epoch {epoch}: {epoch_metrics}")
            
            # Check for improvement
            val_loss = val_metrics['val_loss']
            is_best = val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config['training']['save_frequency'] == 0:
                self.save_checkpoint()
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        self.save_checkpoint()
        
        # Save training history
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("Training completed successfully!")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train RippleNet-TFT')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--data', type=str, default='data/merged.csv', help='Merged data file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = RippleNetTrainer(config, device=args.device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(args.data)

if __name__ == "__main__":
    main()
