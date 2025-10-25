"""
Advanced Training Loop for GeoRipNet.

Implements state-of-the-art training techniques:
- Mixed precision training (torch.cuda.amp)
- Stochastic Weight Averaging (SWA)
- Early stopping with patience
- Gradient clipping
- Learning rate scheduling (OneCycleLR, CosineAnnealing)
- Checkpointing and resuming
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable
import time
import json
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping (default: 15)
        min_delta: Minimum change to qualify as improvement (default: 1e-4)
        mode: 'min' or 'max' (default: 'min' for loss)
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            epoch: Current epoch number
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class GeoRipNetTrainer:
    """
    Advanced trainer for GeoRipNet models.
    
    Implements complete training pipeline with all modern techniques.
    
    Args:
        model: GeoRipNetModel instance
        criterion: Loss function
        optimizer: Optimizer (default: AdamW)
        device: Device to train on
        use_amp: Use automatic mixed precision (default: True)
        use_swa: Use Stochastic Weight Averaging (default: True)
        gradient_clip_norm: Max gradient norm for clipping (default: 1.0)
        swa_start_epoch: Epoch to start SWA (default: 10)
        checkpoint_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda',
        use_amp: bool = True,
        use_swa: bool = True,
        gradient_clip_norm: float = 1.0,
        swa_start_epoch: int = 10,
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_swa = use_swa
        self.gradient_clip_norm = gradient_clip_norm
        self.swa_start_epoch = swa_start_epoch
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Default optimizer if none provided
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=5e-5,
                weight_decay=1e-5,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optimizer
        
        # Mixed precision scaler
        if self.use_amp:
            self.scaler = GradScaler()
        
        # SWA model
        if self.use_swa:
            self.swa_model = AveragedModel(model)
            self.swa_scheduler = None  # Will be set during training
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(
        self,
        train_loader,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            scheduler: Learning rate scheduler (optional)
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {
            'huber_loss': 0.0,
            'directional_loss': 0.0,
            'correlation_loss': 0.0,
            'directional_accuracy': 0.0
        }
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        batch['benchmark_features'],
                        batch['local_features'],
                        batch['country_ids'],
                        batch['event_embeddings'],
                        batch['trade_adjacency'],
                        batch['trade_weights'],
                        return_components=True
                    )
                    
                    # Compute loss
                    losses = self.criterion(
                        predictions=outputs['predictions'],
                        targets=batch['targets'],
                        prev_targets=batch['prev_targets'],
                        predicted_deltas=outputs.get('propagated_deltas'),
                        true_deltas=batch['targets'] - batch['prev_targets']
                    )
                    
                    loss = losses['total_loss']
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision
                outputs = self.model(
                    batch['benchmark_features'],
                    batch['local_features'],
                    batch['country_ids'],
                    batch['event_embeddings'],
                    batch['trade_adjacency'],
                    batch['trade_weights'],
                    return_components=True
                )
                
                losses = self.criterion(
                    predictions=outputs['predictions'],
                    targets=batch['targets'],
                    prev_targets=batch['prev_targets'],
                    predicted_deltas=outputs.get('propagated_deltas'),
                    true_deltas=batch['targets'] - batch['prev_targets']
                )
                
                loss = losses['total_loss']
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm
                )
                
                self.optimizer.step()
            
            # Update scheduler (if per-batch scheduler like OneCycleLR)
            if scheduler is not None and hasattr(scheduler, 'step_update'):
                scheduler.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            for key in epoch_metrics.keys():
                if key in losses:
                    epoch_metrics[key] += losses[key].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'dir_acc': losses.get('directional_accuracy', 0.0).item()
            })
        
        # Average metrics
        epoch_loss /= num_batches
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches
        
        epoch_metrics['total_loss'] = epoch_loss
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = {
            'huber_loss': 0.0,
            'directional_loss': 0.0,
            'correlation_loss': 0.0,
            'directional_accuracy': 0.0
        }
        num_batches = 0
        
        for batch in tqdm(val_loader, desc='Validation'):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        batch['benchmark_features'],
                        batch['local_features'],
                        batch['country_ids'],
                        batch['event_embeddings'],
                        batch['trade_adjacency'],
                        batch['trade_weights'],
                        return_components=True
                    )
                    
                    losses = self.criterion(
                        predictions=outputs['predictions'],
                        targets=batch['targets'],
                        prev_targets=batch['prev_targets'],
                        predicted_deltas=outputs.get('propagated_deltas'),
                        true_deltas=batch['targets'] - batch['prev_targets']
                    )
            else:
                outputs = self.model(
                    batch['benchmark_features'],
                    batch['local_features'],
                    batch['country_ids'],
                    batch['event_embeddings'],
                    batch['trade_adjacency'],
                    batch['trade_weights'],
                    return_components=True
                )
                
                losses = self.criterion(
                    predictions=outputs['predictions'],
                    targets=batch['targets'],
                    prev_targets=batch['prev_targets'],
                    predicted_deltas=outputs.get('propagated_deltas'),
                    true_deltas=batch['targets'] - batch['prev_targets']
                )
            
            val_loss += losses['total_loss'].item()
            for key in val_metrics.keys():
                if key in losses:
                    val_metrics[key] += losses[key].item()
            num_batches += 1
        
        # Average metrics
        val_loss /= num_batches
        for key in val_metrics.keys():
            val_metrics[key] /= num_batches
        
        val_metrics['total_loss'] = val_loss
        
        return val_metrics
    
    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        scheduler_type: str = 'onecycle',
        early_stopping_patience: int = 15,
        save_best_only: bool = True,
        verbose: bool = True
    ):
        """
        Complete training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs to train
            scheduler_type: 'onecycle', 'cosine', or None
            early_stopping_patience: Patience for early stopping
            save_best_only: Only save best model (vs. save every epoch)
            verbose: Print training progress
        """
        # Initialize scheduler
        if scheduler_type == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=1e-3,
                epochs=num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-7
            )
        else:
            scheduler = None
        
        # Initialize SWA scheduler
        if self.use_swa and self.swa_start_epoch < num_epochs:
            self.swa_scheduler = SWALR(
                self.optimizer,
                swa_lr=5e-5,
                anneal_epochs=5
            )
        
        # Early stopping
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-4,
            mode='min'
        )
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"SWA: {self.use_swa} (starts at epoch {self.swa_start_epoch})")
        print(f"Gradient Clipping: {self.gradient_clip_norm}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, scheduler)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_times'].append(epoch_time)
            
            # Print progress
            if verbose:
                print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['total_loss']:.4f} | "
                      f"Val Loss: {val_metrics['total_loss']:.4f}")
                print(f"  Train Dir Acc: {train_metrics['directional_accuracy']:.3f} | "
                      f"Val Dir Acc: {val_metrics['directional_accuracy']:.3f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Update SWA
            if self.use_swa and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                if self.swa_scheduler is not None:
                    self.swa_scheduler.step()
            
            # Scheduler step (for epoch-based schedulers)
            if scheduler is not None and scheduler_type == 'cosine':
                scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
            
            if is_best or not save_best_only:
                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_metrics['total_loss'],
                    is_best=is_best
                )
            
            # Early stopping
            if early_stopper(val_metrics['total_loss'], epoch):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {early_stopper.best_score:.4f} "
                      f"at epoch {early_stopper.best_epoch+1}")
                break
        
        # Finalize SWA
        if self.use_swa and self.swa_start_epoch < num_epochs:
            print("\nFinalizing Stochastic Weight Averaging...")
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=self.device)
            self.save_swa_model()
        
        # Save training history
        self.save_history()
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'checkpoint_best.pth')
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
    
    def save_swa_model(self):
        """Save SWA model."""
        torch.save(
            self.swa_model.state_dict(),
            self.checkpoint_dir / 'swa_model.pth'
        )
        print("  → Saved SWA model")
    
    def save_history(self):
        """Save training history."""
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        self.history = checkpoint['history']
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


if __name__ == "__main__":
    print("GeoRipNet Trainer module loaded successfully!")
    print("\nKey features:")
    print("  ✓ Mixed precision training (AMP)")
    print("  ✓ Stochastic Weight Averaging (SWA)")
    print("  ✓ Early stopping")
    print("  ✓ Gradient clipping")
    print("  ✓ Advanced LR scheduling")
    print("  ✓ Checkpointing and resuming")

