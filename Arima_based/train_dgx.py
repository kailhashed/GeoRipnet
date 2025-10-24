#!/usr/bin/env python3
"""
RippleNet-TFT Training Script for NVIDIA DGX A100
Optimized for multi-GPU training with mixed precision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import time
from datetime import datetime
import json
import wandb

# Import project modules
from data.dataset import create_dataset_from_merged_data
from model.ripple_tft import create_model
from train import train_epoch, validate_epoch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DGXTrainingConfig:
    """Configuration for DGX training"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # DGX-specific optimizations
        self.batch_size = 128  # Larger batch size for DGX
        self.learning_rate = 0.0005  # Higher LR for faster convergence
        self.epochs = 300  # More epochs for better performance
        self.mixed_precision = True
        self.gradient_accumulation_steps = 4
        self.warmup_epochs = 10
        
        # Multi-GPU settings
        self.world_size = torch.cuda.device_count()
        self.distributed = self.world_size > 1
        
        logger.info(f"DGX Configuration: {self.world_size} GPUs, batch_size={self.batch_size}")

def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_dgx_worker(rank, world_size, config_path, data_path):
    """Worker function for distributed training"""
    
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Load configuration
    dgx_config = DGXTrainingConfig(config_path)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    # Initialize wandb (only on rank 0)
    if rank == 0:
        wandb.init(
            project="ripplenet-tft-dgx",
            config=dgx_config.config,
            name=f"ripplenet-tft-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Load data
    logger.info(f"Loading data on rank {rank}...")
    data_module = create_dataset_from_merged_data(data_path, dgx_config.config)
    
    # Create data loaders with distributed sampling
    train_loader, val_loader, test_loader = data_module.get_dataloaders(
        batch_size=dgx_config.batch_size,
        num_workers=8,  # More workers for DGX
        pin_memory=True
    )
    
    if dgx_config.distributed:
        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank)
        
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=dgx_config.batch_size,
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=dgx_config.batch_size,
            sampler=val_sampler,
            num_workers=8,
            pin_memory=True
        )
    
    # Create model
    logger.info(f"Creating model on rank {rank}...")
    model = create_model(dgx_config.config)
    model = model.to(device)
    
    # Wrap model for distributed training
    if dgx_config.distributed:
        model = DDP(model, device_ids=[rank])
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=dgx_config.learning_rate,
        weight_decay=0.0001
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=dgx_config.learning_rate * 10,
        epochs=dgx_config.epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if dgx_config.mixed_precision else None
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20
    
    logger.info(f"Starting training on rank {rank}...")
    start_time = time.time()
    
    for epoch in range(dgx_config.epochs):
        # Set epoch for distributed sampler
        if dgx_config.distributed:
            train_sampler.set_epoch(epoch)
        
        # Training
        train_loss = train_epoch_dgx(
            model, train_loader, optimizer, device, 
            scaler, dgx_config.gradient_accumulation_steps
        )
        
        # Validation
        val_loss = validate_epoch_dgx(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Logging (only on rank 0)
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch+1}/{dgx_config.epochs}:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_loss:.6f}")
            logger.info(f"  LR: {current_lr:.2e}")
            
            # Wandb logging
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if dgx_config.distributed else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'config': dgx_config.config,
                    'scalers': data_module.get_scalers()
                }
                
                torch.save(checkpoint, 'checkpoints/best_model_dgx.pt')
                logger.info(f"Saved best model with val_loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    if rank == 0:
        logger.info("Running final evaluation...")
        test_loss = validate_epoch_dgx(model, test_loader, device)
        logger.info(f"Final test loss: {test_loss:.6f}")
        
        # Save final results
        results = {
            'best_val_loss': best_val_loss,
            'final_test_loss': test_loss,
            'total_epochs': epoch + 1,
            'training_time': time.time() - start_time,
            'gpu_count': world_size,
            'batch_size': dgx_config.batch_size
        }
        
        with open('results/dgx_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
    
    # Cleanup
    if dgx_config.distributed:
        cleanup_distributed()

def train_epoch_dgx(model, train_loader, optimizer, device, scaler, gradient_accumulation_steps):
    """Training epoch optimized for DGX"""
    model.train()
    total_loss = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        past_target = batch['past_target'].to(device)
        observed_cov = batch['observed_covariates'].to(device)
        known_future = batch['known_future'].to(device)
        target = batch['target'].to(device)
        
        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                prediction = model(past_target, observed_cov, known_future)
                loss = nn.MSELoss()(prediction, target)
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            prediction = model(past_target, observed_cov, known_future)
            loss = nn.MSELoss()(prediction, target)
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
    
    return total_loss / len(train_loader)

def validate_epoch_dgx(model, val_loader, device):
    """Validation epoch optimized for DGX"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            past_target = batch['past_target'].to(device)
            observed_cov = batch['observed_covariates'].to(device)
            known_future = batch['known_future'].to(device)
            target = batch['target'].to(device)
            
            prediction = model(past_target, observed_cov, known_future)
            loss = nn.MSELoss()(prediction, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RippleNet-TFT DGX Training')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--data', default='data/merged.csv', help='Data file path')
    parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs')
    
    args = parser.parse_args()
    
    # Determine world size
    world_size = args.world_size or torch.cuda.device_count()
    
    if world_size > 1:
        logger.info(f"Starting distributed training with {world_size} GPUs")
        mp.spawn(train_dgx_worker, args=(world_size, args.config, args.data), nprocs=world_size, join=True)
    else:
        logger.info("Starting single GPU training")
        train_dgx_worker(0, 1, args.config, args.data)

if __name__ == "__main__":
    import os
    main()
