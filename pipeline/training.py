"""
Training pipeline for SICKLE++ baseline models.

Implements:
- Standard supervised training loop
- Multi-task training (crop classification + phenology)
- Validation and early stopping
- Checkpoint management
- Logging and visualization
"""

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union
import json
import logging
from datetime import datetime
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for model training."""
    
    def __init__(self,
                 max_epochs: int = 50,
                 batch_size: int = 8,
                 learning_rate: float = 1e-4,
                 optimizer: str = 'adamw',
                 weight_decay: float = 1e-5,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 gradient_clip: float = 1.0,
                 warmup_epochs: int = 0,
                 patience: int = 10,
                 mixed_precision: bool = False,
                 task_type: str = 'crop_classification'):
        """
        Initialize training configuration.
        
        Args:
            max_epochs: Maximum training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            optimizer: Optimizer type ('adamw', 'sgd', 'adam')
            weight_decay: Weight decay / L2 regularization
            num_workers: DataLoader workers
            pin_memory: Pin memory for GPU
            gradient_clip: Gradient clipping value
            warmup_epochs: Number of warmup epochs
            patience: Early stopping patience
            mixed_precision: Use float16 precision
            task_type: Task type ('crop_classification', 'phenology', 'yield')
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.gradient_clip = gradient_clip
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.mixed_precision = mixed_precision
        self.task_type = task_type
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.__dict__


class TrainingLogger:
    """Log training metrics to file and console."""
    
    def __init__(self, log_dir: Path):
        """Initialize logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_metrics: Optional[Dict] = None,
                  val_metrics: Optional[Dict] = None,
                  learning_rate: float = 0):
        """Log epoch results."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(learning_rate)
        
        if train_metrics:
            self.train_metrics.append(train_metrics)
        if val_metrics:
            self.val_metrics.append(val_metrics)
        
        # Track best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
        
        # Print progress
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {learning_rate:.6f}")
    
    def save(self):
        """Save logs to file."""
        logs = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        with open(self.log_dir / 'training_logs.json', 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"✅ Logs saved: {self.log_dir / 'training_logs.json'}")


class SICKLETrainer:
    """
    SICKLE++ baseline model trainer.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: Union[str, torch.device] = 'cuda',
                 config: Optional[TrainingConfig] = None,
                 output_dir: Optional[Path] = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on
            config: Training configuration
            output_dir: Directory to save checkpoints and logs
        """
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir) if output_dir else Path('checkpoints')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize logger
        self.logger = TrainingLogger(self.output_dir / 'logs')
        
        # Best model path
        self.best_model_path = self.output_dir / 'best_model.pth'
        self.latest_model_path = self.output_dir / 'latest_model.pth'
        
        logger.info(f"🔧 Trainer initialized on device: {self.device}")
        logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"   Task: {self.config.task_type}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer == 'adamw':
            return AdamW(params,
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay,
                        betas=(0.9, 0.999))
        
        elif self.config.optimizer == 'adam':
            return Adam(params,
                       lr=self.config.learning_rate,
                       weight_decay=self.config.weight_decay)
        
        elif self.config.optimizer == 'sgd':
            return SGD(params,
                      lr=self.config.learning_rate,
                      momentum=0.9,
                      weight_decay=self.config.weight_decay)
        
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on task type."""
        if self.config.task_type == 'crop_classification':
            return nn.CrossEntropyLoss()
        elif self.config.task_type == 'yield':
            return nn.MSELoss()
        elif self.config.task_type == 'phenology':
            return nn.L1Loss()  # MAE for day prediction
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   criterion: nn.Module) -> Tuple[float, Dict]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            
        Returns:
            Average loss and metrics dict
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch in pbar:
            # Get data
            images = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            if isinstance(outputs, tuple):
                # Multi-task output
                loss = criterion(outputs[0], targets)
            else:
                loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            n_batches += 1
            
            # Store predictions for metrics
            if self.config.task_type == 'crop_classification':
                with torch.no_grad():
                    if isinstance(outputs, tuple):
                        probs = torch.softmax(outputs[0], dim=1)
                    else:
                        probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        
        # Compute metrics
        metrics = {}
        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            from sklearn.metrics import accuracy_score, f1_score
            metrics['accuracy'] = accuracy_score(all_targets, all_preds)
            metrics['f1'] = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self,
                val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, Dict]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Average loss and metrics dict
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        for batch in pbar:
            # Get data
            images = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            if isinstance(outputs, tuple):
                loss = criterion(outputs[0], targets)
            else:
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Store predictions
            if self.config.task_type == 'crop_classification':
                if isinstance(outputs, tuple):
                    probs = torch.softmax(outputs[0], dim=1)
                else:
                    probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        
        # Compute metrics
        metrics = {}
        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            from sklearn.metrics import accuracy_score, f1_score
            metrics['accuracy'] = accuracy_score(all_targets, all_preds)
            metrics['f1'] = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        return avg_loss, metrics
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: Optional[int] = None):
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config.max_epochs if None)
        """
        num_epochs = num_epochs or self.config.max_epochs
        criterion = self._create_criterion()
        
        patience_counter = 0
        epochs_since_improvement = 0
        
        logger.info(f"🚀 Starting training for {num_epochs} epochs")
        logger.info(f"   Optimizer: {self.config.optimizer}")
        logger.info(f"   LR: {self.config.learning_rate}")
        logger.info(f"   Batch size: {self.config.batch_size}")
        
        for epoch in range(num_epochs):
            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, criterion)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader, criterion)
            
            # Log
            self.logger.log_epoch(epoch, train_loss, val_loss,
                                 train_metrics, val_metrics, current_lr)
            
            # Save checkpoint
            self._save_checkpoint(epoch, train_loss, val_loss, is_best=(val_loss < self.logger.best_val_loss))
            
            # Early stopping
            if val_loss < self.logger.best_val_loss:
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                logger.info(f"⏹️ Early stopping at epoch {epoch} (patience={self.config.patience})")
                break
        
        # Save final logs
        self.logger.save()
        
        logger.info(f"✅ Training complete. Best loss: {self.logger.best_val_loss:.4f} (epoch {self.logger.best_epoch})")
        logger.info(f"   Best model: {self.best_model_path}")
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Always save latest
        torch.save(checkpoint, self.latest_model_path)
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            logger.info(f"🏆 New best model: {self.best_model_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"✅ Checkpoint loaded: {checkpoint_path}")


def train_baseline_model(model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        config: Optional[TrainingConfig] = None,
                        output_dir: Optional[Path] = None):
    """
    Train a baseline SICKLE++ model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        output_dir: Output directory
    """
    trainer = SICKLETrainer(model, config=config, output_dir=output_dir)
    trainer.train(train_loader, val_loader)
    
    return trainer
