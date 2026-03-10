#!/usr/bin/env python3
"""
SICKLE++ Baseline Training Script

Trains baseline models for:
- Crop classification
- Phenology prediction
- Yield prediction
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import pipeline components
from pipeline.datasets import SICKLEDataset, create_dataloaders
from pipeline.training import SICKLETrainer, TrainingConfig
from pipeline.evaluation import EvaluationPipeline
from pipeline.models.inference import ModelInference

# Import models
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
try:
    from utae import UTAE
    from unet3d import UNet3D
    from unet3d_multitask import UNet3D as UNet3DMultitask
    from pastis_unet3d import PaSTiSUNet3D
except ImportError:
    print("⚠️ Warning: Some models could not be imported")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model(model_type: str,
                input_channels: int = 21,
                num_classes: int = 20,
                **kwargs) -> nn.Module:
    """
    Create model architecture.
    
    Args:
        model_type: 'utae', 'unet3d', 'unet3d_multitask', or 'pastis'
        input_channels: Number of input channels
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    logger.info(f"🏗️ Creating model: {model_type}")
    logger.info(f"   Input channels: {input_channels}")
    logger.info(f"   Output classes: {num_classes}")
    
    if model_type == 'utae':
        model = UTAE(
            input_dim=input_channels,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, num_classes],
            n_head=16,
            d_model=256,
            **kwargs
        )
    
    elif model_type == 'unet3d':
        model = UNet3D(
            in_channel=input_channels,
            n_classes=num_classes,
            timesteps=61,
            dropout=0.2,
            **kwargs
        )
    
    elif model_type == 'unet3d_multitask':
        model = UNet3DMultitask(
            in_channel=input_channels,
            n_classes=num_classes,
            timesteps=61,
            dropout=0.2,
            **kwargs
        )
    
    elif model_type == 'pastis':
        model = PaSTiSUNet3D(
            input_dim=input_channels,
            num_classes=num_classes,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Model parameters: {n_params:,}")
    
    return model


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description='Train SICKLE++ baseline models'
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--labels_path', type=str, required=True,
                       help='Path to labels file')
    parser.add_argument('--task', type=str, default='crop_classification',
                       choices=['crop_classification', 'phenology', 'yield'],
                       help='Task type')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='utae',
                       choices=['utae', 'unet3d', 'unet3d_multitask', 'pastis'],
                       help='Model architecture')
    parser.add_argument('--input_channels', type=int, default=21,
                       help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=20,
                       help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Training setup
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory')
    parser.add_argument('--save_config', action='store_true',
                       help='Save training config')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"🎯 Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset
        logger.info(f"📂 Loading dataset from {args.data_dir}")
        
        dataset = SICKLEDataset(
            satellite_data_dir=args.data_dir,
            labels_path=args.labels_path,
            task_type=args.task,
            normalize=True,
            cache_in_memory=False
        )
        
        # Create data loaders
        logger.info(f"🔄 Creating data loaders")
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed
        )
        
        # Create model
        model = create_model(
            args.model,
            input_channels=args.input_channels,
            num_classes=args.num_classes
        )
        
        # Create training config
        config = TrainingConfig(
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            patience=args.patience,
            task_type=args.task
        )
        
        # Save config
        if args.save_config:
            config_path = output_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            logger.info(f"✅ Config saved: {config_path}")
        
        # Create trainer
        trainer = SICKLETrainer(
            model=model,
            device=device,
            config=config,
            output_dir=output_dir
        )
        
        # Train
        logger.info(f"🚀 Starting training...")
        trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        logger.info(f"📊 Evaluating on test set...")
        evaluator = EvaluationPipeline(
            output_dir=output_dir / 'test_results',
            verbose=True
        )
        
        # Load best model
        trainer.model = create_model(
            args.model,
            input_channels=args.input_channels,
            num_classes=args.num_classes
        )
        trainer.load_checkpoint(trainer.best_model_path)
        
        # Get test predictions
        trainer.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                targets = batch['label'].to(device)
                
                outputs = trainer.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if args.task == 'crop_classification':
                    preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                else:
                    preds = outputs
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Evaluate
        if args.task == 'crop_classification':
            metrics = evaluator.evaluate_crop_classification(all_targets, all_preds)
        elif args.task == 'yield':
            metrics = evaluator.evaluate_yield_prediction(all_targets, all_preds)
        else:
            metrics = evaluator.evaluate_phenology_prediction(all_targets, all_preds)
        
        logger.info(f"✅ Training and evaluation complete!")
        logger.info(f"   Results: {output_dir / 'test_results'}")
        
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        raise


if __name__ == '__main__':
    import numpy as np
    main()
