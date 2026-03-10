#!/usr/bin/env python3
"""
SICKLE++ Baseline Demo & Example Usage

Demonstrates:
1. Creating synthetic SICKLE-compatible data
2. Training a baseline model
3. Running inference
4. Evaluating results
5. Analyzing agricultural outputs
"""

import numpy as np
import json
import logging
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.datasets import SICKLEDataset, create_dataloaders
from pipeline.datasets.preprocessing import SICKLEPreprocessingPipeline
from pipeline.training import SICKLETrainer, TrainingConfig
from pipeline.evaluation import EvaluationPipeline
from pipeline.models.inference import ModelInference, SICKLEInference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import UTAE model for demo
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
try:
    from utae import UTAE
except ImportError:
    logger.warning("UTAE model not available")


def create_synthetic_sickle_data(n_samples: int = 50,
                                 temporal_frames: int = 61,
                                 patch_size: int = 32) -> tuple:
    """
    Create synthetic SICKLE-compatible satellite data for demo.
    
    Returns:
        (satellite_data, labels) where:
        - satellite_data: (n_samples, temporal_frames, 21, patch_size, patch_size)
          21 channels = 12 S2 + 2 S1 + 7 L8
        - labels: (n_samples,) with crop types (0-19)
    """
    logger.info(f"🔨 Creating synthetic SICKLE data...")
    logger.info(f"   Samples: {n_samples}")
    logger.info(f"   Temporal frames: {temporal_frames}")
    logger.info(f"   Spatial size: {patch_size}x{patch_size}")
    
    # Create satellite data (T, C, H, W) format per sample
    data_list = []
    labels_list = []
    
    for i in range(n_samples):
        # Simulate temporal sequence of multi-sensor data
        temporal_seq = []
        
        for t in range(temporal_frames):
            # Progress through growing season
            progress = t / temporal_frames
            
            # Simulate S2 data (12 bands)
            # NDVI increases during growing season
            s2_data = np.random.randn(12, patch_size, patch_size) * 0.2
            
            # Add NDVI pattern (grows then decreases)
            ndvi_pattern = np.sin(progress * np.pi) * 0.4 + 0.3
            s2_data[7] += ndvi_pattern  # NIR band
            s2_data[3] -= ndvi_pattern  # Red band
            
            # Normalize
            s2_data = np.clip(s2_data, 0, 1)
            
            # Simulate S1 data (2 bands, VV and VH)
            s1_data = np.random.randn(2, patch_size, patch_size) * 0.1
            s1_data = np.clip(s1_data, -1, 1)
            
            # Simulate L8 data (7 bands)
            l8_data = np.random.randn(7, patch_size, patch_size) * 0.2
            l8_data = np.clip(l8_data, 0, 1)
            
            # Stack all sensors
            frame = np.concatenate([s2_data, s1_data, l8_data], axis=0)  # (21, H, W)
            temporal_seq.append(frame)
        
        # Stack temporal dimension
        sample = np.stack(temporal_seq, axis=0)  # (T, C, H, W)
        data_list.append(sample)
        
        # Create synthetic label
        label = i % 20  # 20 crop classes (PASTIS style)
        labels_list.append(label)
    
    satellite_data = np.stack(data_list, axis=0)  # (N, T, C, H, W)
    labels = np.array(labels_list)
    
    logger.info(f"✅ Synthetic data created:")
    logger.info(f"   Shape: {satellite_data.shape}")
    logger.info(f"   Labels shape: {labels.shape}")
    
    return satellite_data, labels


def create_demo_model(input_channels: int = 21,
                     num_classes: int = 20) -> torch.nn.Module:
    """Create UTAE model for demo."""
    logger.info(f"🏗️ Creating UTAE model...")
    
    model = UTAE(
        input_dim=input_channels,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, num_classes],
        n_head=16,
        d_model=256
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Parameters: {n_params:,}")
    
    return model


def demo_1_synthetic_data():
    """Demo 1: Create and explore synthetic data."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 1: Creating Synthetic SICKLE Data")
    logger.info("="*60)
    
    # Create synthetic data
    data, labels = create_synthetic_sickle_data(n_samples=50)
    
    # Save for later use
    data_dir = Path('./demo_data')
    data_dir.mkdir(exist_ok=True)
    
    np.save(data_dir / 'synthetic_satellite_data.npy', data)
    np.save(data_dir / 'synthetic_labels.npy', labels)
    
    with open(data_dir / 'metadata.json', 'w') as f:
        json.dump({
            'n_samples': len(data),
            'temporal_frames': data.shape[1],
            'n_channels': data.shape[2],
            'spatial_size': (data.shape[3], data.shape[4]),
            'channels': ['S2_12bands', 'S1_2bands', 'L8_7bands'],
            'n_classes': len(np.unique(labels))
        }, f, indent=2)
    
    logger.info(f"✅ Synthetic data saved to: {data_dir}")
    
    return data, labels, data_dir


def demo_2_preprocessing():
    """Demo 2: Preprocessing pipeline."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 2: Data Preprocessing")
    logger.info("="*60)
    
    # Load synthetic data
    data_dir = Path('./demo_data')
    data = np.load(data_dir / 'synthetic_satellite_data.npy')
    
    logger.info(f"📊 Original data range: [{data.min():.3f}, {data.max():.3f}]")
    
    # Create preprocessing pipeline
    preprocessor = SICKLEPreprocessingPipeline(
        cloud_threshold=20.0,
        normalization='minmax',
        include_indices=True,
        quality_control=True
    )
    
    # Preprocess a single sample
    sample = data[0, 0]  # First sample, first timestep
    
    # Split into sensor components
    s2 = sample[:12]
    s1 = sample[12:14]
    l8 = sample[14:]
    
    results = preprocessor.process_single_temporal(sentinel2=s2, sentinel1=s1, landsat8=l8)
    
    logger.info(f"✅ Preprocessing complete:")
    logger.info(f"   Processed shape: {results['stacked_shape']}")
    logger.info(f"   Total bands: {results['total_bands']}")
    logger.info(f"   Validity scores: {results['validity_scores']}")
    
    return data_dir


def demo_3_training():
    """Demo 3: Model training."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 3: Model Training")
    logger.info("="*60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🎯 Device: {device}")
    
    # Load synthetic data
    data_dir = Path('./demo_data')
    satellite_data = np.load(data_dir / 'synthetic_satellite_data.npy')
    labels = np.load(data_dir / 'synthetic_labels.npy')
    
    # Convert to PyTorch tensors
    data_tensor = torch.from_numpy(satellite_data).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    # Create dataset
    dataset = TensorDataset(data_tensor, labels_tensor)
    
    # Split into train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: {
            'image': torch.stack([x[0] for x in batch]),
            'label': torch.stack([torch.tensor(x[1]) for x in batch])
        }
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=lambda batch: {
            'image': torch.stack([x[0] for x in batch]),
            'label': torch.stack([torch.tensor(x[1]) for x in batch])
        }
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        collate_fn=lambda batch: {
            'image': torch.stack([x[0] for x in batch]),
            'label': torch.stack([torch.tensor(x[1]) for x in batch])
        }
    )
    
    # Create model
    model = create_demo_model(input_channels=21, num_classes=20)
    
    # Create training config
    config = TrainingConfig(
        max_epochs=3,  # Short demo
        batch_size=4,
        learning_rate=1e-4,
        optimizer='adamw',
        patience=5,
        task_type='crop_classification'
    )
    
    # Create trainer
    output_dir = Path('./demo_checkpoints')
    trainer = SICKLETrainer(
        model=model,
        device=device,
        config=config,
        output_dir=output_dir
    )
    
    # Train
    logger.info(f"🚀 Starting training for {config.max_epochs} epochs...")
    trainer.train(train_loader, val_loader, num_epochs=3)
    
    logger.info(f"✅ Training complete!")
    logger.info(f"   Best model: {trainer.best_model_path}")
    
    return trainer.best_model_path


def demo_4_inference():
    """Demo 4: Inference."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 4: Model Inference")
    logger.info("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load trained model
    model_path = Path('./demo_checkpoints/best_model.pth')
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    logger.info(f"📦 Loading model: {model_path}")
    
    inference = ModelInference(
        model_path=str(model_path),
        model_type='utae',
        device=device
    )
    
    # Load test data
    data_dir = Path('./demo_data')
    satellite_data = np.load(data_dir / 'synthetic_satellite_data.npy')
    labels = np.load(data_dir / 'synthetic_labels.npy')
    
    # Run inference on first 10 samples
    logger.info(f"🔮 Running inference on 10 samples...")
    
    test_data = satellite_data[:10]
    test_labels = labels[:10]
    
    # Reshape for inference (T, C, H, W) format
    test_sample = test_data[0]  # First sample
    
    results = inference.predict(test_sample, return_probabilities=True)
    
    logger.info(f"✅ Inference complete!")
    logger.info(f"   Input shape: {results['input_shape']}")
    logger.info(f"   Output shape: {results['output_shape']}")
    logger.info(f"   Mean confidence: {results['mean_confidence']:.3f}")
    logger.info(f"   Predicted class: {results['class_predictions']}")
    
    return inference


def demo_5_evaluation():
    """Demo 5: Model evaluation."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 5: Model Evaluation")
    logger.info("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and test data
    model_path = Path('./demo_checkpoints/best_model.pth')
    data_dir = Path('./demo_data')
    
    satellite_data = np.load(data_dir / 'synthetic_satellite_data.npy')
    labels = np.load(data_dir / 'synthetic_labels.npy')
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    inference = ModelInference(
        model_path=str(model_path),
        model_type='utae',
        device=device
    )
    
    # Get predictions on test set
    all_preds = []
    
    logger.info(f"🔮 Getting predictions for all samples...")
    
    for sample in satellite_data:
        results = inference.predict(sample, return_probabilities=False)
        all_preds.append(results['class_predictions'])
    
    all_preds = np.array(all_preds)
    
    # Evaluate
    evaluator = EvaluationPipeline(
        output_dir=Path('./demo_results'),
        verbose=True
    )
    
    metrics = evaluator.evaluate_crop_classification(
        labels,
        all_preds.flatten(),
        save_results=True
    )
    
    logger.info(f"✅ Evaluation complete!")
    logger.info(f"   Results: ./demo_results/")
    
    return metrics


def demo_6_agricultural_analysis():
    """Demo 6: Agricultural analysis."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 6: Agricultural Analysis")
    logger.info("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model_path = Path('./demo_checkpoints/best_model.pth')
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Create SICKLE-specific inference wrapper
    sickle_inference = SICKLEInference(
        model_path=str(model_path),
        model_type='utae'
    )
    
    # Load test sample
    data_dir = Path('./demo_data')
    satellite_data = np.load(data_dir / 'synthetic_satellite_data.npy')
    
    test_sample = satellite_data[0]
    
    logger.info(f"🌾 Running agricultural analysis...")
    
    results = sickle_inference.predict_field(test_sample)
    
    logger.info(f"✅ Agricultural analysis complete!")
    
    if 'agricultural_analysis' in results:
        agr = results['agricultural_analysis']
        logger.info(f"\n   Dominant crop: {agr['dominant_crop']}")
        logger.info(f"   Dominance: {agr['dominant_crop_percentage']:.1f}%")
        logger.info(f"   Field diversity: {agr['field_diversity']} crop types")
        logger.info(f"   Classification confidence: {agr['field_classification_confidence']:.3f}")
        
        if 'crop_distribution' in agr:
            logger.info(f"\n   Crop distribution:")
            for crop, dist in agr['crop_distribution'].items():
                logger.info(f"      {crop}: {dist['percentage']:.1f}%")


def main():
    """Run all demos."""
    logger.info("\n")
    logger.info("🎯 SICKLE++ BASELINE COMPLETE DEMO")
    logger.info("="*60)
    
    try:
        # Demo 1: Create synthetic data
        data, labels, data_dir = demo_1_synthetic_data()
        
        # Demo 2: Preprocessing
        demo_2_preprocessing()
        
        # Demo 3: Training
        model_path = demo_3_training()
        
        # Demo 4: Inference
        demo_4_inference()
        
        # Demo 5: Evaluation
        demo_5_evaluation()
        
        # Demo 6: Agricultural analysis
        demo_6_agricultural_analysis()
        
        logger.info("\n" + "="*60)
        logger.info("✅ ALL DEMOS COMPLETE!")
        logger.info("="*60)
        logger.info("\nGenerated files:")
        logger.info("  - demo_data/              (synthetic satellite data)")
        logger.info("  - demo_checkpoints/       (trained models)")
        logger.info("  - demo_results/           (evaluation results)")
        logger.info("\nNext steps:")
        logger.info("  1. Replace synthetic data with real satellite imagery")
        logger.info("  2. Run train.py with your dataset")
        logger.info("  3. Use infer.py for production inference")
        
    except Exception as e:
        logger.error(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
