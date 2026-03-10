"""
SICKLE++ Pipeline Package

Complete pipeline for multi-sensor satellite-based crop monitoring baseline.

Modules:
---------
- datasets: Dataset classes and loaders for SICKLE data
- download: Google Earth Engine data downloading
- features: Feature engineering (indices, SAR ratios)
- geometry: Geometry loading and validation
- models: Model inference wrapper
- evaluation: Evaluation metrics and visualization
- training: Training pipeline and utilities

Key Classes:
-----------
- SICKLEDataset: PyTorch dataset for satellite data
- SICKLETimeSeriesDataset: Temporal sequence handling
- EarthEngineDownloader: Multi-sensor data download
- FeatureEngineer: Agricultural indices computation
- ModelInference: Model inference wrapper
- SICKLEInference: Agricultural-specific inference
- SICKLETrainer: Training loop manager
- EvaluationPipeline: Metrics and evaluation

Example Usage:
--------------
from pipeline.datasets import SICKLEDataset, create_dataloaders
from pipeline.training import SICKLETrainer, TrainingConfig
from pipeline.models.inference import ModelInference

# Load data
dataset = SICKLEDataset('data_dir', 'labels.json')
train_loader, val_loader, test_loader = create_dataloaders(dataset)

# Train model
trainer = SICKLETrainer(model, config=TrainingConfig())
trainer.train(train_loader, val_loader)

# Inference
inference = ModelInference('model.pth')
results = inference.predict(data)
"""

from . import datasets
from . import download
from . import features
from . import geometry
from . import models
from . import evaluation
from . import training

__all__ = [
    'datasets',
    'download',
    'features',
    'geometry',
    'models',
    'evaluation',
    'training'
]
