# SICKLE++ Baseline Reproduction Guide

Complete guide to reproducing the SICKLE baseline using the SICKLE++ pipeline.

## Overview

This guide establishes a **reproducible baseline** for the SICKLE paper:
- **SICKLE: A Multi-Sensor Satellite Imagery Dataset**
- Conference: WACV 2024
- Key tasks: Crop classification, Phenology prediction, Yield estimation

## Baseline Target Performance

Paper reported baseline accuracies:
- **Crop Classification**: 70-85% accuracy
- **Phenology Prediction**: 5-10 days MAE
- **Yield Prediction**: RMSE 0.4-0.6

## Architecture & Components

### Data Pipeline
```
Raw Satellite Data (Earth Engine)
        ↓
Cloud Masking & Preprocessing
        ↓
Feature Engineering (Indices, SAR)
        ↓
Temporal Sequence Construction
        ↓
PyTorch DataLoaders
```

### Learning Pipeline
```
Dataset Loading
        ↓
Model Creation (UTAE, UNet3D, etc.)
        ↓
Training Loop with Validation
        ↓
Early Stopping & Checkpoints
        ↓
Evaluation on Test Set
```

## Installation

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install earthengine-api geopandas rasterio
pip install scikit-learn matplotlib seaborn pandas tqdm
```

### 2. Google Earth Engine Authentication

```bash
# Authenticate with Google
earthengine authenticate

# Verify
earthengine info
```

## Dataset Preparation

### Option A: Download Real Data (Recommended)

```python
from pipeline.download.earth_engine import EarthEngineDownloader
from pipeline.geometry.loader import load_geometry

# Load your field geometry
geometry_file = 'field.geojson'  # KML or GeoJSON

# Create downloader
downloader = EarthEngineDownloader()

# Download multi-sensor data
s2 = downloader.download_sentinel2(geometry, '2021-01-01', '2021-12-31')
s1 = downloader.download_sentinel1(geometry, '2021-01-01', '2021-12-31')
l8 = downloader.download_landsat8(geometry, '2021-01-01', '2021-12-31')
```

### Option B: Use Synthetic Data (Quick Start)

```bash
# Run demo with synthetic data
python scripts/demo.py
```

This creates:
- 50 synthetic temporal sequences
- SICKLE-compatible multi-sensor format
- 61 timesteps per sequence
- 20 crop classes

## Training a Model

### Quick Start

```bash
python scripts/train.py \
    --data_dir ./demo_data \
    --labels_path ./demo_data/synthetic_labels.npy \
    --model utae \
    --task crop_classification \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --output_dir ./checkpoints
```

### Advanced Training Configuration

```python
from pipeline.training import SICKLETrainer, TrainingConfig
from pipeline.datasets import SICKLEDataset, create_dataloaders

# Load dataset
dataset = SICKLEDataset(
    'satellite_data_dir',
    'labels.json',
    task_type='crop_classification',
    normalize=True
)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    batch_size=8,
    num_workers=4
)

# Configure training
config = TrainingConfig(
    max_epochs=50,
    learning_rate=1e-4,
    optimizer='adamw',
    weight_decay=1e-5,
    patience=10,
    task_type='crop_classification'
)

# Train
trainer = SICKLETrainer(model, config=config)
trainer.train(train_loader, val_loader)
```

## Model Architectures

Implemented baseline models:

### 1. UTAE (U-TAE)
- Temporal attention mechanism
- Efficient for time series
- **Recommended for beginners**

```python
from models.utae import UTAE

model = UTAE(
    input_dim=21,
    encoder_widths=[64, 64, 64, 128],
    decoder_widths=[32, 32, 64, 128],
    out_conv=[32, 20],
    n_head=16,
    d_model=256
)
```

### 2. UNet3D
- 3D CNN for spatio-temporal
- Good for semantic segmentation

```python
from models.unet3d import UNet3D

model = UNet3D(
    in_channel=21,
    n_classes=20,
    timesteps=61,
    dropout=0.2
)
```

### 3. PaSTiS-UNet3D
- Multi-scale temporal fusion
- State-of-the-art for series

```python
from models.pastis_unet3d import PaSTiSUNet3D

model = PaSTiSUNet3D(
    input_dim=21,
    num_classes=20
)
```

## Feature Engineering

The pipeline includes SICKLE-compatible feature engineering:

### Vegetation Indices
- **NDVI**: Primary vegetation indicator
- **GNDVI**: Green NDVI (chlorophyll)
- **NDRE**: Red edge (crop stress)
- **EVI**: Enhanced vegetation index
- **SAVI**: Soil-adjusted VI

### SAR Indices (Sentinel-1)
- **VV/VH ratio**: Crop structure
- **NDSAR**: Normalized SAR difference
- **RVI**: Radar vegetation index

### Preprocessing Steps
1. Cloud masking (Sentinel-2)
2. Speckle filtering (Sentinel-1)
3. Spatial alignment
4. Min-max normalization
5. Quality control (>75% valid pixels)

## Dataset Structure

Expected directory layout:

```
dataset_dir/
├── image_001.npy          # (T, C, H, W) or (C, H, W)
├── image_002.npy
├── ...
├── metadata.json          # Dataset info
└── labels.json or labels.npy
```

Data formats supported:
- **NPY**: Numpy binary arrays
- **NPZ**: Compressed numpy
- **GeoTIFF**: Rasterio compatible
- **JSON**: Metadata and labels

## Inference Pipeline

### Basic Inference

```bash
python scripts/infer.py \
    --model_path ./checkpoints/best_model.pth \
    --data_path ./satellite_data.npy \
    --task crop_classification \
    --output_dir ./results
```

### Python API

```python
from pipeline.models.inference import ModelInference, SICKLEInference

# Load model
inference = ModelInference(
    'best_model.pth',
    model_type='utae'
)

# Run inference
results = inference.predict(satellite_data)

# Get agricultural analysis
sickle = SICKLEInference('best_model.pth')
field_results = sickle.predict_field(data)

print(f"Dominant crop: {field_results['dominant_crop']}")
print(f"Confidence: {field_results['field_classification_confidence']:.3f}")
```

## Evaluation Metrics

### Crop Classification

```python
from pipeline.evaluation import EvaluationPipeline

evaluator = EvaluationPipeline(output_dir='./results')

metrics = evaluator.evaluate_crop_classification(
    y_true=labels,
    y_pred=predictions,
    save_results=True
)

# Returns:
# - accuracy
# - F1 score (macro & weighted)
# - confusion matrix
# - per-class metrics
```

**Target baseline**: **70-85% accuracy**

### Phenology Prediction

```python
metrics = evaluator.evaluate_phenology_prediction(
    y_true=true_doy,      # Day of year
    y_pred=pred_doy,
    save_results=True
)

# Returns:
# - MAE in days
# - RMSE in days
# - percentage error
```

**Target baseline**: **5-10 days MAE**

### Yield Prediction

```python
metrics = evaluator.evaluate_yield_prediction(
    y_true=true_yield,
    y_pred=pred_yield
)

# Returns:
# - RMSE
# - MAE
# - R² score
# - MAPE
```

**Target baseline**: **RMSE 0.4-0.6**

## Complete Workflow Example

```python
# 1. Load data
from pipeline.datasets import SICKLEDataset, create_dataloaders

dataset = SICKLEDataset('data_dir', 'labels.json')
train_loader, val_loader, test_loader = create_dataloaders(dataset)

# 2. Create model
from models.utae import UTAE

model = UTAE(input_dim=21, encoder_widths=[64,64,64,128], 
             decoder_widths=[32,32,64,128], out_conv=[32,20],
             n_head=16, d_model=256)

# 3. Train
from pipeline.training import SICKLETrainer, TrainingConfig

config = TrainingConfig(max_epochs=50, learning_rate=1e-4)
trainer = SICKLETrainer(model, config=config, output_dir='./checkpoints')
trainer.train(train_loader, val_loader)

# 4. Evaluate
from pipeline.evaluation import EvaluationPipeline

evaluator = EvaluationPipeline(output_dir='./results')
trainer.model.eval()

with torch.no_grad():
    predictions = []
    for batch in test_loader:
        outputs = trainer.model(batch['image'])
        predictions.append(outputs.argmax(dim=1))

metrics = evaluator.evaluate_crop_classification(
    test_targets, torch.cat(predictions)
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch_size 4

# Use gradient checkpointing
python train.py --gradient_checkpoint

# Cache less data
dataset = SICKLEDataset(..., cache_in_memory=False)
```

### Poor Performance

1. **Check data quality**
   - Verify cloud masking
   - Check validity thresholds
   - Examine data distributions

2. **Hyperparameter tuning**
   - Learning rate: try 1e-5 to 1e-3
   - Batch size: 4, 8, 16
   - Optimizer: adamw (recommended)

3. **Add regularization**
   - Increase weight decay
   - Add dropout
   - Use data augmentation

### CUDA Errors

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU if needed
python scripts/train.py ... --device cpu
```

## Expected Results Timeline

| Week | Task | Expected Progress |
|------|------|------------------|
| 1 | Dataset setup & validation | ✅ Complete |
| 2 | Preprocessing pipeline | ✅ Complete |
| 3 | Baseline model training | UTAE: ~75% accuracy |
| 4 | Hyperparameter tuning | Improve to 80%+ |
| 5 | Advanced model (temporal transformer) | Target 85%+ |

## Paper Improvement Ideas

Once baseline is established, improve with:

### 1. Temporal Modeling
- Replace LSTM with Transformer
- Add temporal positional encoding
- Multi-scale temporal pooling

### 2. Multi-Sensor Fusion
- Cross-attention between sensors
- Adaptive fusion weights
- Late vs. early fusion comparison

### 3. Self-Supervised Learning
- Pre-train with masked autoencoders
- Contrastive learning on time series
- Foundation models (SatMAE, etc.)

### 4. Data Augmentation
- Temporal warping
- Spectral augmentation
- Spatial augmentation

### 5. Ensemble Methods
- Combine multiple architectures
- Temporal ensembling
- Multi-task learning

## Citation & References

**SICKLE Paper**:
```bibtex
@inproceedings{sani2024sickle,
  title={SICKLE: A Multi-Sensor Satellite Imagery Dataset Annotated with Multiple Key Cropping Parameters},
  author={Sani, Depanshu and others},
  booktitle={WACV 2024},
  year={2024}
}
```

## Support & Contributing

This codebase provides:
- ✅ Complete preprocessing pipeline
- ✅ Multiple baseline architectures
- ✅ Training framework
- ✅ Evaluation metrics
- ✅ Agricultural analysis tools

For questions or improvements, see PIPELINE_README.md

---

**Last Updated**: March 2024
**Status**: Production Ready
**Test Coverage**: Synthetic data demo included
