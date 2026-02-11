# SICKLE++ Live Inference Pipeline

A comprehensive pipeline for satellite-based crop monitoring and yield prediction using Google Earth Engine, designed for live inference and dataset building.

## Features

- **Multi-sensor data download** from Google Earth Engine (Sentinel-1, Sentinel-2, Landsat 8)
- **KML/GeoJSON geometry support** with automatic conversion
- **Comprehensive feature engineering** including vegetation indices, SAR ratios, and spectral transformations
- **Live inference pipeline** for real-time crop monitoring
- **Dataset building mode** for training data collection
- **Batch processing** for multiple geometries

## Quick Start

### 1. Prerequisites

Install required packages:
```bash
pip install earthengine-api geopandas torch numpy
```

Authenticate with Google Earth Engine:
```bash
earthengine authenticate
```

### 2. Basic Usage

#### Live Inference
```bash
python dataset_download/download_script.py --mode live --geometry field.kml --model trained_model.pth
```

#### Dataset Building
```bash
python dataset_download/download_script.py --mode dataset --geometry field.geojson --output ./dataset
```

#### Demo Mode
```bash
python dataset_download/download_script.py --mode demo
```

## Architecture

```
pipeline/
├── geometry/           # KML/GeoJSON loading and conversion
│   └── loader.py      
├── download/           # Earth Engine data download
│   └── earth_engine.py
├── features/           # Feature engineering pipeline
│   └── engineering.py
├── models/             # Model inference wrapper
│   └── inference.py
└── orchestrator/       # Live pipeline orchestration
    └── live_pipeline.py
```

## Core Components

### Geometry Loader
- Loads KML/GeoJSON files
- Converts KML to GeoJSON automatically
- Validates geometry size
- Creates Earth Engine geometry objects

### Earth Engine Downloader
- Downloads Sentinel-1 SAR data with speckle filtering
- Downloads Sentinel-2 optical data with cloud masking
- Downloads Landsat 8 data (optional)
- Multi-sensor stacking capabilities

### Feature Engineer
- Vegetation indices (NDVI, EVI, SAVI, GNDVI, etc.)
- SAR indices (VV/VH ratios, radar vegetation index)
- Texture features (GLCM)
- Spectral transformations (Tasseled Cap)

### Model Inference
- PyTorch model wrapper
- Support for yield prediction and classification
- Batch processing capabilities
- Confidence metrics

### Live Pipeline Orchestrator
- End-to-end pipeline execution
- Configuration management
- Result caching and export
- Error handling and logging

## API Usage

### Python API

```python
from pipeline.orchestrator.live_pipeline import LiveInferencePipeline

# Initialize pipeline
pipeline = LiveInferencePipeline(model_path="model.pth")

# Run inference
results = pipeline.run_live_inference(
    geometry_file="field.geojson",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(f"Mean yield: {results['inference_results']['mean_yield']}")
```

### Quick Functions

```python
from pipeline.orchestrator.live_pipeline import quick_inference, quick_dataset_build

# Quick inference
results = quick_inference(
    geometry_file="field.kml",
    model_path="model.pth"
)

# Quick dataset building
dataset_info = quick_dataset_build(
    geometry_file="field.geojson",
    date_ranges=[("2023-01-01", "2023-12-31")],
    output_dir="./dataset"
)
```

## Configuration

Pipeline behavior can be configured via `config.json` or by passing a config file:

```json
{
  "download": {
    "cloud_threshold": 15.0,
    "scale": 10,
    "include_s1": true,
    "include_s2": true,
    "include_l8": false
  },
  "features": {
    "include_vegetation": true,
    "include_sar": true,
    "include_texture": false,
    "include_spectral": false,
    "normalization": "minmax"
  },
  "inference": {
    "task_type": "yield_prediction",
    "confidence_threshold": 0.5
  }
}
```

## Examples

### 1. Field-level Yield Prediction

```bash
# Create field geometry (KML/GeoJSON)
python dataset_download/download_script.py --mode live \
  --geometry corn_field_2024.kml \
  --model corn_yield_model.pth \
  --start-date 2024-06-01 \
  --end-date 2024-08-31 \
  --output ./results
```

### 2. Multi-temporal Dataset Creation

```bash
# Build training dataset
python dataset_download/download_script.py --mode dataset \
  --geometry training_fields.geojson \
  --output ./training_data
```

### 3. Batch Processing Multiple Fields

```bash
# Process all KML files in directory
python dataset_download/download_script.py --mode batch \
  --geometries "fields/*.kml" \
  --model production_model.pth \
  --output ./batch_results
```

## Output Structure

### Live Inference Output
```
output/
├── results.json           # Pipeline results and metadata
├── features.npz          # Feature arrays
└── inference_outputs.npz # Model predictions
```

### Dataset Output
```
dataset/
├── dataset.npz           # Stacked feature arrays
└── dataset_info.json    # Dataset metadata
```

## Model Integration

The pipeline supports PyTorch models with the following interface:

```python
import torch
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        # Your model architecture
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        return predictions
```

Save models with metadata:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {'input_channels': 10, 'num_classes': 1},
    'metadata': {'model_type': 'yield_prediction'}
}, 'model.pth')
```

## Best Practices

1. **Geometry Size**: Keep geometries under 10,000 km² for optimal performance
2. **Date Ranges**: Use 30-day windows for reliable cloud-free composites
3. **Feature Selection**: Enable only needed features to reduce processing time
4. **Model Input**: Ensure model expects correct number of input channels
5. **Caching**: Pipeline caches intermediate results for debugging

## Troubleshooting

### Earth Engine Authentication
```bash
earthengine authenticate
```

### Memory Issues
- Reduce geometry size or increase scale parameter
- Disable texture/spectral features if not needed
- Use batch processing for large areas

### No Data Available
- Check date ranges and cloud thresholds
- Verify geometry is over land areas
- Use data availability info in results

## License

MIT License - See LICENSE file for details.