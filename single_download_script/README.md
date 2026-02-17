# SICKLE++ Data Download Script

This script downloads satellite data from Google Earth Engine for agricultural field analysis.

## Features

- ✅ **Downloads Sentinel-1, Sentinel-2, and Landsat-8 data** for agricultural analysis
- 📁 **Creates organized directory structure** for downloaded files
- 🗺️ **Validates field geometry** and area suitability  
- 💾 **Saves metadata** about downloads and processing
- 🌾 **Optimized for crop monitoring** with SICKLE-compatible bands

## Usage

### Basic Download

```bash
python gee_download.py test_field.geojson
```

### Advanced Options

```bash
python gee_download.py test_field.geojson --output-dir ./my_downloads --start-date 2018-08-01 --end-date 2020-08-01
```

### Arguments

- `geojson_file`: Path to GeoJSON file containing field geometry (required)
- `--output-dir`: Output directory for downloaded data (default: `downloads`)
- `--start-date`: Start date in YYYY-MM-DD format (default: `2018-08-01`)
- `--end-date`: End date in YYYY-MM-DD format (default: `2020-08-01`)

## Directory Structure

The script creates an organized directory structure:

```
downloads/
├── sentinel1/          # SAR data files (VV, VH polarizations)
├── sentinel2/          # Multispectral data (12 bands: B1-B12)
├── landsat8/           # Optical + Thermal data (8 bands: B1-B7, B10)
├── metadata/           # Download metadata and configuration
│   └── download_metadata.json
└── geometry/           # Copy of field geometry file
    └── test_field.geojson
```

## Prerequisites

1. **Google Earth Engine Authentication**:
   ```bash
   earthengine authenticate
   ```

2. **Required Python packages** (see project requirements):
   - `earthengine-api`
   - `geojson`
   - `numpy`

## Satellite Data

### Sentinel-2 (Multispectral)
- **Bands**: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
- **Resolution**: 10m
- **Cloud filtering**: <20% cloud coverage
- **Agricultural focus**: Vegetation indices, chlorophyll, moisture

### Sentinel-1 (SAR)
- **Bands**: VV, VH polarizations
- **Resolution**: 10m  
- **Mode**: Interferometric Wide (IW)
- **Agricultural focus**: Crop structure, biomass, phenology

### Landsat-8 (Optical + Thermal)
- **Bands**: SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7, ST_B10
- **Resolution**: 30m (optical), 100m (thermal)
- **Cloud filtering**: <20% cloud coverage
- **Agricultural focus**: Vegetation health, thermal stress, crop type classification
- **Thermal band**: Surface temperature (ST_B10) for crop stress analysis

## Output Files

Files are exported to Google Drive in the `sickle_downloads` folder. Download them manually and place in the appropriate directories:

1. **Sentinel-2 files** → `sentinel2/` directory
2. **Sentinel-1 files** → `sentinel1/` directory
3. **Landsat-8 files** → `landsat8/` directory

## Example Field File

The included `test_field.geojson` shows the expected format:

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {
      "name": "Test Agricultural Field",
      "crop_type": "corn",
      "location": "Iowa, USA"
    },
    "geometry": {
      "type": "Polygon",
      "coordinates": [[
        [-93.5, 41.5], [-93.4, 41.5], 
        [-93.4, 41.6], [-93.5, 41.6], 
        [-93.5, 41.5]
      ]]
    }
  }]
}
```

## Integration with Inference Pipeline

After downloading data, use the inference pipeline:

```python
from pipeline.orchestrator.inference_pipeline import InferencePipeline

# Initialize pipeline with downloaded data
pipeline = InferencePipeline(model_path="path/to/model.pth")
results = pipeline.run_inference_from_downloaded_data("./downloads")
```

