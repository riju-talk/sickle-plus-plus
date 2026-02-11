# Google Earth Engine Dry Run Script

A standalone script to analyze satellite data availability from Google Earth Engine without downloading any actual data.

## Features

- 📍 **Geometry Analysis**: Validates GeoJSON files and computes area/bounds
- 🛰️ **Multi-Sensor Coverage**: Checks Sentinel-2, Sentinel-1, and Landsat 8 availability  
- 📊 **Band Information**: Lists all available bands with descriptions
- 💾 **Size Estimation**: Estimates download size for planning
- 🎯 **Smart Recommendations**: Suggests optimal parameters
- 💾 **JSON Export**: Saves detailed analysis results

## Quick Start

1. **Authenticate with Earth Engine**:
   ```bash
   earthengine authenticate
   ```

2. **Run dry run analysis**:
   ```bash
   python gee_dry_run.py test_field.geojson
   ```

3. **Specify custom date range**:
   ```bash
   python gee_dry_run.py test_field.geojson --start-date 2024-01-01 --end-date 2024-12-31
   ```

4. **Save results to file**:
   ```bash
   python gee_dry_run.py test_field.geojson --output analysis_results.json
   ```

## What It Analyzes

### Sentinel-2 (Optical)
- 🔬 **13 spectral bands** from coastal to SWIR
- ☁️ **Cloud coverage analysis** (shows total vs. cloud-free images)
- 📏 **10m spatial resolution**

### Sentinel-1 (SAR)
- 📡 **VV/VH polarizations** for crop monitoring
- 🛰️ **Ascending/descending orbits** for complete coverage
- 📏 **10m spatial resolution**

### Landsat 8 (Optical)
- 🌈 **11 spectral bands** including thermal
- ☁️ **Cloud coverage filtering**
- 📏 **30m spatial resolution**

## Example Output

```
🛰️  Google Earth Engine Dry Run Analysis
==================================================
Date range: 2024-01-01 to 2024-12-31

📍 Analyzing geometry: test_field.geojson
   Area: 121.00 km²
   Bounding box: (-93.5, 41.5, -93.4, 41.6)
   Centroid: [-93.45, 41.55]
   Size valid: ✅

🛰️  Analyzing Sentinel-2 data...
   Total images available: 47
   Images with <20% clouds: 31
   Available bands: 13
   📊 Sentinel-2 Bands:
      ✅ B2: Blue (490nm)
      ✅ B3: Green (560nm)
      ✅ B4: Red (665nm)
      ✅ B8: NIR (842nm)
      ...

📡 Analyzing Sentinel-1 SAR data...
   Total SAR images: 73
   Ascending orbit: 37
   Descending orbit: 36
   Available bands: 3
   📊 Sentinel-1 Bands:
      ✅ VV: Vertical transmit, vertical receive
      ✅ VH: Vertical transmit, horizontal receive
      ✅ angle: Incidence angle

🌍 Analyzing Landsat 8 data...
   Total images available: 23
   Images with <20% clouds: 18
   Available bands: 18
   📊 Landsat 8 Bands:
      ✅ SR_B2: Blue (482nm)
      ✅ SR_B3: Green (562nm)
      ✅ SR_B4: Red (655nm)
      ...

💾 Estimating download size...
   SENTINEL2:
      Images: 31
      Size per image: 15.7 MB
      Total size: 487.2 MB
   SENTINEL1:
      Images: 73
      Size per image: 9.7 MB  
      Total size: 708.1 MB
   LANDSAT8:
      Images: 18
      Size per image: 21.8 MB
      Total size: 392.4 MB
   📊 TOTAL ESTIMATED SIZE: 1587.7 MB (1.59 GB)

🎯 Recommendations:
   ✅ Good data availability
   💾 Moderate download size - should be manageable
```

## Command Line Options

```bash
python gee_dry_run.py <geojson_file> [options]

Arguments:
  geojson_file          Path to GeoJSON file with area of interest

Options:
  --start-date YYYY-MM-DD  Start date (default: 30 days ago)
  --end-date YYYY-MM-DD    End date (default: today)
  --output FILE            Save detailed analysis to JSON file
  -h, --help               Show help message
```

## Use Cases

- 📊 **Data availability assessment** before starting analysis projects
- 💰 **Cost estimation** for commercial Earth Engine usage 
- 📅 **Optimal date range selection** for crop monitoring
- 🗺️ **Multi-sensor data planning** for research projects
- ⚡ **Quick feasibility checks** for new study areas

## Notes

- Requires authenticated Google Earth Engine account
- Analysis is free (no data download charges)
- Size estimates are approximate - actual sizes may vary
- Cloud filtering uses 20% threshold by default
- Works with any valid GeoJSON geometry