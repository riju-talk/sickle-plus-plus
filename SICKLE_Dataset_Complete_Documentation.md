# SICKLE Dataset: Complete Documentation

## Overview and Introduction

**SICKLE** (A Multi-Sensor Satellite Imagery Dataset Annotated with Multiple Key Cropping Parameters) is a comprehensive agricultural remote sensing dataset specifically designed for paddy cultivation analysis in the Cauvery Delta region of Tamil Nadu, India. The dataset was published at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2024 as an oral presentation.

### Key Highlights
- **Multi-sensor**: Combines data from Landsat-8, Sentinel-1, and Sentinel-2 satellites
- **Multi-modal**: Includes optical, thermal, and microwave sensor data  
- **Multi-resolution**: Provides annotations at 3m, 10m, and 30m spatial resolutions
- **Multi-temporal**: Time-series data spanning January 2018 to March 2021
- **Multi-task**: Supports multiple agricultural prediction tasks
- **Real ground-truth**: Based on extensive field surveys from January 2021 to February 2022

## Dataset Characteristics

### Geographic Coverage
- **Study Region**: Cauvery Delta Region, Tamil Nadu, India
- **Focus**: Paddy cultivation areas
- **Field Data Collection**: Ground-based surveys across 388 unique plots
- **Average Plot Size**: 0.38 acres

### Temporal Coverage
- **Time Span**: January 2018 - March 2021 (3+ years)
- **Field Data Collection Period**: January 2021 - February 2022
- **Time-Series Preparation**: Based on regional growing seasons of paddy cultivation
- **Total Satellite Images**: ~209,000 images

### Scale and Scope
- **Unique Plots Surveyed**: 388
- **Time-Series Sequences**: 2,370 samples
- **Crop Types**: 21 different types
- **Spatial Resolutions**: 3m, 10m, and 30m
- **Data Preparation Strategy**: Season-based temporal sequences

## Satellite Data Sources and Specifications

### 1. Sentinel-2 (S2)
- **Spectral Bands**: 12 bands ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
- **RGB Bands**: [B4, B3, B2] (Red, Green, Blue)
- **Default Mask Resolution**: 10m
- **Default Image Size**: 32×32 pixels
- **Sensor Type**: Optical

### 2. Sentinel-1 (S1) 
- **Sensor Type**: Synthetic Aperture Radar (SAR) - Microwave
- **Data Type**: Backscatter coefficients
- **Polarizations**: VV and VH
- **Temporal Resolution**: High revisit frequency

### 3. Landsat-8 (L8)
- **Sensor Type**: Optical and Thermal
- **Spectral Bands**: Multiple bands including thermal infrared
- **Spatial Resolution**: 30m (most bands)
- **Temporal Resolution**: 16-day revisit cycle

## Data Structure and Format

### Directory Structure
```
SICKLE/
├── images/
│   ├── S1/npy/{uid}/*.npz      # Sentinel-1 data
│   ├── S2/npy/{uid}/*.npz      # Sentinel-2 data  
│   └── L8/npy/{uid}/*.npz      # Landsat-8 data
├── masks/
│   ├── 3m/{uid}.tif            # 3m resolution masks
│   ├── 10m/{uid}.tif           # 10m resolution masks
│   └── 30m/{uid}.tif           # 30m resolution masks
├── metadata/
│   └── dataframe.csv           # Plot metadata and annotations
```

### Data Format
- **Images**: Compressed numpy arrays (.npz format)
- **Masks**: Multi-band GeoTIFF files
- **Metadata**: CSV format with comprehensive plot information
- **Image Dimensions**: Configurable (default 32×32)
- **Tensor Format**: Time×Channels×Height×Width (T×C×H×W)

### Mask Layers (6 channels)
1. **Layer 0**: Plot boundary masks
2. **Layer 1**: Crop type (binary: paddy vs non-paddy)
3. **Layer 2**: Sowing date
4. **Layer 3**: Transplanting date  
5. **Layer 4**: Harvesting date
6. **Layer 5**: Crop yield

## Annotation Details

### Cropping Parameters Available
1. **Crop Type Mapping** (Classification)
   - Binary classification: Paddy vs Non-paddy
   - 21 original crop types mapped to binary
   
2. **Sowing Date Prediction** (Regression)
   - Day of year when seeds are sown
   - Range: 0-365 days

3. **Transplanting Date Prediction** (Regression)
   - Day of year when saplings are transplanted
   - Specific to paddy cultivation practices

4. **Harvesting Date Prediction** (Regression) 
   - Day of year when crops are harvested
   - End of growing season indicator

5. **Crop Yield Prediction** (Regression)
   - Quantitative yield measurements
   - Units: Agricultural standard measurements

### Data Splits
- **Training Set**: Training split with data augmentation
- **Validation Set**: Validation split for hyperparameter tuning
- **Test Set**: Held-out test set for final evaluation
- **Split Strategy**: Plot-based to prevent data leakage

## Benchmarked Tasks and Models

### Primary Tasks
1. **Binary Crop Type Mapping**
2. **Sowing Date Prediction**
3. **Transplanting Date Prediction** 
4. **Harvesting Date Prediction**
5. **Crop Yield Prediction**

### Implemented Models
1. **U-TAE** (Spatio-Temporal Attention)
   - Encoder widths: [64, 128]
   - Decoder widths: [32, 128]
   - Output conv layers: [32, 16]
   - Attention heads: 16
   - Model dimension: 256

2. **U-Net 3D** (3D Convolutional)
   - 3D convolutional architecture
   - Temporal modeling through 3D kernels

3. **ConvLSTM** (Recurrent)
   - LSTM with convolutional operations
   - Sequential temporal processing

4. **ConvGRU** (Recurrent variant)
5. **FPN** (Feature Pyramid Network)

### Model Configurations
- **Single Satellite**: [S1], [S2], [L8]
- **Multi-Satellite Fusion**: [S1,S2,L8]
- **Input Strategies**: Regional Standards (RS) vs Actual Season (AS)

## Dataset Classes and Implementation

### Main Dataset Class: `SICKLE_Dataset`
```python
SICKLE_Dataset(
    df,                    # DataFrame with metadata
    data_dir,              # Path to data directory
    satellites={           # Satellite configurations
        "S2": {
            "bands": ['B1', 'B2', ..., 'B12'],
            "rgb_bands": [3, 2, 1],
            "mask_res": 10,
            "img_size": (32, 32)
        }
    },
    ignore_index=-999,     # Ignore value for missing data
    transform=None,        # Data augmentation
    actual_season=False,   # Use actual vs standard seasons
    phase="eval"           # train/val/test phase
)
```

### Key Features
- **Multi-satellite support**: Handles multiple satellite data simultaneously
- **Flexible preprocessing**: Configurable image sizes and band selections  
- **Season handling**: Both standard and actual growing seasons
- **Data augmentation**: Training-time augmentations
- **Zero filtering**: Removes images with >25% zero pixels
- **Temporal indexing**: Day-based temporal positioning

## Data Preprocessing and Augmentations

### Preprocessing Pipeline
1. **Temporal Alignment**: Images aligned to growing season start dates
2. **Quality Filtering**: Remove images with >25% missing pixels
3. **Resizing**: Albumentation-based resizing to target dimensions
4. **Normalization**: Band-specific normalization
5. **Temporal Ordering**: Sort images by acquisition date

### Data Augmentations (Training only)
- **Horizontal Flipping**: Random horizontal flip (50% probability)
- **Vertical Flipping**: Random vertical flip (50% probability)  
- **Brightness Adjustment**: Random brightness factor (0.5-1.5x)
- **Gaussian Blur**: 3×3 kernel blur (50% probability)
- **Consistent Transforms**: Same transform applied to all satellites and masks

### Quality Control
- **Zero Percentage Check**: Filter images with excessive missing data
- **Temporal Validation**: Ensure proper temporal sequence
- **Plot Matching**: Remove unmatched plots across splits
- **Spatial Consistency**: Maintain spatial alignment across sensors

## Evaluation Metrics

### Classification Tasks (Crop Type)
- **F1-Score**: Per-class and macro-averaged
- **Accuracy**: Overall and per-class accuracy
- **IoU (Jaccard Index)**: Intersection over Union
- **Class Weights**: [0.62013, 0.37987] for imbalanced classes

### Regression Tasks (Dates and Yield)
- **RMSE**: Root Mean Square Error with ignore index handling
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

### Specialized Metrics
- **Plot-Level Aggregation**: Yield aggregated at plot level
- **Ignore Index Handling**: -999 values excluded from calculations
- **Season-Aware Evaluation**: Metrics computed within growing seasons

## Benchmarking Results

### Performance Summary
The dataset provides benchmark results for all tasks across different satellite combinations and temporal strategies. Results demonstrate:

- **Single Satellite Performance**: Individual sensor capabilities
- **Multi-Sensor Fusion Benefits**: Improved performance with sensor fusion
- **Temporal Strategy Impact**: Regional Standards vs Actual Season comparison
- **Architecture Comparison**: Relative performance of different models

### Key Findings
- Multi-sensor fusion generally outperforms single-sensor approaches
- U-TAE shows strong performance on temporal sequences
- Actual season data provides more realistic but challenging scenarios
- Sentinel-2 optical data crucial for crop type mapping
- Sentinel-1 SAR data valuable for weather-independent monitoring

## Comparison with Related Datasets

SICKLE distinguishes itself from existing agricultural datasets:

### Unique Advantages
1. **Multi-Parameter Annotations**: Unlike most datasets focusing on single tasks
2. **Multi-Sensor Integration**: Combines optical, thermal, and microwave data
3. **Fine Temporal Resolution**: High-frequency time series
4. **Real Field Validation**: Extensive ground truth collection
5. **Practical Focus**: Real-world paddy cultivation scenarios
6. **Multiple Spatial Resolutions**: 3m, 10m, and 30m annotations

### Comparison Datasets
- **SUSTAINBENCH**: Broader scope, less agricultural focus
- **Radiant ML Hub**: Different geographic regions
- **Agriculture-Vision**: Different crop types and regions
- **PixelSet**: Different temporal characteristics
- **PASTIS-R**: European focus, different crop types
- **Crop Harvest**: Global scale, less detailed annotations

## Additional Research Applications

### Benchmarked Tasks
1. Binary Crop Type Mapping
2. Sowing Date Prediction  
3. Transplanting Date Prediction
4. Harvesting Date Prediction
5. Crop Yield Prediction

### Potential Research Directions (Not Benchmarked)
1. **Panoptic Segmentation**: Instance-level crop field segmentation
2. **Synthetic Band Generation**: Cross-sensor data synthesis
3. **Image Super-Resolution**: Higher resolution prediction from low-resolution inputs
4. **Multi-Task Learning**: Joint optimization of multiple agricultural parameters
5. **Cross-Satellite Sensor Fusion**: Advanced fusion methodologies
6. **Temporal Pattern Mining**: Discovery of agricultural patterns
7. **Climate Impact Analysis**: Weather-crop interaction studies

## Technical Requirements

### Dependencies
```
opencv-python~=4.5.5
torch~=1.13.0  
torchvision~=0.14.0
torchnet~=0.0.4
albumentations~=1.3.0
rasterio~=1.3.4
torchmetrics~=0.11.0
wandb==0.13.10
geopandas~=0.12.2
tqdm~=4.64.1
seaborn~=0.13.0
```

### System Requirements
- **GPU**: CUDA-compatible GPU recommended
- **Memory**: Sufficient RAM for batch processing
- **Storage**: Substantial storage for full dataset (~several GB)
- **Python**: Python 3.7+ with PyTorch ecosystem

## Dataset Access and Usage

### Access Process
1. **Request Form**: Fill out the [access request form](https://docs.google.com/forms/d/e/1FAIpQLSdq7Dcj5FF1VmlKozrQ7XNoq006iVKrUIMTK2jReBJDuO1N2g/viewform)
2. **Dataset Download**: Access to full and toy datasets
3. **Pre-trained Weights**: Benchmark model weights provided
4. **Documentation**: Comprehensive usage documentation

### Available Resources
- **Full Dataset**: Complete multi-sensor time series
- **Toy Dataset**: Subset for testing and development
- **Pre-trained Models**: Weights for benchmark architectures
- **Evaluation Scripts**: Standard evaluation protocols
- **Tutorial Notebook**: Jupyter notebook demonstration

### Usage Workflow
1. **Data Setup**: Download and organize dataset files
2. **Environment Setup**: Install required dependencies
3. **Data Loading**: Use provided dataset classes
4. **Model Training/Evaluation**: Run provided scripts
5. **Custom Experiments**: Adapt for new research questions

## Code Architecture

### Main Components
1. **`utils/dataset.py`**: Core dataset classes and data loading
2. **`utils/transforms.py`**: Data augmentation and preprocessing  
3. **`utils/metric.py`**: Evaluation metrics and loss functions
4. **`utils/model_utils.py`**: Model architectures and utilities
5. **`train.py`**: Training pipeline and experiment management
6. **`evaluate.py`**: Evaluation pipeline and benchmarking

### Model Implementations
- **`models/utae.py`**: U-TAE spatio-temporal attention model
- **`models/unet3d.py`**: 3D U-Net convolutional architecture
- **`models/convlstm.py`**: ConvLSTM recurrent model
- **`models/convgru.py`**: ConvGRU recurrent variant
- **`models/fpn.py`**: Feature Pyramid Network

### Execution Scripts
- **`train.sh`**: Training script wrapper
- **`evaluate.sh`**: Evaluation script wrapper
- **`SICKLE_demo.ipynb`**: Demonstration notebook

## Research Impact and Applications

### Academic Contributions
- **Novel Dataset**: First multi-sensor paddy cultivation dataset
- **Benchmark Establishment**: Standard evaluation protocols
- **Multi-Task Framework**: Unified framework for agricultural tasks
- **Temporal Modeling**: Time-series agricultural analysis

### Practical Applications
- **Precision Agriculture**: Optimized farming practices
- **Crop Monitoring**: Real-time agricultural surveillance  
- **Yield Prediction**: harvest planning and market analysis
- **Policy Making**: Agricultural policy and planning support
- **Insurance**: Crop insurance and risk assessment

### Research Enablement
- **Algorithm Development**: Test bed for new architectures
- **Fusion Methods**: Multi-sensor integration research
- **Temporal Analysis**: Time-series modeling advancement
- **Agricultural AI**: Domain-specific AI development

## Citation and Acknowledgments

### Citation
```bibtex
@InProceedings{Sani_2024_WACV,
author = {Sani, Depanshu and Mahato, Sandeep and Saini, Sourabh and Agarwal, Harsh Kumar and Devshali, Charu Chandra and Anand, Saket and Arora, Gaurav and Jayaraman, Thiagarajan},
title = {SICKLE: A Multi-Sensor Satellite Imagery Dataset Annotated With Multiple Key Cropping Parameters},
booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
month = {January},
year = {2024},
pages = {5995-6004}
}
```

### Acknowledgments
- **Google AI for Social Good**: "Impact Scholars" program support
- **Infosys Center for Artificial Intelligence**: IIIT-Delhi support
- **MS Swaminathan Research Foundation**: Field expertise and data collection
- **Research Team**: IIIT-Delhi computer vision and agricultural experts
- **Field Partners**: Local agricultural experts and farmers

### Contact Information
- **Code-related queries**: sourabh19113@iiitd.ac.in (GitHub issues preferred)
- **General inquiries**: depanshus@iiitd.ac.in
- **Project Website**: [SICKLE Project Site](https://sites.google.com/iiitd.ac.in/sickle/home)

## Future Directions

### Dataset Enhancements
- **Temporal Extension**: Additional years of data
- **Geographic Expansion**: Other agricultural regions
- **Crop Diversity**: Additional crop types and varieties
- **Higher Resolution**: Finer spatial resolution annotations

### Technical Improvements  
- **Real-Time Processing**: Streaming data pipelines
- **Edge Computing**: On-device agricultural monitoring
- **Automated Annotation**: AI-assisted ground truth generation
- **Uncertainty Quantification**: Confidence estimation in predictions

### Application Domains
- **Climate Change**: Agricultural adaptation studies
- **Sustainability**: Environmental impact assessment  
- **Food Security**: Global food production monitoring
- **Policy Research**: Evidence-based agricultural policies

---

This comprehensive documentation provides a complete overview of the SICKLE dataset, its structure, implementation, and research potential. The dataset represents a significant contribution to agricultural remote sensing and provides a robust foundation for advancing AI applications in agriculture.