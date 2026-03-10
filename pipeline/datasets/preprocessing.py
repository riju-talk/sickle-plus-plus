"""
Satellite data preprocessing for SICKLE++ baseline.

Handles:
- Cloud masking for optical data
- Temporal sequence construction
- Data normalization and quality control
- Multi-sensor fusion
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime
import ee

logger = logging.getLogger(__name__)


class CloudMasking:
    """Cloud and shadow masking for optical satellite data."""
    
    @staticmethod
    def mask_sentinel2_clouds(image: np.ndarray, 
                             qa_band: Optional[np.ndarray] = None,
                             cloud_threshold: float = 20.0) -> np.ndarray:
        """
        Mask clouds and shadows in Sentinel-2 data.
        
        Args:
            image: Image array (C, H, W)
            qa_band: QA60 band with cloud flags
            cloud_threshold: Cloud probability threshold
            
        Returns:
            Masked image array
        """
        if qa_band is None:
            # Use NDVI-based cloud detection
            return CloudMasking._mask_clouds_ndvi(image)
        
        # Use QA60 band if available
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        
        # Create mask (valid pixels = 0)
        mask = (qa_band & cloud_bit_mask == 0) & (qa_band & cirrus_bit_mask == 0)
        
        # Apply mask to all bands
        for c in range(image.shape[0]):
            image[c][~mask] = 0
        
        return image
    
    @staticmethod
    def _mask_clouds_ndvi(image: np.ndarray) -> np.ndarray:
        """
        Cloud detection using NDVI (for preprocessing).
        Clouds typically have NDVI < 0.3
        """
        # Find red (B4, index 3) and NIR (B8, index 7)
        if image.shape[0] >= 8:
            red = image[3]
            nir = image[7]
            
            # Calculate NDVI
            ndvi = (nir - red) / (nir + red + 1e-8)
            
            # Clouds have low NDVI and high reflectance
            bright = np.mean(image[[2, 3, 4]], axis=0) > 0.3  # Blue, green, red bright
            low_ndvi = ndvi < 0.3
            
            cloud_mask = bright & low_ndvi
            
            # Apply mask
            for c in range(image.shape[0]):
                image[c][cloud_mask] = 0
        
        return image
    
    @staticmethod
    def mask_by_validity(image: np.ndarray, 
                        zero_threshold: float = 0.25) -> Tuple[np.ndarray, float]:
        """
        Mask pixels with too many zero values (SICKLE quality control).
        
        Args:
            image: Image array (C, H, W)
            zero_threshold: Max fraction of zero bands allowed (SICKLE uses 0.25)
            
        Returns:
            Masked image and validity percentage
        """
        # Count zero pixels per band
        zero_counts = np.sum(image == 0, axis=(1, 2)) / (image.shape[1] * image.shape[2])
        
        # Create validity mask: keep pixels with <25% zero bands
        zero_fraction = np.zeros((image.shape[1], image.shape[2]))
        for c in range(image.shape[0]):
            zero_fraction += (image[c] == 0).astype(float)
        zero_fraction /= image.shape[0]
        
        valid_mask = zero_fraction <= zero_threshold
        validity = np.sum(valid_mask) / valid_mask.size
        
        # Apply mask
        image[:, ~valid_mask] = 0
        
        return image, validity


class SatelliteDataPreprocessor:
    """
    Multi-sensor satellite data preprocessing for SICKLE++ baseline.
    """
    
    def __init__(self, 
                 sentinel2_bands: Optional[List[str]] = None,
                 sentinel1_bands: Optional[List[str]] = None,
                 landsat8_bands: Optional[List[str]] = None):
        """
        Initialize preprocessor with band configurations.
        
        Args:
            sentinel2_bands: List of S2 bands to use
            sentinel1_bands: List of S1 bands to use
            landsat8_bands: List of L8 bands to use
        """
        # SICKLE standard bands
        self.sentinel2_bands = sentinel2_bands or [
            'coastal', 'blue', 'green', 'red', 'rededge1', 'rededge2',
            'rededge3', 'nir', 'nir_narrow', 'watervapor', 'swir1', 'swir2'
        ]
        
        self.sentinel1_bands = sentinel1_bands or ['vv', 'vh']
        
        self.landsat8_bands = landsat8_bands or [
            'coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2'
        ]
    
    @staticmethod
    def stack_multi_sensor(s2_data: Optional[np.ndarray] = None,
                          s1_data: Optional[np.ndarray] = None,
                          l8_data: Optional[np.ndarray] = None,
                          fusion_type: str = 'early') -> np.ndarray:
        """
        Stack multi-sensor data.
        
        Args:
            s2_data: Sentinel-2 array (C, H, W) - 12 bands
            s1_data: Sentinel-1 array (C, H, W) - 2 bands
            l8_data: Landsat-8 array (C, H, W) - 7+ bands
            fusion_type: 'early' (concatenate channels), 'dict' (separate)
            
        Returns:
            Stacked array shape (C_total, H, W) for early fusion
            or Dict[sensor_name -> array] for late fusion
        """
        valid_arrays = []
        
        if s2_data is not None:
            valid_arrays.append(('sentinel2', s2_data))
        if s1_data is not None:
            valid_arrays.append(('sentinel1', s1_data))
        if l8_data is not None:
            valid_arrays.append(('landsat8', l8_data))
        
        if not valid_arrays:
            raise ValueError("At least one sensor data must be provided")
        
        if fusion_type == 'early':
            # Concatenate along channel dimension
            stacked = np.concatenate([arr for _, arr in valid_arrays], axis=0)
            return stacked
        
        elif fusion_type == 'dict':
            # Return as dictionary
            return dict(valid_arrays)
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    @staticmethod
    def normalize_sickle(data: np.ndarray, 
                        method: str = 'minmax',
                        percentile_range: Tuple[float, float] = (2, 98)) -> np.ndarray:
        """
        Apply SICKLE-style normalization to satellite data.
        
        Args:
            data: Input data (C, H, W) or (T, C, H, W)
            method: 'minmax' or 'zscore'
            percentile_range: Percentiles for robust normalization
            
        Returns:
            Normalized data in [0, 1] or standardized
        """
        if len(data.shape) == 4:  # (T, C, H, W)
            for t in range(data.shape[0]):
                data[t] = SatelliteDataPreprocessor.normalize_sickle(
                    data[t], method, percentile_range
                )
            return data
        
        # (C, H, W) case
        if method == 'minmax':
            normalized = np.zeros_like(data, dtype=np.float32)
            
            for c in range(data.shape[0]):
                band = data[c]
                
                # Use percentiles to handle outliers (SICKLE approach)
                p_min, p_max = np.percentile(band, percentile_range)
                
                if p_max > p_min:
                    normalized[c] = (band - p_min) / (p_max - p_min)
                else:
                    normalized[c] = band
                
                # Clip to [0, 1]
                normalized[c] = np.clip(normalized[c], 0, 1)
            
            return normalized
        
        elif method == 'zscore':
            normalized = np.zeros_like(data, dtype=np.float32)
            
            for c in range(data.shape[0]):
                band = data[c]
                mean = np.mean(band)
                std = np.std(band) + 1e-8
                normalized[c] = (band - mean) / std
            
            return normalized
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def create_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate NDVI vegetation index."""
        ndvi = (nir - red) / (nir + red + 1e-8)
        return np.clip(ndvi, -1, 1)
    
    @staticmethod
    def create_indices_stack(data: np.ndarray) -> np.ndarray:
        """
        Create agricultural indices from multi-spectral data.
        Expects S2 12-band data with standard ordering.
        
        Returns:
            Data with indices appended along channel dim
        """
        indices_list = []
        
        if data.shape[0] >= 12:
            # Sentinel-2 bands (assuming standard order)
            # 0: coastal, 1: blue, 2: green, 3: red, 4: rededge1, 5: rededge2
            # 6: rededge3, 7: nir, 8: nir_narrow, 9: watervapor, 10: swir1, 11: swir2
            
            red = data[3]
            nir = data[7]
            green = data[2]
            swir1 = data[10]
            swir2 = data[11]
            rededge1 = data[4]
            
            # NDVI
            ndvi = (nir - red) / (nir + red + 1e-8)
            indices_list.append(ndvi)
            
            # GNDVI - Green NDVI
            gndvi = (nir - green) / (nir + green + 1e-8)
            indices_list.append(gndvi)
            
            # NDRE - Red Edge NDVI
            ndre = (nir - rededge1) / (nir + rededge1 + 1e-8)
            indices_list.append(ndre)
            
            # NDWI - Water Index
            ndwi = (green - swir1) / (green + swir1 + 1e-8)
            indices_list.append(ndwi)
            
            # NBR - Normalized Burn Ratio (Senescence)
            nbr = (nir - swir2) / (nir + swir2 + 1e-8)
            indices_list.append(nbr)
            
            # SAVI - Soil-Adjusted VI
            L = 0.5
            savi = (nir - red) / (nir + red + L) * (1 + L)
            indices_list.append(savi)
            
            if indices_list:
                indices = np.stack(indices_list, axis=0)
                data = np.concatenate([data, indices], axis=0)
        
        return data


class TemporalSequenceBuilder:
    """Build temporal sequences from multi-temporal satellite data."""
    
    @staticmethod
    def create_time_series(data_list: List[np.ndarray],
                          dates: List[str],
                          target_length: int = 61,
                          interpolation: str = 'linear') -> Tuple[np.ndarray, List[str]]:
        """
        Create temporal sequence from ordered list of images.
        
        Args:
            data_list: List of image arrays (C, H, W)
            dates: Corresponding dates (YYYY-MM-DD)
            target_length: Target sequence length
            interpolation: How to handle missing timesteps
            
        Returns:
            Temporal array (T, C, H, W) and interpolated dates
        """
        if len(data_list) < 2:
            raise ValueError("Need at least 2 timesteps")
        
        # Stack available data
        stacked = np.stack(data_list, axis=0)  # (T, C, H, W)
        
        if len(data_list) >= target_length:
            # Subsample if too many timesteps
            indices = np.linspace(0, len(data_list) - 1, target_length, dtype=int)
            stacked = stacked[indices]
            dates = [dates[i] for i in indices]
        
        elif len(data_list) < target_length and interpolation == 'linear':
            # Interpolate missing timesteps
            stacked = TemporalSequenceBuilder._interpolate_temporal(stacked, target_length)
            
            # Generate interpolated dates
            from datetime import datetime, timedelta
            start_date = datetime.strptime(dates[0], '%Y-%m-%d')
            end_date = datetime.strptime(dates[-1], '%Y-%m-%d')
            
            date_range = (end_date - start_date).days
            
            dates = [
                (start_date + timedelta(days=date_range * i / (target_length - 1))).strftime('%Y-%m-%d')
                for i in range(target_length)
            ]
        
        else:
            # Repeat last frame to reach target length
            while stacked.shape[0] < target_length:
                stacked = np.vstack([stacked, stacked[-1:]])
        
        return stacked, dates
    
    @staticmethod
    def _interpolate_temporal(data: np.ndarray, target_length: int) -> np.ndarray:
        """
        Interpolate temporal dimension to target length.
        
        Args:
            data: Input array (T, C, H, W)
            target_length: Target T dimension
            
        Returns:
            Interpolated array
        """
        T, C, H, W = data.shape
        
        # Create output array
        output = np.zeros((target_length, C, H, W), dtype=data.dtype)
        
        # Compute interpolation indices
        indices = np.linspace(0, T - 1, target_length)
        
        # Linear interpolation for each channel
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    output[:, c, h, w] = np.interp(
                        indices,
                        np.arange(T),
                        data[:, c, h, w]
                    )
        
        return output
    
    @staticmethod
    def fill_temporal_gaps(data: np.ndarray,
                          missing_mask: np.ndarray) -> np.ndarray:
        """
        Fill temporal gaps (missing observations) using forward fill.
        
        Args:
            data: Temporal array (T, C, H, W)
            missing_mask: Boolean mask where True = missing
            
        Returns:
            Array with gaps filled
        """
        T, C, H, W = data.shape
        filled = data.copy()
        
        # Forward fill along temporal dimension
        for t in range(T):
            if missing_mask[t]:
                # Find last valid frame
                for prev_t in range(t - 1, -1, -1):
                    if not missing_mask[prev_t]:
                        filled[t] = filled[prev_t]
                        break
        
        return filled


class SICKLEPreprocessingPipeline:
    """
    Complete preprocessing pipeline following SICKLE dataset standard.
    """
    
    def __init__(self,
                 cloud_threshold: float = 20.0,
                 normalization: str = 'minmax',
                 include_indices: bool = True,
                 quality_control: bool = True):
        """
        Initialize preprocessing pipeline.
        
        Args:
            cloud_threshold: Cloud percentage threshold
            normalization: Normalization method
            include_indices: Compute agricultural indices
            quality_control: Apply SICKLE quality filtering
        """
        self.cloud_threshold = cloud_threshold
        self.normalization = normalization
        self.include_indices = include_indices
        self.quality_control = quality_control
        self.preprocessor = SatelliteDataPreprocessor()
    
    def process_single_temporal(self,
                               sentinel2: Optional[np.ndarray] = None,
                               sentinel1: Optional[np.ndarray] = None,
                               landsat8: Optional[np.ndarray] = None) -> Dict:
        """
        Process single temporal observation from multiple sensors.
        
        Args:
            sentinel2: S2 data (12, H, W)
            sentinel1: S1 data (2, H, W)
            landsat8: L8 data (7+, H, W)
            
        Returns:
            Dict with processed data and metadata
        """
        results = {
            'raw_shapes': {},
            'processed_data': {},
            'validity_scores': {}
        }
        
        # Process Sentinel-2
        if sentinel2 is not None:
            # Cloud masking
            if self.quality_control:
                sentinel2_masked, validity_s2 = CloudMasking.mask_by_validity(sentinel2)
                results['validity_scores']['sentinel2'] = float(validity_s2)
            else:
                sentinel2_masked = sentinel2.copy()
            
            # Normalization
            if self.normalization:
                sentinel2_masked = self.preprocessor.normalize_sickle(
                    sentinel2_masked, self.normalization
                )
            
            # Add indices
            if self.include_indices:
                sentinel2_masked = self.preprocessor.create_indices_stack(sentinel2_masked)
            
            results['processed_data']['sentinel2'] = sentinel2_masked
            results['raw_shapes']['sentinel2'] = sentinel2.shape
        
        # Process Sentinel-1
        if sentinel1 is not None:
            # Normalize SAR data
            if self.normalization:
                sentinel1_norm = self.preprocessor.normalize_sickle(
                    sentinel1.copy(), self.normalization
                )
            else:
                sentinel1_norm = sentinel1.copy()
            
            results['processed_data']['sentinel1'] = sentinel1_norm
            results['raw_shapes']['sentinel1'] = sentinel1.shape
        
        # Stack multi-sensor data
        data_to_stack = []
        if 'sentinel2' in results['processed_data']:
            data_to_stack.append(results['processed_data']['sentinel2'])
        if 'sentinel1' in results['processed_data']:
            data_to_stack.append(results['processed_data']['sentinel1'])
        
        if data_to_stack:
            stacked = np.concatenate(data_to_stack, axis=0)
            results['stacked_data'] = stacked
            results['stacked_shape'] = stacked.shape
            results['total_bands'] = stacked.shape[0]
        
        return results
    
    def process_temporal_sequence(self,
                                 data_sequence: List[Dict],
                                 target_length: int = 61) -> Dict:
        """
        Process entire temporal sequence.
        
        Args:
            data_sequence: List of dicts with 's2', 's1', 'l8' keys
            target_length: Target sequence length (PASTIS/SICKLE standard)
            
        Returns:
            Processed temporal sequence
        """
        processed_frames = []
        metadata = []
        
        # Process each frame
        for frame_data in data_sequence:
            results = self.process_single_temporal(
                sentinel2=frame_data.get('s2'),
                sentinel1=frame_data.get('s1'),
                landsat8=frame_data.get('l8')
            )
            
            if 'stacked_data' in results:
                processed_frames.append(results['stacked_data'])
                metadata.append({
                    'validity': results['validity_scores'],
                    'shape': results['stacked_shape']
                })
        
        if not processed_frames:
            raise ValueError("No valid frames to process")
        
        # Create temporal sequence
        sequence_array = np.stack(processed_frames, axis=0)
        
        # Resize to target length
        if sequence_array.shape[0] < target_length:
            sequence_array, _ = TemporalSequenceBuilder.create_time_series(
                [sequence_array[i] for i in range(sequence_array.shape[0])],
                [f"t_{i}" for i in range(sequence_array.shape[0])],
                target_length=target_length
            )
        elif sequence_array.shape[0] > target_length:
            indices = np.linspace(0, sequence_array.shape[0] - 1, target_length, dtype=int)
            sequence_array = sequence_array[indices]
        
        return {
            'sequence': sequence_array,
            'shape': sequence_array.shape,
            'target_length': target_length,
            'n_frames': len(data_sequence),
            'metadata': metadata
        }
