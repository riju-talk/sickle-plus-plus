import ee
import numpy as np
from typing import Dict, List, Tuple, Optional


class FeatureEngineer:
    """
    Feature engineering for satellite imagery data.
    Computes vegetation indices, SAR ratios, and other derived features.
    """
    
    @staticmethod
    def compute_agricultural_indices(image: ee.Image, sickle_compatible: bool = True) -> ee.Image:
        """
        Compute agricultural indices optimized for crop monitoring (SICKLE-compatible).
        
        Args:
            image: Earth Engine image with optical bands
            sickle_compatible: Use SICKLE dataset agricultural indices
            
        Returns:
            Image with agricultural indices added
        """
        band_names = image.bandNames()
        
        # Flexible band selection for different naming conventions
        def safe_select(band_options):
            for option in band_options:
                if band_names.contains(option).getInfo():
                    return image.select(option)
            return None
        
        # Get bands with multiple naming options
        red = safe_select(['red', 'B4', 'SR_B4'])
        nir = safe_select(['nir', 'B8', 'SR_B5'])
        green = safe_select(['green', 'B3', 'SR_B3'])
        blue = safe_select(['blue', 'B2', 'SR_B2'])
        rededge1 = safe_select(['rededge1', 'B5'])
        rededge2 = safe_select(['rededge2', 'B6'])  
        rededge3 = safe_select(['rededge3', 'B7'])
        nir_narrow = safe_select(['nir_narrow', 'B8A'])
        swir1 = safe_select(['swir1', 'B11', 'SR_B6'])
        swir2 = safe_select(['swir2', 'B12', 'SR_B7'])
        
        indices = image
        
        if sickle_compatible and red and nir:
            # Core SICKLE agricultural indices
            
            # NDVI - Primary vegetation index for crop monitoring
            ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi')
            indices = indices.addBands(ndvi)
            
            # GNDVI - Green NDVI for chlorophyll content
            if green:
                gndvi = nir.subtract(green).divide(nir.add(green)).rename('gndvi')
                indices = indices.addBands(gndvi)
            
            # NDRE - Red Edge NDVI for crop stress detection
            if rededge1:
                ndre = nir.subtract(rededge1).divide(nir.add(rededge1)).rename('ndre')
                indices = indices.addBands(ndre)
            
            # NDWI - Normalized Difference Water Index for crop water content
            if green and swir1:
                ndwi = green.subtract(swir1).divide(green.add(swir1)).rename('ndwi')
                indices = indices.addBands(ndwi)
                
            # EVI - Enhanced Vegetation Index for canopy structure
            if green and blue:
                evi = nir.subtract(red).divide(
                    nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
                ).multiply(2.5).rename('evi')
                indices = indices.addBands(evi)
                
            # SAVI - Soil-Adjusted Vegetation Index
            L = 0.5  # Soil brightness correction factor
            savi = nir.subtract(red).divide(
                nir.add(red).add(L)
            ).multiply(1 + L).rename('savi')
            indices = indices.addBands(savi)
            
            # Red Edge Indices for crop health monitoring
            if rededge1 and rededge2:
                # Red Edge Position Index
                reci = nir.divide(rededge1).subtract(1).rename('reci')
                indices = indices.addBands(reci)
                
            # Crop-specific indices
            if swir1 and swir2:
                # NBR - Normalized Burn Ratio (for crop senescence)
                nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename('nbr')
                indices = indices.addBands(nbr)
                
                # NDII - Normalized Difference Infrared Index (moisture)
                ndii = nir.subtract(swir1).divide(nir.add(swir1)).rename('ndii')
                indices = indices.addBands(ndii)
        
        else:
            # Standard vegetation indices
            if red and nir:
                ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi')
                indices = indices.addBands(ndvi)
                
                if green:
                    gndvi = nir.subtract(green).divide(nir.add(green)).rename('gndvi')
                    indices = indices.addBands(gndvi)
        
        return indices
    
    @staticmethod
    def compute_crop_sar_indices(image: ee.Image, sickle_compatible: bool = True) -> ee.Image:
        """
        Compute SAR-based indices optimized for crop monitoring (SICKLE-compatible).
        
        Args:
            image: Earth Engine image with SAR bands (VV, VH)
            sickle_compatible: Use SICKLE dataset SAR processing
            
        Returns:
            Image with crop SAR indices added
        """
        band_names = image.bandNames()
        
        # Check for VV and VH bands
        vv = None
        vh = None
        
        if band_names.contains('vv').getInfo():
            vv = image.select('vv')
        if band_names.contains('vh').getInfo():
            vh = image.select('vh')
        
        indices = image
        
        if vv and vh and sickle_compatible:
            # SICKLE-style agricultural SAR indices
            
            # VV/VH ratio - Primary crop structure indicator
            vv_vh_ratio = vv.divide(vh).rename('vv_vh_ratio')
            indices = indices.addBands(vv_vh_ratio)
            
            # Normalized Difference SAR Index
            ndsar = vv.subtract(vh).divide(vv.add(vh)).rename('ndsar')
            indices = indices.addBands(ndsar)
            
            # Radar Vegetation Index (RVI) - Crop biomass indicator
            rvi = vh.multiply(4).divide(vv.add(vh)).rename('rvi')
            indices = indices.addBands(rvi)
            
            # Dual Pol SAR Vegetation Index
            dpsvi = vh.divide(vv.add(vh)).rename('dpsvi') 
            indices = indices.addBands(dpsvi)
            
            # Cross-polarization ratio for volume scattering
            if band_names.contains('cross_pol').getInfo():
                cross_pol = image.select('cross_pol')
                cross_ratio = cross_pol.divide(vv.add(vh)).rename('cross_ratio')
                indices = indices.addBands(cross_ratio)
        
        elif vv and vh:
            # Standard SAR indices
            # VV/VH ratio
            ratio = vv.divide(vh).rename('vv_vh_ratio')
            indices = indices.addBands(ratio)
            
            # Difference
            diff = vv.subtract(vh).rename('vv_vh_diff')
            indices = indices.addBands(diff)
            
            # Normalized difference
            ndsar = vv.subtract(vh).divide(vv.add(vh)).rename('ndsar')
            indices = indices.addBands(ndsar)
            
            # RVI - Radar Vegetation Index
            rvi = vh.multiply(4).divide(vv.add(vh)).rename('rvi')
            indices = indices.addBands(rvi)
        
        return indices
    
    @staticmethod
    def compute_texture_features(image: ee.Image, band: str = 'nir', radius: int = 2) -> ee.Image:
        """
        Compute texture features using GLCM.
        
        Args:
            image: Input image
            band: Band to compute texture on
            radius: Neighborhood radius
            
        Returns:
            Image with texture features
        """
        # GLCM texture
        glcm = image.select(band).glcmTexture(size=radius)
        
        # Select relevant texture metrics
        texture_bands = [
            f"{band}_contrast",
            f"{band}_corr", 
            f"{band}_var",
            f"{band}_idm",
            f"{band}_savg",
            f"{band}_ent"
        ]
        
        # Add texture bands if they exist
        texture = glcm.select(texture_bands, 
                             ['contrast', 'correlation', 'variance', 
                              'homogeneity', 'mean', 'entropy'])
        
        return image.addBands(texture)
    
    @staticmethod
    def compute_spectral_transformations(image: ee.Image) -> ee.Image:
        """
        Compute spectral transformations like PCA and tasseled cap.
        
        Args:
            image: Input image with optical bands
            
        Returns:
            Image with spectral transformations
        """
        band_names = image.bandNames()
        
        # Try to get standard optical bands
        optical_bands = []
        standard_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        
        for name in standard_names:
            if band_names.contains(name).getInfo():
                optical_bands.append(name)
        
        if len(optical_bands) >= 3:
            optical_image = image.select(optical_bands)
            
            # Tasseled Cap Transformation (for Landsat-like bands)
            if len(optical_bands) >= 6:
                # Coefficients for Landsat 8 (approximate)
                brightness_coeffs = [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872]
                greenness_coeffs = [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608]
                wetness_coeffs = [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559]
                
                brightness = optical_image.expression(
                    'b * b_c + g * g_c + r * r_c + nir * nir_c + s1 * s1_c + s2 * s2_c',
                    {
                        'b': optical_image.select(optical_bands[0]),
                        'g': optical_image.select(optical_bands[1]),
                        'r': optical_image.select(optical_bands[2]),
                        'nir': optical_image.select(optical_bands[3]),
                        's1': optical_image.select(optical_bands[4]),
                        's2': optical_image.select(optical_bands[5]),
                        'b_c': brightness_coeffs[0],
                        'g_c': brightness_coeffs[1],
                        'r_c': brightness_coeffs[2],
                        'nir_c': brightness_coeffs[3],
                        's1_c': brightness_coeffs[4],
                        's2_c': brightness_coeffs[5]
                    }
                ).rename('brightness')
                
                greenness = optical_image.expression(
                    'b * b_c + g * g_c + r * r_c + nir * nir_c + s1 * s1_c + s2 * s2_c',
                    {
                        'b': optical_image.select(optical_bands[0]),
                        'g': optical_image.select(optical_bands[1]),
                        'r': optical_image.select(optical_bands[2]),
                        'nir': optical_image.select(optical_bands[3]),
                        's1': optical_image.select(optical_bands[4]),
                        's2': optical_image.select(optical_bands[5]),
                        'b_c': greenness_coeffs[0],
                        'g_c': greenness_coeffs[1],
                        'r_c': greenness_coeffs[2],
                        'nir_c': greenness_coeffs[3],
                        's1_c': greenness_coeffs[4],
                        's2_c': greenness_coeffs[5]
                    }
                ).rename('greenness')
                
                wetness = optical_image.expression(
                    'b * b_c + g * g_c + r * r_c + nir * nir_c + s1 * s1_c + s2 * s2_c',
                    {
                        'b': optical_image.select(optical_bands[0]),
                        'g': optical_image.select(optical_bands[1]),
                        'r': optical_image.select(optical_bands[2]),
                        'nir': optical_image.select(optical_bands[3]),
                        's1': optical_image.select(optical_bands[4]),
                        's2': optical_image.select(optical_bands[5]),
                        'b_c': wetness_coeffs[0],
                        'g_c': wetness_coeffs[1],
                        'r_c': wetness_coeffs[2],
                        'nir_c': wetness_coeffs[3],
                        's1_c': wetness_coeffs[4],
                        's2_c': wetness_coeffs[5]
                    }
                ).rename('wetness')
                
                image = image.addBands([brightness, greenness, wetness])
        
        return image
    
    @staticmethod
    def create_agricultural_feature_stack(image: ee.Image,
                                        include_vegetation: bool = True,
                                        include_sar: bool = True,
                                        include_texture: bool = False,
                                        include_spectral: bool = False,
                                        sickle_compatible: bool = True,
                                        crop_focus: bool = True) -> ee.Image:
        """
        Create agricultural feature stack optimized for crop monitoring (SICKLE-compatible).
        
        Args:
            image: Input Earth Engine image
            include_vegetation: Include agricultural vegetation indices
            include_sar: Include crop SAR indices
            include_texture: Include texture features
            include_spectral: Include spectral transformations
            sickle_compatible: Use SICKLE dataset specifications
            crop_focus: Apply crop-specific processing
            
        Returns:
            Agricultural feature stack image
        """
        feature_image = image
        
        if include_vegetation:
            feature_image = FeatureEngineer.compute_agricultural_indices(
                feature_image, sickle_compatible=sickle_compatible
            )
        
        if include_sar:
            feature_image = FeatureEngineer.compute_crop_sar_indices(
                feature_image, sickle_compatible=sickle_compatible
            )
        
        if include_texture and crop_focus:
            # Apply texture analysis on key agricultural bands
            band_names = feature_image.bandNames()
            if band_names.contains('ndvi').getInfo():
                feature_image = FeatureEngineer.compute_texture_features(
                    feature_image, 'ndvi', radius=3
                )
            elif band_names.contains('nir').getInfo():
                feature_image = FeatureEngineer.compute_texture_features(
                    feature_image, 'nir', radius=3
                )
        
        if include_spectral:
            feature_image = FeatureEngineer.compute_spectral_transformations(feature_image)
        
        if sickle_compatible and crop_focus:
            # Add crop-specific temporal features if multi-temporal
            feature_image = FeatureEngineer._add_crop_temporal_features(feature_image)
        
        return feature_image
    
    @staticmethod
    def _add_crop_temporal_features(image: ee.Image) -> ee.Image:
        """
        Add crop-specific temporal features for growing season analysis.
        """
        # This would be expanded for multi-temporal analysis
        # Currently returns image unchanged for single-time analysis
        return image
    
    @staticmethod
    def apply_sickle_quality_control(image: ee.Image, 
                                   zero_threshold: float = 0.25) -> ee.Image:
        """
        Apply SICKLE-style quality control filtering.
        
        Args:
            image: Input image
            zero_threshold: Maximum fraction of zero pixels allowed (SICKLE uses 0.25)
            
        Returns:
            Quality-controlled image
        """
        # Create a mask for non-zero pixels
        non_zero_mask = image.neq(0).reduce(ee.Reducer.allNonZero())
        
        # Apply mask to remove images with >25% zero pixels (SICKLE standard)
        return image.updateMask(non_zero_mask)
    
    @staticmethod
    def normalize_features(image: ee.Image, method: str = 'minmax') -> ee.Image:
        """
        Normalize features in the image.
        
        Args:
            image: Input image
            method: Normalization method ('minmax', 'zscore')
            
        Returns:
            Normalized image
        """
        if method == 'minmax':
            # Min-max normalization (0-1)
            min_vals = image.reduceRegion(
                reducer=ee.Reducer.min(),
                scale=30,
                maxPixels=1e9
            )
            max_vals = image.reduceRegion(
                reducer=ee.Reducer.max(),
                scale=30,
                maxPixels=1e9
            )
            
            normalized = image.subtract(ee.Image.constant(min_vals)).divide(
                ee.Image.constant(max_vals).subtract(ee.Image.constant(min_vals))
            )
            
        elif method == 'zscore':
            # Z-score normalization
            mean = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                scale=30,
                maxPixels=1e9
            )
            std = image.reduceRegion(
                reducer=ee.Reducer.stdDev(),
                scale=30,
                maxPixels=1e9
            )
            
            normalized = image.subtract(ee.Image.constant(mean)).divide(
                ee.Image.constant(std)
            )
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    @staticmethod
    def to_numpy_features(image: ee.Image,
                         geometry: ee.Geometry,
                         scale: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Convert Earth Engine image to numpy array for ML processing.
        
        Args:
            image: Feature image
            geometry: Region of interest
            scale: Pixel resolution
            
        Returns:
            Tuple of (feature_array, band_names)
        """
        # Get band names
        band_names = image.bandNames().getInfo()
        
        # Sample the image
        sample = image.sampleRectangle(
            region=geometry,
            defaultValue=0,
            properties=[],
            maxPixels=1e8
        )
        
        # Convert to numpy arrays
        arrays = []
        for band in band_names:
            band_data = np.array(sample.get(band).getInfo())
            arrays.append(band_data)
        
        # Stack into 3D array (height, width, bands)
        feature_array = np.stack(arrays, axis=-1)
        
        print(f"Feature array shape: {feature_array.shape}")
        print(f"Feature bands: {band_names}")
        
        return feature_array, band_names