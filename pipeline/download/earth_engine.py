import ee
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import datetime
import os


class EarthEngineDownloader:
    """
    Earth Engine downloader for multi-sensor satellite data.
    """
    
    def __init__(self, initialize_ee: bool = True):
        """Initialize Earth Engine connection."""
        if initialize_ee:
            try:
                ee.Initialize()
                print("Earth Engine initialized successfully")
            except Exception as e:
                print(f"Error initializing Earth Engine: {e}")
                print("Please run: earthengine authenticate")
                raise
    
    def download_sentinel2(self, 
                          geometry: ee.Geometry,
                          start_date: str,
                          end_date: str,
                          cloud_threshold: float = 20.0,
                          scale: int = 10,
                          sickle_compatible: bool = True) -> ee.Image:
        """
        Download Sentinel-2 surface reflectance data with SICKLE-compatible bands.
        
        Args:
            geometry: Earth Engine geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_threshold: Maximum cloud cover percentage
            scale: Pixel resolution in meters
            sickle_compatible: Use SICKLE dataset band configuration
            
        Returns:
            Earth Engine image with selected bands
        """
        collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)))
        
        if sickle_compatible:
            # SICKLE dataset uses these 12 specific bands for agricultural analysis
            bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
            band_names = ['coastal', 'blue', 'green', 'red', 'rededge1', 'rededge2', 
                         'rededge3', 'nir', 'nir_narrow', 'watervapor', 'swir1', 'swir2']
        else:
            # Standard bands: Blue, Green, Red, NIR, SWIR1, SWIR2  
            bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
            band_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        
        # Apply cloud masking
        def mask_clouds(image):
            qa = image.select('QA60')
            cloud_bit_mask = 1 << 10
            cirrus_bit_mask = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                   qa.bitwiseAnd(cirrus_bit_mask).eq(0))
            return image.updateMask(mask).select(bands).rename(band_names)
        
        # Get median composite
        composite = collection.map(mask_clouds).median().clip(geometry)
        
        print(f"Sentinel-2 collection size: {collection.size().getInfo()}")
        return composite
    
    def download_sentinel1(self,
                          geometry: ee.Geometry,
                          start_date: str,
                          end_date: str,
                          orbit: str = 'BOTH',
                          scale: int = 10,
                          sickle_compatible: bool = True) -> ee.Image:
        """
        Download Sentinel-1 GRD data with SICKLE-compatible configuration.
        
        Args:
            geometry: Earth Engine geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            orbit: Orbit direction ('ASCENDING', 'DESCENDING', 'BOTH')
            scale: Pixel resolution in meters
            sickle_compatible: Use SICKLE dataset SAR configuration
            
        Returns:
            Earth Engine image with VV and VH bands
        """
        collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                     .filter(ee.Filter.eq('instrumentMode', 'IW')))
        
        if orbit != 'BOTH':
            collection = collection.filter(ee.Filter.eq('orbitProperties_pass', orbit))
        
        # SICKLE-style speckle filtering for agricultural applications
        def apply_agricultural_sar_processing(image):
            # Speckle filtering - SICKLE uses median filtering
            vv = image.select('VV').focal_median(50, 'circle', 'meters')
            vh = image.select('VH').focal_median(50, 'circle', 'meters')
            
            if sickle_compatible:
                # Agricultural-specific processing
                # Add VV/VH ratio for crop structure analysis
                ratio = vv.divide(vh).rename('VV_VH_ratio')
                # Add cross-polarization for volume scattering
                cross_pol = vh.subtract(vv).rename('cross_pol')
                return image.addBands([vv.rename('vv'), vh.rename('vh'), ratio, cross_pol])
            else:
                return image.addBands(vv.rename('VV_filtered')).addBands(vh.rename('VH_filtered'))
        
        # Get median composite with agricultural processing
        composite = collection.map(apply_agricultural_sar_processing).median()
        
        if sickle_compatible:
            composite = composite.select(['vv', 'vh', 'VV_VH_ratio', 'cross_pol']).clip(geometry)
        else:
            composite = composite.select(['VV_filtered', 'VH_filtered'], ['vv', 'vh']).clip(geometry)
        
        print(f"Sentinel-1 collection size: {collection.size().getInfo()}")
        return composite
    
    def download_landsat8(self,
                         geometry: ee.Geometry,
                         start_date: str,
                         end_date: str,
                         cloud_threshold: float = 20.0,
                         scale: int = 30) -> ee.Image:
        """
        Download Landsat 8 surface reflectance data.
        
        Args:
            geometry: Earth Engine geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_threshold: Maximum cloud cover percentage
            scale: Pixel resolution in meters
            
        Returns:
            Earth Engine image with selected bands
        """
        collection = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt("CLOUD_COVER", cloud_threshold)))
        
        # Select bands and apply scaling factors
        def scale_landsat(image):
            optical_bands = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(0.0000275).add(-0.2)
            return image.addBands(optical_bands, None, True)
        
        # Cloud masking
        def mask_landsat_clouds(image):
            qa = image.select('QA_PIXEL')
            cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)  # Cloud
            shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)  # Cloud shadow
            mask = cloud_mask.And(shadow_mask)
            return image.updateMask(mask)
        
        # Process collection
        bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        band_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        
        composite = (collection
                    .map(scale_landsat)
                    .map(mask_landsat_clouds)
                    .median()
                    .select(bands, band_names)
                    .clip(geometry))
        
        print(f"Landsat 8 collection size: {collection.size().getInfo()}")
        return composite
    
    def download_multisensor_stack(self,
                                  geometry: ee.Geometry,
                                  start_date: str,
                                  end_date: str,
                                  include_s1: bool = True,
                                  include_s2: bool = True,
                                  include_l8: bool = False,
                                  scale: int = 10,
                                  sickle_compatible: bool = True,
                                  agricultural_focus: bool = True) -> ee.Image:
        """
        Download and stack multiple sensors with SICKLE-compatible configuration.
        
        Args:
            geometry: Earth Engine geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_s1: Include Sentinel-1 data
            include_s2: Include Sentinel-2 data
            include_l8: Include Landsat 8 data
            scale: Output pixel resolution
            sickle_compatible: Use SICKLE dataset specifications
            agricultural_focus: Apply agricultural-specific processing
            
        Returns:
            Stacked Earth Engine image
        """
        images = []
        
        if include_s2:
            s2 = self.download_sentinel2(geometry, start_date, end_date, 
                                       scale=scale, sickle_compatible=sickle_compatible)
            if agricultural_focus and sickle_compatible:
                # Add agricultural indices for SICKLE compatibility
                s2 = self._add_agricultural_indices(s2)
            images.append(s2)
        
        if include_s1:
            s1 = self.download_sentinel1(geometry, start_date, end_date, 
                                       scale=scale, sickle_compatible=sickle_compatible)
            images.append(s1)
        
        if include_l8:
            l8 = self.download_landsat8(geometry, start_date, end_date, scale=scale)
            # Resample to target resolution if needed
            if scale != 30:
                l8 = l8.resample('bilinear').reproject(crs='EPSG:4326', scale=scale)
            images.append(l8)
        
        # Stack all images
        if len(images) == 1:
            return images[0]
        else:
            stacked = images[0]
            for img in images[1:]:
                stacked = stacked.addBands(img)
            return stacked
    
    def _add_agricultural_indices(self, s2_image: ee.Image) -> ee.Image:
        """
        Add agricultural-specific indices to Sentinel-2 image for SICKLE compatibility.
        """
        # SICKLE-style agricultural indices
        red = s2_image.select('red')
        nir = s2_image.select('nir')
        green = s2_image.select('green')
        rededge1 = s2_image.select('rededge1')
        swir1 = s2_image.select('swir1')
        
        # Key agricultural indices used in SICKLE
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi')
        gndvi = nir.subtract(green).divide(nir.add(green)).rename('gndvi')
        ndre = nir.subtract(rededge1).divide(nir.add(rededge1)).rename('ndre')
        ndwi = green.subtract(swir1).divide(green.add(swir1)).rename('ndwi')
        
        return s2_image.addBands([ndvi, gndvi, ndre, ndwi])
    
    def export_to_drive(self,
                       image: ee.Image,
                       geometry: ee.Geometry,
                       description: str,
                       folder: str = "GEE_Exports",
                       scale: int = 10,
                       file_format: str = 'GeoTIFF') -> ee.batch.Task:
        """
        Export image to Google Drive.
        
        Args:
            image: Earth Engine image to export
            geometry: Region of interest
            description: Export description
            folder: Drive folder name
            scale: Export resolution in meters
            file_format: File format ('GeoTIFF' or 'NPY')
            
        Returns:
            Export task
        """
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            fileNamePrefix=description,
            scale=scale,
            region=geometry,
            fileFormat=file_format,
            maxPixels=1e13
        )
        
        task.start()
        print(f"Export task '{description}' started")
        return task
    
    def download_to_array(self,
                         image: ee.Image,
                         geometry: ee.Geometry,
                         scale: int = 10,
                         max_pixels: int = 1e8) -> np.ndarray:
        """
        Download image data as numpy array.
        
        Args:
            image: Earth Engine image
            geometry: Region of interest
            scale: Pixel resolution in meters
            max_pixels: Maximum number of pixels
            
        Returns:
            Numpy array of image data
        """
        try:
            # Get image data as array
            array = image.sampleRectangle(
                region=geometry,
                defaultValue=0,
                properties=[],
                maxPixels=max_pixels
            )
            
            # Convert to numpy array
            band_names = image.bandNames().getInfo()
            arrays = []
            
            for band in band_names:
                band_data = np.array(array.get(band).getInfo())
                arrays.append(band_data)
            
            # Stack bands
            stacked = np.stack(arrays, axis=-1)
            
            print(f"Downloaded array shape: {stacked.shape}")
            print(f"Band names: {band_names}")
            
            return stacked
            
        except Exception as e:
            print(f"Error downloading array: {e}")
            print("Try reducing the area or increasing scale parameter")
            raise
    
    def get_collection_info(self,
                           geometry: ee.Geometry,
                           start_date: str,
                           end_date: str) -> Dict:
        """
        Get information about available data in the time period.
        
        Returns:
            Dictionary with collection statistics
        """
        info = {}
        
        # Sentinel-2
        s2_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(geometry)
                        .filterDate(start_date, end_date))
        info['sentinel2_count'] = s2_collection.size().getInfo()
        
        # Sentinel-1
        s1_collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
                        .filterBounds(geometry)
                        .filterDate(start_date, end_date))
        info['sentinel1_count'] = s1_collection.size().getInfo()
        
        # Landsat 8
        l8_collection = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                        .filterBounds(geometry)
                        .filterDate(start_date, end_date))
        info['landsat8_count'] = l8_collection.size().getInfo()
        
        print(f"Available images in {start_date} to {end_date}:")
        print(f"  Sentinel-2: {info['sentinel2_count']}")
        print(f"  Sentinel-1: {info['sentinel1_count']}")
        print(f"  Landsat 8: {info['landsat8_count']}")
        
        return info