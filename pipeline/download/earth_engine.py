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
                          scale: int = 10) -> ee.Image:
        """
        Download Sentinel-2 surface reflectance data.
        
        Args:
            geometry: Earth Engine geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_threshold: Maximum cloud cover percentage
            scale: Pixel resolution in meters
            
        Returns:
            Earth Engine image with selected bands
        """
        collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)))
        
        # Select bands: Blue, Green, Red, NIR, SWIR1, SWIR2
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
                          scale: int = 10) -> ee.Image:
        """
        Download Sentinel-1 GRD data.
        
        Args:
            geometry: Earth Engine geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            orbit: Orbit direction ('ASCENDING', 'DESCENDING', 'BOTH')
            scale: Pixel resolution in meters
            
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
        
        # Apply speckle filtering
        def apply_speckle_filter(image):
            vv = image.select('VV').focal_median(50, 'circle', 'meters')
            vh = image.select('VH').focal_median(50, 'circle', 'meters')
            return image.addBands(vv.rename('VV_filtered')).addBands(vh.rename('VH_filtered'))
        
        # Get median composite
        composite = collection.map(apply_speckle_filter).median()
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
                                  scale: int = 10) -> ee.Image:
        """
        Download and stack multiple sensors into one image.
        
        Args:
            geometry: Earth Engine geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_s1: Include Sentinel-1 data
            include_s2: Include Sentinel-2 data
            include_l8: Include Landsat 8 data
            scale: Output pixel resolution
            
        Returns:
            Stacked Earth Engine image
        """
        images = []
        
        if include_s2:
            s2 = self.download_sentinel2(geometry, start_date, end_date, scale=scale)
            images.append(s2)
        
        if include_s1:
            s1 = self.download_sentinel1(geometry, start_date, end_date, scale=scale)
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