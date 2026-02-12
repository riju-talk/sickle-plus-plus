#!/usr/bin/env python3
"""
Google Earth Engine Dry Run Script
==================================

Performs a dry run analysis for satellite data download from Google Earth Engine.
Shows available data and band information for Landsat 8, Sentinel-1, and Sentinel-2
without actually downloading any data.

Usage:
    python gee_dry_run.py path/to/field.geojson [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]

Example:
    python gee_dry_run.py ../examples/iowa_field.geojson --start-date 2024-01-01 --end-date 2024-12-31
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ee
from pipeline.geometry.loader import load_geometry, validate_geometry_size
from pipeline.download.earth_engine import EarthEngineDownloader


class GEEDryRunAnalyzer:
    """
    Google Earth Engine dry run analyzer for satellite data availability.
    """
    
    def __init__(self):
        """Initialize the analyzer with Earth Engine."""
        try:
            ee.Authenticate()
            ee.Initialize(project='sickle-plus-plus')
            print("✅ Earth Engine initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Earth Engine: {e}")
            print("Please run: earthengine authenticate")
            raise
        
        self.downloader = EarthEngineDownloader(initialize_ee=False)
    
    def analyze_geometry(self, geojson_path: str) -> dict:
        """
        Analyze the geometry file and return information.
        
        Args:
            geojson_path: Path to GeoJSON file
            
        Returns:
            Dictionary with geometry information
        """
        print(f"\n📍 Analyzing geometry: {geojson_path}")
        
        # Load geometry
        geometry, bbox, metadata = load_geometry(geojson_path)
        
        # Validate size
        is_valid = validate_geometry_size(bbox, max_area_km2=50000)
        
        print(f"   Area: {metadata['area_km2']:.2f} km²")
        print(f"   Bounding box: {bbox}")
        print(f"   Centroid: {metadata['centroid']}")
        print(f"   Size valid: {'✅' if is_valid else '❌'}")
        
        if not is_valid:
            print("   ⚠️  Warning: Large geometry may cause slow processing")
        
        return {
            'geometry': geometry,
            'bbox': bbox,
            'metadata': metadata,
            'is_valid': is_valid
        }
    
    def analyze_sentinel2_availability(self, geometry: ee.Geometry, start_date: str, end_date: str) -> dict:
        """Analyze Sentinel-2 data availability."""
        print(f"\n🛰️  Analyzing Sentinel-2 data...")
        
        # Get collection
        collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date))
        
        # Get collection with cloud filtering
        cloud_filtered = collection.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        
        # Get sample image for band info
        sample_image = collection.first()
        
        try:
            total_images = collection.size().getInfo()
            cloud_filtered_count = cloud_filtered.size().getInfo()
            
            if sample_image:
                band_names = sample_image.bandNames().getInfo()
            else:
                band_names = []
            
            print(f"   Total images available: {total_images}")
            print(f"   Images with <20% clouds: {cloud_filtered_count}")
            print(f"   Available bands: {len(band_names)}")
            
            # SICKLE-compatible S2 bands (agricultural focus)
            s2_bands = {
                'B1': 'Coastal aerosol (443nm) - Atmospheric correction',
                'B2': 'Blue (490nm) - Water feature detection',
                'B3': 'Green (560nm) - Vegetation health',
                'B4': 'Red (665nm) - Chlorophyll absorption', 
                'B5': 'Red edge 1 (705nm) - Leaf area index',
                'B6': 'Red edge 2 (740nm) - Chlorophyll content',
                'B7': 'Red edge 3 (783nm) - Vegetation stress',
                'B8': 'NIR (842nm) - Vegetation biomass',
                'B8A': 'Narrow NIR (865nm) - Precise vegetation analysis',
                'B9': 'Water vapour (945nm) - Atmospheric correction',
                'B11': 'SWIR 1 (1610nm) - Crop moisture',
                'B12': 'SWIR 2 (2190nm) - Crop senescence',
                'QA60': 'Cloud mask - Quality control'
            }
            
            # SICKLE dataset uses these 12 specific bands
            sickle_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
            
            print("   📊 Sentinel-2 Bands (SICKLE Agricultural Focus):")
            for band, description in s2_bands.items():
                if band in band_names:
                    is_sickle = '🌾' if band in sickle_bands else '  '
                    print(f"      ✅ {is_sickle} {band}: {description}")
                else:
                    print(f"      ❌    {band}: {description} (not available)")
            
            # Agricultural suitability check
            sickle_available = sum(1 for band in sickle_bands if band in band_names)
            print(f"   🌾 SICKLE Compatibility: {sickle_available}/12 required bands available")
            if sickle_available >= 10:
                print(f"   ✅ Excellent for agricultural analysis")
            elif sickle_available >= 8:
                print(f"   ⚠️  Good for agricultural analysis (some bands missing)")
            else:
                print(f"   ❌ Limited agricultural analysis capability")
            
            return {
                'total_images': total_images,
                'cloud_filtered_count': cloud_filtered_count,
                'available_bands': band_names,
                'band_descriptions': s2_bands,
                'scale': 10  # meters
            }
        except Exception as e:
            print(f"   ❌ Error analyzing Sentinel-2: {e}")
            return {'error': str(e)}
    
    def analyze_sentinel1_availability(self, geometry: ee.Geometry, start_date: str, end_date: str) -> dict:
        """Analyze Sentinel-1 data availability."""
        print(f"\n📡 Analyzing Sentinel-1 SAR data...")
        
        # Get collection
        collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                     .filter(ee.Filter.eq('instrumentMode', 'IW')))
        
        # Get ascending and descending orbits
        ascending = collection.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        descending = collection.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        
        # Get sample image for band info
        sample_image = collection.first()
        
        try:
            total_images = collection.size().getInfo()
            ascending_count = ascending.size().getInfo()
            descending_count = descending.size().getInfo()
            
            if sample_image:
                band_names = sample_image.bandNames().getInfo()
            else:
                band_names = []
            
            print(f"   Total SAR images: {total_images}")
            print(f"   Ascending orbit: {ascending_count}")
            print(f"   Descending orbit: {descending_count}")
            print(f"   Available bands: {len(band_names)}")
            
            # SICKLE-compatible S1 bands (agricultural SAR analysis)
            s1_bands = {
                'VV': 'Vertical-Vertical - Crop structure/biomass',
                'VH': 'Vertical-Horizontal - Crop volume/roughness', 
                'HH': 'Horizontal-Horizontal - Surface conditions',
                'HV': 'Horizontal-Vertical - Volume scattering',
                'angle': 'Incidence angle - Geometric correction'
            }
            
            # SICKLE uses VV and VH for crop monitoring
            sickle_sar_bands = ['VV', 'VH']
            
            print("   📊 Sentinel-1 Bands (Agricultural SAR Analysis):")
            for band, description in s1_bands.items():
                if band in band_names:
                    is_sickle = '🌾' if band in sickle_sar_bands else '  '
                    print(f"      ✅ {is_sickle} {band}: {description}")
                else:
                    print(f"      ❌    {band}: {description} (not available)")
                    
            # Agricultural SAR suitability
            sar_available = sum(1 for band in sickle_sar_bands if band in band_names)
            print(f"   🌾 SICKLE SAR Compatibility: {sar_available}/2 required bands available")
            if sar_available == 2:
                print(f"   ✅ Optimal for crop monitoring (VV+VH dual-pol)")
            elif sar_available == 1:
                print(f"   ⚠️  Limited crop analysis (single polarization)")
            else:
                print(f"   ❌ No suitable SAR data for crop monitoring")
            
            return {
                'total_images': total_images,
                'ascending_count': ascending_count,
                'descending_count': descending_count,
                'available_bands': band_names,
                'band_descriptions': s1_bands,
                'scale': 10  # meters
            }
        except Exception as e:
            print(f"   ❌ Error analyzing Sentinel-1: {e}")
            return {'error': str(e)}
    
    def analyze_landsat8_availability(self, geometry: ee.Geometry, start_date: str, end_date: str) -> dict:
        """Analyze Landsat 8 data availability."""
        print(f"\n🌍 Analyzing Landsat 8 data...")
        
        # Get collection
        collection = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date))
        
        # Get collection with cloud filtering
        cloud_filtered = collection.filter(ee.Filter.lt("CLOUD_COVER", 20))
        
        # Get sample image for band info
        sample_image = collection.first()
        
        try:
            total_images = collection.size().getInfo()
            cloud_filtered_count = cloud_filtered.size().getInfo()
            
            if sample_image:
                band_names = sample_image.bandNames().getInfo()
            else:
                band_names = []
            
            print(f"   Total images available: {total_images}")
            print(f"   Images with <20% clouds: {cloud_filtered_count}")
            print(f"   Available bands: {len(band_names)}")
            
            # Standard L8 bands with descriptions
            l8_bands = {
                'SR_B1': 'Coastal aerosol (443nm)',
                'SR_B2': 'Blue (482nm)',
                'SR_B3': 'Green (562nm)',
                'SR_B4': 'Red (655nm)', 
                'SR_B5': 'NIR (865nm)',
                'SR_B6': 'SWIR 1 (1610nm)',
                'SR_B7': 'SWIR 2 (2200nm)',
                'ST_B10': 'Thermal infrared (10.9μm)',
                'QA_PIXEL': 'Pixel quality',
                'QA_RADSAT': 'Radiometric saturation',
                'ST_QA': 'Surface temperature quality',
                'ST_TRAD': 'Thermal radiance',
                'ST_URAD': 'Upwelled radiance',
                'ST_DRAD': 'Downwelled radiance',
                'ST_ATRAN': 'Atmospheric transmission',
                'ST_EMIS': 'Emissivity',
                'ST_EMSD': 'Emissivity standard deviation',
                'ST_CDIST': 'Distance to cloud'
            }
            
            print("   📊 Landsat 8 Bands:")
            for band, description in l8_bands.items():
                if band in band_names:
                    print(f"      ✅ {band}: {description}")
                else:
                    print(f"      ❌ {band}: {description} (not available)")
            
            return {
                'total_images': total_images,
                'cloud_filtered_count': cloud_filtered_count,
                'available_bands': band_names,
                'band_descriptions': l8_bands,
                'scale': 30  # meters
            }
        except Exception as e:
            print(f"   ❌ Error analyzing Landsat 8: {e}")
            return {'error': str(e)}
    
    def estimate_download_size(self, geometry_info: dict, s2_info: dict, s1_info: dict, l8_info: dict) -> dict:
        """Estimate download size for the data."""
        print(f"\n💾 Estimating download size...")
        
        # Get area in square meters
        area_km2 = geometry_info['metadata']['area_km2']
        area_m2 = area_km2 * 1e6
        
        estimates = {}
        total_size_mb = 0
        
        # Sentinel-2 estimation (10m resolution, ~13 bands, 2 bytes per pixel)
        if 'total_images' in s2_info and s2_info['total_images'] > 0:
            s2_pixels = area_m2 / (10 * 10)  # 10m pixels
            s2_bands = len(s2_info.get('available_bands', []))
            s2_size_mb = (s2_pixels * s2_bands * 2) / 1e6  # 2 bytes per pixel
            estimates['sentinel2'] = {
                'images': s2_info['cloud_filtered_count'],
                'size_mb_per_image': s2_size_mb,
                'total_size_mb': s2_size_mb * s2_info['cloud_filtered_count']
            }
            total_size_mb += estimates['sentinel2']['total_size_mb']
        
        # Sentinel-1 estimation (10m resolution, ~2-3 bands, 4 bytes per pixel)
        if 'total_images' in s1_info and s1_info['total_images'] > 0:
            s1_pixels = area_m2 / (10 * 10)  # 10m pixels
            s1_bands = len(s1_info.get('available_bands', []))
            s1_size_mb = (s1_pixels * s1_bands * 4) / 1e6  # 4 bytes per pixel (float32)
            estimates['sentinel1'] = {
                'images': s1_info['total_images'],
                'size_mb_per_image': s1_size_mb,
                'total_size_mb': s1_size_mb * s1_info['total_images']
            }
            total_size_mb += estimates['sentinel1']['total_size_mb']
        
        # Landsat 8 estimation (30m resolution, ~18 bands, 2 bytes per pixel)
        if 'total_images' in l8_info and l8_info['total_images'] > 0:
            l8_pixels = area_m2 / (30 * 30)  # 30m pixels
            l8_bands = len(l8_info.get('available_bands', []))
            l8_size_mb = (l8_pixels * l8_bands * 2) / 1e6  # 2 bytes per pixel
            estimates['landsat8'] = {
                'images': l8_info['cloud_filtered_count'],
                'size_mb_per_image': l8_size_mb,
                'total_size_mb': l8_size_mb * l8_info['cloud_filtered_count']
            }
            total_size_mb += estimates['landsat8']['total_size_mb']
        
        estimates['total_size_mb'] = total_size_mb
        estimates['total_size_gb'] = total_size_mb / 1000
        
        for sensor, info in estimates.items():
            if sensor != 'total_size_mb' and sensor != 'total_size_gb':
                print(f"   {sensor.upper()}:")
                print(f"      Images: {info['images']}")
                print(f"      Size per image: {info['size_mb_per_image']:.1f} MB")
                print(f"      Total size: {info['total_size_mb']:.1f} MB")
        
        print(f"   📊 TOTAL ESTIMATED SIZE: {total_size_mb:.1f} MB ({estimates['total_size_gb']:.2f} GB)")
        
        return estimates
    
    def analyze_agricultural_suitability(self, geometry_info: Dict, s2_info: Dict, 
                                        s1_info: Dict, l8_info: Dict) -> Dict:
        """Analyze suitability for agricultural applications based on SICKLE requirements."""
        print(f"\n🌾 Analyzing agricultural suitability...")
        
        suitability = {
            'overall_score': 0,
            'recommendations': [],
            'sickle_compatibility': {},
            'agricultural_readiness': {}
        }
        
        # Area suitability (SICKLE works with 0.38 acre average plots = ~0.0015 km²)
        area_km2 = geometry_info['metadata']['area_km2']
        if area_km2 >= 0.001:  # Minimum field size
            area_score = min(10, area_km2 * 5)  # Scale to 10
            area_suitable = True
            print(f"   ✅ Field size suitable: {area_km2:.3f} km²")
        else:
            area_score = 0
            area_suitable = False
            print(f"   ❌ Field too small: {area_km2:.3f} km² (min: 0.001 km²)")
        
        # SICKLE S2 band compatibility
        sickle_s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        s2_bands_available = s2_info.get('available_bands', [])
        s2_compatibility = sum(1 for band in sickle_s2_bands if band in s2_bands_available)
        s2_score = (s2_compatibility / 12) * 30  # 30 points max
        
        print(f"   🔍 Sentinel-2 SICKLE compatibility: {s2_compatibility}/12 bands ({s2_score:.1f}/30 pts)")
        
        # SICKLE S1 SAR compatibility
        sickle_s1_bands = ['VV', 'VH']
        s1_bands_available = s1_info.get('available_bands', [])
        s1_compatibility = sum(1 for band in sickle_s1_bands if band in s1_bands_available) 
        s1_score = (s1_compatibility / 2) * 20  # 20 points max
        
        print(f"   📡 Sentinel-1 SICKLE compatibility: {s1_compatibility}/2 bands ({s1_score:.1f}/20 pts)")
        
        # Data availability for time series analysis
        total_images = (
            s2_info.get('cloud_filtered_count', 0) +
            s1_info.get('total_images', 0) +
            l8_info.get('cloud_filtered_count', 0)
        )
        
        if total_images >= 12:  # Monthly time series
            temporal_score = 20
            print(f"   📈 Excellent temporal coverage: {total_images} images (20/20 pts)")
        elif total_images >= 6:
            temporal_score = 15
            print(f"   📈 Good temporal coverage: {total_images} images (15/20 pts)")
        elif total_images >= 3:
            temporal_score = 10
            print(f"   📈 Limited temporal coverage: {total_images} images (10/20 pts)")
        else:
            temporal_score = 0
            print(f"   📈 Insufficient temporal coverage: {total_images} images (0/20 pts)")
        
        # Cloud coverage impact on agricultural analysis
        s2_cloud_ratio = (
            s2_info.get('cloud_filtered_count', 0) / 
            max(s2_info.get('total_images', 1), 1)
        )
        
        if s2_cloud_ratio > 0.7:
            cloud_score = 10
            print(f"   ☀️ Excellent cloud-free data: {s2_cloud_ratio:.1%} clear (10/10 pts)")
        elif s2_cloud_ratio > 0.4:
            cloud_score = 7
            print(f"   ⛅ Good cloud-free data: {s2_cloud_ratio:.1%} clear (7/10 pts)")
        elif s2_cloud_ratio > 0.2:
            cloud_score = 4
            print(f"   ☁️ Limited cloud-free data: {s2_cloud_ratio:.1%} clear (4/10 pts)")
        else:
            cloud_score = 0
            print(f"   ⛈️ Insufficient cloud-free data: {s2_cloud_ratio:.1%} clear (0/10 pts)")
        
        # Calculate overall agricultural suitability score
        suitability['overall_score'] = s2_score + s1_score + temporal_score + cloud_score
        max_score = 80  # 30+20+20+10
        score_percentage = (suitability['overall_score'] / max_score) * 100
        
        # Generate recommendations
        recommendations = []
        
        if score_percentage >= 80:
            recommendations.append("✅ Excellent for SICKLE-style agricultural analysis")
        elif score_percentage >= 60:
            recommendations.append("✅ Good for agricultural analysis with minor limitations")
        elif score_percentage >= 40:
            recommendations.append("⚠️ Suitable for basic agricultural analysis")
        else:
            recommendations.append("❌ Limited suitability for agricultural analysis")
        
        if not area_suitable:
            recommendations.append("📍 Consider larger field area for better results")
        
        if s2_compatibility < 10:
            recommendations.append("🌈 Missing key Sentinel-2 bands for vegetation analysis")
            
        if s1_compatibility < 2:
            recommendations.append("📡 Missing SAR polarizations for crop structure analysis")
            
        if total_images < 6:
            recommendations.append("📅 Extend date range for better temporal analysis")
            
        if s2_cloud_ratio < 0.3:
            recommendations.append("☁️ Consider different season for less cloud cover")
        
        # SICKLE compatibility assessment
        suitability['sickle_compatibility'] = {
            's2_bands': f"{s2_compatibility}/12",
            's1_bands': f"{s1_compatibility}/2",
            'ready_for_sickle': s2_compatibility >= 10 and s1_compatibility == 2
        }
        
        # Agricultural readiness
        suitability['agricultural_readiness'] = {
            'field_size_ok': area_suitable,
            'multispectral_ready': s2_compatibility >= 8,
            'sar_ready': s1_compatibility >= 1,
            'temporal_ready': total_images >= 6,
            'cloud_acceptable': s2_cloud_ratio >= 0.3
        }
        
        suitability['recommendations'] = recommendations
        
        print(f"\n🎆 Agricultural Suitability Score: {suitability['overall_score']:.1f}/{max_score} ({score_percentage:.1f}%)")
        for rec in recommendations:
            print(f"   {rec}")
        
        return suitability
    
    def analyze_agricultural_suitability(self, geometry_info: Dict, s2_info: Dict, 
                                        s1_info: Dict, l8_info: Dict) -> Dict:
        """Analyze suitability for agricultural applications based on SICKLE requirements."""
        print(f"\n🌾 Analyzing agricultural suitability...")
        
        suitability = {
            'overall_score': 0,
            'recommendations': [],
            'sickle_compatibility': {},
            'agricultural_readiness': {}
        }
        
        # Area suitability (SICKLE works with 0.38 acre average plots = ~0.0015 km²)
        area_km2 = geometry_info['metadata']['area_km2']
        if area_km2 >= 0.001:  # Minimum field size
            area_score = min(10, area_km2 * 5)  # Scale to 10
            area_suitable = True
            print(f"   ✅ Field size suitable: {area_km2:.3f} km²")
        else:
            area_score = 0
            area_suitable = False
            print(f"   ❌ Field too small: {area_km2:.3f} km² (min: 0.001 km²)")
        
        # SICKLE S2 band compatibility
        sickle_s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        s2_bands_available = s2_info.get('available_bands', [])
        s2_compatibility = sum(1 for band in sickle_s2_bands if band in s2_bands_available)
        s2_score = (s2_compatibility / 12) * 30  # 30 points max
        
        print(f"   🔍 Sentinel-2 SICKLE compatibility: {s2_compatibility}/12 bands ({s2_score:.1f}/30 pts)")
        
        # SICKLE S1 SAR compatibility
        sickle_s1_bands = ['VV', 'VH']
        s1_bands_available = s1_info.get('available_bands', [])
        s1_compatibility = sum(1 for band in sickle_s1_bands if band in s1_bands_available) 
        s1_score = (s1_compatibility / 2) * 20  # 20 points max
        
        print(f"   📡 Sentinel-1 SICKLE compatibility: {s1_compatibility}/2 bands ({s1_score:.1f}/20 pts)")
        
        # Data availability for time series analysis
        total_images = (
            s2_info.get('cloud_filtered_count', 0) +
            s1_info.get('total_images', 0) +
            l8_info.get('cloud_filtered_count', 0)
        )
        
        if total_images >= 12:  # Monthly time series
            temporal_score = 20
            print(f"   📈 Excellent temporal coverage: {total_images} images (20/20 pts)")
        elif total_images >= 6:
            temporal_score = 15
            print(f"   📈 Good temporal coverage: {total_images} images (15/20 pts)")
        elif total_images >= 3:
            temporal_score = 10
            print(f"   📈 Limited temporal coverage: {total_images} images (10/20 pts)")
        else:
            temporal_score = 0
            print(f"   📈 Insufficient temporal coverage: {total_images} images (0/20 pts)")
        
        # Cloud coverage impact on agricultural analysis
        s2_cloud_ratio = (
            s2_info.get('cloud_filtered_count', 0) / 
            max(s2_info.get('total_images', 1), 1)
        )
        
        if s2_cloud_ratio > 0.7:
            cloud_score = 10
            print(f"   ☀️ Excellent cloud-free data: {s2_cloud_ratio:.1%} clear (10/10 pts)")
        elif s2_cloud_ratio > 0.4:
            cloud_score = 7
            print(f"   ⛅ Good cloud-free data: {s2_cloud_ratio:.1%} clear (7/10 pts)")
        elif s2_cloud_ratio > 0.2:
            cloud_score = 4
            print(f"   ☁️ Limited cloud-free data: {s2_cloud_ratio:.1%} clear (4/10 pts)")
        else:
            cloud_score = 0
            print(f"   ⛈️ Insufficient cloud-free data: {s2_cloud_ratio:.1%} clear (0/10 pts)")
        
        # Calculate overall agricultural suitability score
        suitability['overall_score'] = s2_score + s1_score + temporal_score + cloud_score
        max_score = 80  # 30+20+20+10
        score_percentage = (suitability['overall_score'] / max_score) * 100
        
        # Generate recommendations
        recommendations = []
        
        if score_percentage >= 80:
            recommendations.append("✅ Excellent for SICKLE-style agricultural analysis")
        elif score_percentage >= 60:
            recommendations.append("✅ Good for agricultural analysis with minor limitations")
        elif score_percentage >= 40:
            recommendations.append("⚠️ Suitable for basic agricultural analysis")
        else:
            recommendations.append("❌ Limited suitability for agricultural analysis")
        
        if not area_suitable:
            recommendations.append("📌 Consider larger field area for better results")
        
        if s2_compatibility < 10:
            recommendations.append("🌈 Missing key Sentinel-2 bands for vegetation analysis")
            
        if s1_compatibility < 2:
            recommendations.append("📡 Missing SAR polarizations for crop structure analysis")
            
        if total_images < 6:
            recommendations.append("📅 Extend date range for better temporal analysis")
            
        if s2_cloud_ratio < 0.3:
            recommendations.append("☁️ Consider different season for less cloud cover")
        
        # SICKLE compatibility assessment
        suitability['sickle_compatibility'] = {
            's2_bands': f"{s2_compatibility}/12",
            's1_bands': f"{s1_compatibility}/2",
            'ready_for_sickle': s2_compatibility >= 10 and s1_compatibility == 2
        }
        
        # Agricultural readiness
        suitability['agricultural_readiness'] = {
            'field_size_ok': area_suitable,
            'multispectral_ready': s2_compatibility >= 8,
            'sar_ready': s1_compatibility >= 1,
            'temporal_ready': total_images >= 6,
            'cloud_acceptable': s2_cloud_ratio >= 0.3
        }
        
        suitability['recommendations'] = recommendations
        
        print(f"\n🎆 Agricultural Suitability Score: {suitability['overall_score']:.1f}/{max_score} ({score_percentage:.1f}%)")
        for rec in recommendations:
            print(f"   {rec}")
        
        return suitability
    
    def generate_download_summary(self, geometry_info: Dict, s2_info: Dict, s1_info: Dict, 
                                l8_info: Dict, size_estimates: Dict, start_date: str, end_date: str,
                                agricultural_suit: Dict) -> Dict:
        """Generate a complete summary of the dry run analysis including agricultural assessment."""
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'analysis_type': 'agricultural_satellite_data',
            'geometry': {
                'area_km2': geometry_info['metadata']['area_km2'],
                'centroid': geometry_info['metadata']['centroid'],
                'bbox': geometry_info['bbox']
            },
            'date_range': {
                'start': start_date,
                'end': end_date
            },
            'sentinel2': s2_info,
            'sentinel1': s1_info,
            'landsat8': l8_info,
            'size_estimates': size_estimates,
            'agricultural_suitability': agricultural_suit,
            'recommendations': agricultural_suit['recommendations']
        }
        
        # Generate recommendations
        recommendations = []
        
        # Check data availability
        total_images = (s2_info.get('total_images', 0) + 
                       s1_info.get('total_images', 0) + 
                       l8_info.get('total_images', 0))
        
        if total_images == 0:
            recommendations.append("❌ No data available for this region/time period")
        elif total_images < 10:
            recommendations.append("⚠️  Limited data available - consider extending date range")
        else:
            recommendations.append("✅ Good data availability")
        
        # Check cloud coverage
        s2_cloud_ratio = (s2_info.get('cloud_filtered_count', 0) / 
                         max(s2_info.get('total_images', 1), 1))
        l8_cloud_ratio = (l8_info.get('cloud_filtered_count', 0) / 
                         max(l8_info.get('total_images', 1), 1))
        
        if s2_cloud_ratio < 0.3 or l8_cloud_ratio < 0.3:
            recommendations.append("☁️  High cloud coverage - consider different season")
        
        # Check size
        if size_estimates['total_size_gb'] > 10:
            recommendations.append("💾 Large download size - consider reducing area or date range")
        elif size_estimates['total_size_gb'] > 1:
            recommendations.append("💾 Moderate download size - should be manageable")
        else:
            recommendations.append("💾 Small download size - quick to process")
        
        summary['recommendations'] = recommendations
        
        return summary


def main():
    # Hardcoded values
    geojson_path = "test_field.geojson"
    start_date = "2018-08-01"
    end_date = "2020-08-01"
    output_file = None  # Can be set to save results
    
    print("🛰️  Google Earth Engine Dry Run Analysis")
    print("=" * 50)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Field: {geojson_path}")
    
    try:
        # Initialize analyzer
        analyzer = GEEDryRunAnalyzer()
        
        # Analyze geometry
        geometry_info = analyzer.analyze_geometry(geojson_path)
        
        # Analyze each satellite dataset
        s2_info = analyzer.analyze_sentinel2_availability(
            geometry_info['geometry'], start_date, end_date
        )
        
        s1_info = analyzer.analyze_sentinel1_availability(
            geometry_info['geometry'], start_date, end_date
        )
        
        l8_info = analyzer.analyze_landsat8_availability(
            geometry_info['geometry'], start_date, end_date
        )
        
        # Estimate download size
        size_estimates = analyzer.estimate_download_size(
            geometry_info, s2_info, s1_info, l8_info
        )
        
        # Analyze agricultural suitability
        agricultural_suitability = analyzer.analyze_agricultural_suitability(
            geometry_info, s2_info, s1_info, l8_info
        ) 
         
        # Generate summary
        summary = analyzer.generate_download_summary(
            geometry_info, s2_info, s1_info, l8_info, size_estimates,
            start_date, end_date, agricultural_suitability
        )
        
        # Print final recommendations
        print(f"\n🎆 Final Assessment:")
        for rec in summary['agricultural_suitability']['recommendations']:
            print(f"   {rec}")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n💾 Analysis results saved to: {output_file}")
        
        print(f"\n✅ Dry run analysis complete!")
        print(f"   Set output_file variable to save detailed results")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())