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
from typing import Dict, List, Tuple

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
            ee.Initialize()
            print("✅ Earth Engine initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Earth Engine: {e}")
            print("Please run: earthengine authenticate")
            raise
        
        self.downloader = EarthEngineDownloader(initialize_ee=False)
    
    def analyze_geometry(self, geojson_path: str) -> Dict:
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
    
    def analyze_sentinel2_availability(self, geometry: ee.Geometry, start_date: str, end_date: str) -> Dict:
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
            
            # Standard S2 bands with descriptions
            s2_bands = {
                'B1': 'Coastal aerosol (443nm)',
                'B2': 'Blue (490nm)',
                'B3': 'Green (560nm)',
                'B4': 'Red (665nm)',
                'B5': 'Vegetation red edge (705nm)',
                'B6': 'Vegetation red edge (740nm)',
                'B7': 'Vegetation red edge (783nm)',
                'B8': 'NIR (842nm)',
                'B8A': 'Vegetation red edge (865nm)',
                'B9': 'Water vapour (945nm)',
                'B11': 'SWIR 1 (1610nm)',
                'B12': 'SWIR 2 (2190nm)',
                'QA60': 'Cloud mask'
            }
            
            print("   📊 Sentinel-2 Bands:")
            for band, description in s2_bands.items():
                if band in band_names:
                    print(f"      ✅ {band}: {description}")
                else:
                    print(f"      ❌ {band}: {description} (not available)")
            
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
    
    def analyze_sentinel1_availability(self, geometry: ee.Geometry, start_date: str, end_date: str) -> Dict:
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
            
            # Standard S1 bands with descriptions
            s1_bands = {
                'VV': 'Vertical transmit, vertical receive',
                'VH': 'Vertical transmit, horizontal receive',
                'HH': 'Horizontal transmit, horizontal receive',
                'HV': 'Horizontal transmit, vertical receive',
                'angle': 'Incidence angle'
            }
            
            print("   📊 Sentinel-1 Bands:")
            for band, description in s1_bands.items():
                if band in band_names:
                    print(f"      ✅ {band}: {description}")
                else:
                    print(f"      ❌ {band}: {description} (not available)")
            
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
    
    def analyze_landsat8_availability(self, geometry: ee.Geometry, start_date: str, end_date: str) -> Dict:
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
    
    def estimate_download_size(self, geometry_info: Dict, s2_info: Dict, s1_info: Dict, l8_info: Dict) -> Dict:
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
    
    def generate_download_summary(self, geometry_info: Dict, s2_info: Dict, s1_info: Dict, 
                                l8_info: Dict, size_estimates: Dict, start_date: str, end_date: str) -> Dict:
        """Generate a complete summary of the dry run analysis."""
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
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
            'recommendations': []
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
    parser = argparse.ArgumentParser(
        description='Google Earth Engine dry run analysis for satellite data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('geojson', type=str, help='Path to GeoJSON file')
    parser.add_argument('--start-date', type=str, 
                       help='Start date (YYYY-MM-DD), default: 30 days ago')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--output', type=str,
                       help='Save analysis results to JSON file')
    
    args = parser.parse_args()
    
    # Set default dates
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    if args.start_date is None:
        args.start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print("🛰️  Google Earth Engine Dry Run Analysis")
    print("=" * 50)
    print(f"Date range: {args.start_date} to {args.end_date}")
    
    try:
        # Initialize analyzer
        analyzer = GEEDryRunAnalyzer()
        
        # Analyze geometry
        geometry_info = analyzer.analyze_geometry(args.geojson)
        
        # Analyze each satellite dataset
        s2_info = analyzer.analyze_sentinel2_availability(
            geometry_info['geometry'], args.start_date, args.end_date
        )
        
        s1_info = analyzer.analyze_sentinel1_availability(
            geometry_info['geometry'], args.start_date, args.end_date
        )
        
        l8_info = analyzer.analyze_landsat8_availability(
            geometry_info['geometry'], args.start_date, args.end_date
        )
        
        # Estimate download size
        size_estimates = analyzer.estimate_download_size(
            geometry_info, s2_info, s1_info, l8_info
        )
        
        # Generate summary
        summary = analyzer.generate_download_summary(
            geometry_info, s2_info, s1_info, l8_info, size_estimates,
            args.start_date, args.end_date
        )
        
        # Print recommendations
        print(f"\n🎯 Recommendations:")
        for rec in summary['recommendations']:
            print(f"   {rec}")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n💾 Analysis results saved to: {args.output}")
        
        print(f"\n✅ Dry run analysis complete!")
        print(f"   Run with --output analysis.json to save detailed results")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())