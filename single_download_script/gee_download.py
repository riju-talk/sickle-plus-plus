#!/usr/bin/env python3
"""
Google Earth Engine Download Script
==================================

Downloads satellite data from Google Earth Engine for agricultural field analysis.
Downloads Sentinel-1, Sentinel-2 data and organizes it in a structured directory.

Usage:
    python gee_download.py path/to/field.geojson [--output-dir OUTPUT_DIR] [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]

Example:
    python gee_download.py test_field.geojson --output-dir ./downloads --start-date 2018-08-01 --end-date 2020-08-01
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


class GEEDataDownloader:
    """
    Google Earth Engine data downloader for satellite field data.
    """
    
    def __init__(self, output_dir: str = "downloads"):
        """Initialize the downloader with Earth Engine and output directory."""
        self.output_dir = output_dir
        self.create_directory_structure()
        
        try:
            ee.Authenticate()
            ee.Initialize(project='sickle-plus-plus')
            print("✅ Earth Engine initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Earth Engine: {e}")
            print("Please run: earthengine authenticate")
            raise
        
        self.downloader = EarthEngineDownloader(initialize_ee=False)
    
    def create_directory_structure(self):
        """Create organized directory structure for downloads."""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, 'sentinel1'),
            os.path.join(self.output_dir, 'sentinel2'),
            os.path.join(self.output_dir, 'metadata'),
            os.path.join(self.output_dir, 'geometry')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"📁 Created directory structure in: {self.output_dir}")
    
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
    
    def download_satellite_data(self, geometry: ee.Geometry, start_date: str, end_date: str) -> dict:
        """Download Sentinel-1 and Sentinel-2 data for the given geometry and date range."""
        print(f"\n🛰️ Downloading satellite data...")
        download_results = {}
        
        # Download Sentinel-2 data
        try:
            print("   📡 Downloading Sentinel-2...")
            s2_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                           .filterBounds(geometry)
                           .filterDate(start_date, end_date)
                           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                           .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']))
            
            s2_composite = s2_collection.median()
            
            # Download task
            s2_task = ee.batch.Export.image.toDrive(
                image=s2_composite,
                description=f'sentinel2_composite_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                folder='sickle_downloads',
                region=geometry,
                scale=10,
                maxPixels=1e9
            )
            s2_task.start()
            
            download_results['sentinel2'] = {
                'task_id': s2_task.id,
                'status': 'started',
                'bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
                'scale': 10
            }
            print(f"      ✅ Sentinel-2 task started: {s2_task.id}")
            
        except Exception as e:
            print(f"      ❌ Sentinel-2 download failed: {e}")
            download_results['sentinel2'] = {'error': str(e)}
        
        # Download Sentinel-1 data
        try:
            print("   📡 Downloading Sentinel-1...")
            s1_collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
                           .filterBounds(geometry)
                           .filterDate(start_date, end_date)
                           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                           .filter(ee.Filter.eq('instrumentMode', 'IW'))
                           .select(['VV', 'VH']))
            
            s1_composite = s1_collection.median()
            
            # Download task
            s1_task = ee.batch.Export.image.toDrive(
                image=s1_composite,
                description=f'sentinel1_composite_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                folder='sickle_downloads',
                region=geometry,
                scale=10,
                maxPixels=1e9
            )
            s1_task.start()
            
            download_results['sentinel1'] = {
                'task_id': s1_task.id,
                'status': 'started',
                'bands': ['VV', 'VH'],
                'scale': 10
            }
            print(f"      ✅ Sentinel-1 task started: {s1_task.id}")
            
        except Exception as e:
            print(f"      ❌ Sentinel-1 download failed: {e}")
            download_results['sentinel1'] = {'error': str(e)}
        
        return download_results
    
    def save_metadata(self, geometry_info: dict, download_info: dict, start_date: str, end_date: str):
        """Save metadata about the download to JSON file."""
        metadata = {
            'download_timestamp': datetime.now().isoformat(),
            'date_range': {
                'start': start_date,
                'end': end_date
            },
            'geometry_info': geometry_info,
            'download_info': download_info,
            'directory_structure': {
                'root': self.output_dir,
                'sentinel1': os.path.join(self.output_dir, 'sentinel1'),
                'sentinel2': os.path.join(self.output_dir, 'sentinel2'),
                'metadata': os.path.join(self.output_dir, 'metadata'),
                'geometry': os.path.join(self.output_dir, 'geometry')
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata', 'download_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"💾 Metadata saved to: {metadata_path}")
    
    def copy_geometry_file(self, geojson_path: str):
        """Copy the geometry file to the geometry directory."""
        import shutil
        geometry_dir = os.path.join(self.output_dir, 'geometry')
        filename = os.path.basename(geojson_path)
        dest_path = os.path.join(geometry_dir, filename)
        shutil.copy2(geojson_path, dest_path)
        print(f"📍 Geometry file copied to: {dest_path}")
    
    def run_download(self, geojson_path: str, start_date: str, end_date: str) -> dict:
        """Run complete download process."""
        print("🛰️ Starting satellite data download...")
        print("=" * 50)
        print(f"Date range: {start_date} to {end_date}")
        print(f"Field: {geojson_path}")
        print(f"Output directory: {self.output_dir}")
        
        try:
            # Analyze geometry
            geometry_info = self.analyze_geometry(geojson_path)
            
            # Copy geometry file to downloads
            self.copy_geometry_file(geojson_path)
            
            # Download satellite data 
            download_info = self.download_satellite_data(
                geometry_info['geometry'], start_date, end_date
            )
            
            # Save metadata
            self.save_metadata(geometry_info, download_info, start_date, end_date)
            
            # Print results
            print(f"\n✅ Download initiated successfully!")
            print(f"   📁 Files will be saved to: {self.output_dir}")
            print(f"   🗂️ Organization:")
            print(f"      ├── sentinel1/     (SAR data)")
            print(f"      ├── sentinel2/     (Multispectral data)")
            print(f"      ├── metadata/      (Download info)")
            print(f"      └── geometry/      (Field boundaries)")
            print(f"\n⏳ Google Earth Engine tasks are processing...")
            print(f"   Check Google Drive folder 'sickle_downloads' for completed files")
            
            # Print task IDs
            for sensor, info in download_info.items():
                if 'task_id' in info:
                    print(f"   {sensor.upper()} task ID: {info['task_id']}")
            
            return {
                'status': 'success',
                'output_directory': self.output_dir,
                'geometry_info': geometry_info,
                'download_info': download_info
            }
            
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return {'status': 'failed', 'error': str(e)}


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Download satellite data from Google Earth Engine')
    parser.add_argument('geojson_file', help='Path to GeoJSON file containing field geometry')
    parser.add_argument('--output-dir', default='downloads', help='Output directory for downloaded data')  
    parser.add_argument('--start-date', default='2018-08-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2020-08-01', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print("🛰️ Google Earth Engine Data Download")
    print("=" * 50)
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Field: {args.geojson_file}")
    print(f"Output: {args.output_dir}")
    
    try:
        # Initialize downloader
        downloader = GEEDataDownloader(args.output_dir)
        
        # Run download
        results = downloader.run_download(args.geojson_file, args.start_date, args.end_date)
        
        if results['status'] == 'success':
            print(f"\n🎉 Download process completed successfully!")
            return 0  
        else:
            print(f"\n❌ Download failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        return 1


if __name__ == "__main__":
    exit(main())