#!/usr/bin/env python3
"""Download multi-sensor satellite data via Google Earth Engine.

Simple wrapper around `pipeline.download.earth_engine.EarthEngineDownloader`.
Starts export tasks to Google Drive (user must run `earthengine authenticate`).
"""
import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.geometry.loader import load_geometry, validate_geometry_size
from pipeline.download.earth_engine import EarthEngineDownloader
import ee


def main():
    p = argparse.ArgumentParser(description='Download satellite data via Earth Engine')
    p.add_argument('--geometry', required=True, help='Path to GeoJSON or KML geometry')
    p.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    p.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    p.add_argument('--out_folder', default='sickle_downloads', help='Drive folder name for exports')
    p.add_argument('--include_s1', action='store_true', help='Include Sentinel-1')
    p.add_argument('--include_s2', action='store_true', help='Include Sentinel-2')
    p.add_argument('--include_l8', action='store_true', help='Include Landsat-8')
    p.add_argument('--scale', type=int, default=10, help='Export scale in meters')
    args = p.parse_args()

    # Load geometry
    ee_geometry, bbox, meta = load_geometry(args.geometry)
    if not validate_geometry_size(bbox):
        print('Geometry too large — adjust or split into smaller tiles')
        return

    # Initialize downloader
    downloader = EarthEngineDownloader(initialize_ee=True)

    # Use downloader to create composites
    print('\nStarting multisensor composite creation...')
    composite = downloader.download_multisensor_stack(
        geometry=ee_geometry,
        start_date=args.start,
        end_date=args.end,
        include_s1=args.include_s1,
        include_s2=args.include_s2,
        include_l8=args.include_l8,
        scale=args.scale,
        sickle_compatible=True
    )

    # Start export to Drive
    description = f'sickle_multisensor_{args.start}_{args.end}'
    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=description,
        folder=args.out_folder,
        region=bbox,
        scale=args.scale,
        maxPixels=1e10
    )
    task.start()
    print(f'Export task started: {task.id} (folder: {args.out_folder})')


if __name__ == '__main__':
    main()
