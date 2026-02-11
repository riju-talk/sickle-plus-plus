"""
SICKLE++ Live Inference Pipeline
Complete pipeline for satellite-based crop monitoring and yield prediction.

Usage Examples:

1. Live Inference:
python download_script.py --mode live --geometry field1.kml --model model.pth

2. Dataset Building:
python download_script.py --mode dataset --geometry field1.kml --output ./dataset

3. Batch Processing:
python download_script.py --mode batch --geometries "*.kml" --model model.pth
"""

import sys
import os
import argparse
import glob
from pathlib import Path

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

from pipeline.orchestrator.live_pipeline import (LiveInferencePipeline, quick_inference, 
                                                  quick_dataset_build, quick_agricultural_inference, 
                                                  quick_agricultural_dataset_build)
from pipeline.models.inference import create_dummy_model

def main():
    parser = argparse.ArgumentParser(description='SICKLE++ Live Inference Pipeline')
    parser.add_argument('--mode', choices=['live', 'dataset', 'batch', 'demo', 'agricultural', 'crop-dataset'], 
                       required=True, help='Pipeline mode')
    parser.add_argument('--geometry', type=str, help='Path to KML/GeoJSON file')
    parser.add_argument('--geometries', type=str, help='Pattern for multiple geometry files')
    parser.add_argument('--model', type=str, help='Path to trained model file')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--crop-type', type=str, default='general', help='Crop type for agricultural analysis')
    parser.add_argument('--growing-season', action='store_true', help='Use growing season dates')
    parser.add_argument('--sickle-compatible', action='store_true', help='Use SICKLE-compatible processing')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_demo()
        return
    
    if args.mode == 'agricultural':
        if not args.geometry or not args.model:
            print("Error: --geometry and --model required for agricultural mode")
            return
        
        print(f"Running agricultural inference for {args.geometry} (crop: {args.crop_type})")
        results = quick_agricultural_inference(
            geometry_file=args.geometry,
            model_path=args.model,
            crop_type=args.crop_type,
            start_date=args.start_date,
            end_date=args.end_date,
            growing_season=args.growing_season
        )
        print("Agricultural Results:")
        if results['agricultural_results']:
            for task, task_results in results['agricultural_results'].items():
                print(f"  {task.upper()}:")
                for key, value in task_results.items():
                    if not isinstance(value, np.ndarray):
                        print(f"    {key}: {value}")
        print(f"  Field Area: {results['field_info']['area_km2']:.2f} km²")
        print(f"  Agricultural Features: {results['feature_info']['agricultural_features']}")
        print(f"  SAR Features: {results['feature_info']['sar_features']}")
        
    elif args.mode == 'live':
        if not args.geometry or not args.model:
            print("Error: --geometry and --model required for live mode")
            return
        
        print(f"Running live inference for {args.geometry}")
        results = quick_inference(
            geometry_file=args.geometry,
            model_path=args.model,
            start_date=args.start_date,
            end_date=args.end_date,
            agricultural=args.sickle_compatible
        )
        print("Inference Results:")
        if 'agricultural_results' in results:
            print(f"  Agricultural Analysis: ✅")
            print(f"  Crop Type: {results.get('crop_type', 'general')}")
        else:
            if results['inference_results']:
                print(f"  Mean Yield: {results['inference_results'].get('mean_yield', 'N/A')}")
                print(f"  Confidence: {results['inference_results'].get('confidence', 'N/A')}")
        print(f"  Feature Shape: {results['feature_info']['shape']}")
        print(f"  Features: {len(results['feature_info']['feature_names'])}")
        
    elif args.mode == 'crop-dataset':
        if not args.geometry:
            print("Error: --geometry required for crop dataset mode")
            return
            
        print(f"Building agricultural dataset for {args.geometry} (crop: {args.crop_type})")
        result = quick_agricultural_dataset_build(
            geometry_file=args.geometry,
            crop_type=args.crop_type,
            num_seasons=3,
            output_dir=args.output
        )
        print(f"Agricultural dataset created: {result['num_samples']} samples")
        print(f"Failed samples: {result['failed_samples']}")
        print(f"Saved to: {result['dataset_path']}")
        
    elif args.mode == 'dataset':
        if not args.geometry:
            print("Error: --geometry required for dataset mode")
            return
            
        # Create sample date ranges (seasonal for agricultural data)
        from datetime import datetime, timedelta
        end = datetime.now()
        date_ranges = []
        for i in range(4):  # 4 seasons
            start = end - timedelta(days=90)  # 3-month seasons
            date_ranges.append((start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
            end = start
        
        print(f"Building dataset for {args.geometry}")
        result = quick_dataset_build(
            geometry_file=args.geometry,
            date_ranges=date_ranges,
            output_dir=args.output,
            agricultural=args.sickle_compatible
        )
        print(f"Dataset created: {result['num_samples']} samples, {result['failed_samples']} failed")
        print(f"Saved to: {result['dataset_path']}")
        
    elif args.mode == 'batch':
        if not args.geometries or not args.model:
            print("Error: --geometries and --model required for batch mode")
            return
        
        geometry_files = glob.glob(args.geometries)
        print(f"Processing {len(geometry_files)} geometry files")
        
        for geom_file in geometry_files:
            try:
                print(f"\nProcessing {geom_file}...")
                results = quick_inference(
                    geometry_file=geom_file,
                    model_path=args.model,
                    start_date=args.start_date,
                    end_date=args.end_date
                )
                
                # Save results
                output_dir = os.path.join(args.output, Path(geom_file).stem)
                pipeline = LiveInferencePipeline()
                pipeline._save_results(results, results['feature_info'], output_dir)
                print(f"  Saved to {output_dir}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")


def run_demo():
    """Run a complete demo of the pipeline."""
    print("=== SICKLE++ Pipeline Demo ===\n")
    
    # Create demo geometry file
    demo_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-95.0, 40.0],
                        [-94.9, 40.0], 
                        [-94.9, 40.1],
                        [-95.0, 40.1],
                        [-95.0, 40.0]
                    ]]
                },
                "properties": {"name": "Demo Field"}
            }
        ]
    }
    
    demo_dir = "./demo_output"
    os.makedirs(demo_dir, exist_ok=True)
    
    geojson_path = os.path.join(demo_dir, "demo_field.geojson")
    with open(geojson_path, 'w') as f:
        import json
        json.dump(demo_geojson, f)
    
    print(f"Created demo geometry: {geojson_path}")
    
    # Create dummy model
    model_path = os.path.join(demo_dir, "demo_model.pth")
    dummy_model = create_dummy_model(input_channels=10, save_path=model_path)
    print(f"Created demo model: {model_path}")
    
    # Run pipeline with dummy data
    try:
        print("\nInitializing pipeline...")
        pipeline = LiveInferencePipeline(model_path=model_path)
        
        print("Pipeline status:")
        status = pipeline.get_pipeline_status()
        for key, value in status.items():
            if key != 'config':
                print(f"  {key}: {value}")
        
        print(f"\nDemo completed successfully!")
        print(f"To run real inference, provide actual geometry files and trained models.")
        print(f"Example: python download_script.py --mode live --geometry your_field.kml --model your_model.pth")
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        print("This may be due to Earth Engine authentication.")
        print("Run 'earthengine authenticate' to set up credentials.")


if __name__ == "__main__":
    main()
