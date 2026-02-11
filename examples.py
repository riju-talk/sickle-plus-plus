"""
Example usage script for SICKLE++ Live Inference Pipeline
"""

import os
from datetime import datetime, timedelta
from pipeline.orchestrator.live_pipeline import LiveInferencePipeline, quick_inference
from pipeline.models.inference import create_dummy_model
import json

def create_example_geometry():
    """Create an example GeoJSON file for testing."""
    # Example field in Iowa, USA (corn belt)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Iowa Corn Field",
                    "crop_type": "corn",
                    "year": 2024
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-93.5, 41.5],
                        [-93.4, 41.5],
                        [-93.4, 41.6],
                        [-93.5, 41.6],
                        [-93.5, 41.5]
                    ]]
                }
            }
        ]
    }
    
    os.makedirs("examples", exist_ok=True)
    geojson_path = "examples/iowa_field.geojson"
    
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    return geojson_path

def example_live_inference():
    """Example of running live inference."""
    print("=== Live Inference Example ===")
    
    # Create example geometry
    geometry_file = create_example_geometry()
    print(f"Created example geometry: {geometry_file}")
    
    # Create dummy model
    model_path = "examples/example_model.pth"
    create_dummy_model(input_channels=15, save_path=model_path)
    print(f"Created example model: {model_path}")
    
    # Date range (last 30 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        # Run inference
        results = quick_inference(
            geometry_file=geometry_file,
            model_path=model_path,
            start_date=start_date,
            end_date=end_date
        )
        
        print("\nResults:")
        print(f"  Feature shape: {results['feature_info']['shape']}")
        print(f"  Feature count: {results['feature_info']['num_features']}")
        print(f"  Available S2 images: {results['data_availability']['sentinel2_count']}")
        print(f"  Available S1 images: {results['data_availability']['sentinel1_count']}")
        
        if results['inference_results']:
            inf_results = results['inference_results']
            print(f"  Predicted yield: {inf_results['mean_yield']:.2f}")
            print(f"  Confidence: {inf_results['confidence']:.2f}")
            print(f"  Yield range: {inf_results['min_yield']:.2f} - {inf_results['max_yield']:.2f}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("This may be due to:")
        print("  1. Earth Engine authentication (run 'earthengine authenticate')")
        print("  2. Network connectivity")
        print("  3. Data availability for the specified region/dates")

def example_dataset_building():
    """Example of building a dataset."""
    print("\n=== Dataset Building Example ===")
    
    geometry_file = create_example_geometry()
    
    # Create date ranges (quarterly for 2023)
    date_ranges = [
        ("2023-01-01", "2023-03-31"),
        ("2023-04-01", "2023-06-30"),
        ("2023-07-01", "2023-09-30"),
        ("2023-10-01", "2023-12-31")
    ]
    
    try:
        from pipeline.orchestrator.live_pipeline import quick_dataset_build
        
        result = quick_dataset_build(
            geometry_file=geometry_file,
            date_ranges=date_ranges,
            output_dir="examples/dataset"
        )
        
        print(f"Dataset created:")
        print(f"  Successful samples: {result['num_samples']}")
        print(f"  Failed samples: {result['failed_samples']}")
        print(f"  Dataset path: {result['dataset_path']}")
        
    except Exception as e:
        print(f"Error during dataset building: {e}")

def example_pipeline_status():
    """Example of checking pipeline status."""
    print("\n=== Pipeline Status Example ===")
    
    pipeline = LiveInferencePipeline()
    status = pipeline.get_pipeline_status()
    
    print("Pipeline Status:")
    for key, value in status.items():
        if key != 'config':
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # Run examples
    example_live_inference()
    example_dataset_building()
    example_pipeline_status()
    
    print("\n=== Pipeline Complete ===")
    print("Check the 'examples' directory for outputs.")
    print("\nTo run with real data:")
    print("1. Authenticate with Earth Engine: earthengine authenticate")
    print("2. Provide your own KML/GeoJSON geometries")
    print("3. Train and provide a real model")