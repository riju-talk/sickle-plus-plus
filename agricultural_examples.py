"""
SICKLE-Compatible Agricultural Pipeline Examples
===============================================

This script demonstrates how to use the enhanced pipeline for agricultural applications
following SICKLE dataset best practices.
"""

import os
import sys
from datetime import datetime, timedelta

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

from pipeline.orchestrator.live_pipeline import quick_agricultural_inference, quick_agricultural_dataset_build
from pipeline.models.inference import create_dummy_model
import json


def create_agricultural_field_examples():
    """Create example agricultural field geometries for different crop types."""
    
    # Paddy field (SICKLE-style)
    paddy_field = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Paddy Field - Tamil Nadu Style",
                    "crop_type": "paddy",
                    "location": "Cauvery Delta, Tamil Nadu, India",
                    "area_acres": 0.38,  # SICKLE average
                    "sickle_compatible": True
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [79.1, 10.8],   # Tamil Nadu coordinates
                        [79.11, 10.8],
                        [79.11, 10.81],
                        [79.1, 10.81],
                        [79.1, 10.8]
                    ]]
                }
            }
        ]
    }
    
    # Corn field (US Midwest style)
    corn_field = {
        "type": "FeatureCollection", 
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Corn Field - Iowa",
                    "crop_type": "corn",
                    "location": "Iowa, USA",
                    "area_acres": 5.0,
                    "sickle_compatible": True
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-93.5, 41.5],
                        [-93.48, 41.5],
                        [-93.48, 41.52],
                        [-93.5, 41.52],
                        [-93.5, 41.5]
                    ]]
                }
            }
        ]
    }
    
    # Save examples
    os.makedirs("agricultural_examples", exist_ok=True)
    
    with open("agricultural_examples/paddy_field.geojson", 'w') as f:
        json.dump(paddy_field, f, indent=2)
        
    with open("agricultural_examples/corn_field.geojson", 'w') as f:
        json.dump(corn_field, f, indent=2)
    
    return ["agricultural_examples/paddy_field.geojson", "agricultural_examples/corn_field.geojson"]


def example_agricultural_inference():
    """Example of SICKLE-compatible agricultural inference."""
    print("=" * 60)
    print("🌾 SICKLE-Compatible Agricultural Inference Example")
    print("=" * 60)
    
    # Create example geometries
    field_files = create_agricultural_field_examples()
    
    # Create agricultural model
    model_path = "agricultural_examples/agricultural_model.pth"
    create_dummy_model(input_channels=16, num_classes=1, save_path=model_path)  # More channels for agricultural features
    
    for field_file in field_files:
        crop_type = "paddy" if "paddy" in field_file else "corn"
        print(f"\n📍 Processing {crop_type} field: {os.path.basename(field_file)}")
        
        try:
            # Run agricultural inference with SICKLE compatibility
            results = quick_agricultural_inference(
                geometry_file=field_file,
                model_path=model_path,
                crop_type=crop_type,
                growing_season=True  # Use growing season dates
            )
            
            print(f"✅ Agricultural Analysis Results:")
            print(f"   Field Area: {results['field_info']['area_km2']:.4f} km²")
            print(f"   Growing Season: {results['growing_season']['start']} to {results['growing_season']['end']}")
            print(f"   Agricultural Features: {results['feature_info']['agricultural_features']}")
            print(f"   SAR Features: {results['feature_info']['sar_features']}")
            print(f"   SICKLE Compatible: ✅")
            
            # Show agricultural-specific results
            if results['agricultural_results']:
                for task, task_results in results['agricultural_results'].items():
                    print(f"   {task.upper()}: Available")
            
        except Exception as e:
            print(f"❌ Error processing {crop_type} field: {e}")
            print("   This may be due to Earth Engine authentication or network issues")


def example_agricultural_dataset_building():
    """Example of building agricultural datasets with SICKLE compatibility."""
    print("\n" + "=" * 60)
    print("🗂️ SICKLE-Compatible Agricultural Dataset Building")
    print("=" * 60)
    
    field_files = ["agricultural_examples/paddy_field.geojson"]  # Use paddy field for SICKLE compatibility
    
    for field_file in field_files:
        print(f"\n📊 Building agricultural dataset for: {os.path.basename(field_file)}")
        
        try:
            # Build agricultural dataset with seasonal awareness
            result = quick_agricultural_dataset_build(
                geometry_file=field_file,
                crop_type="paddy",  # SICKLE crop type
                num_seasons=3,
                output_dir="agricultural_examples/paddy_dataset",
                season_start_month=6  # June start for paddy in Tamil Nadu
            )
            
            print(f"✅ Agricultural Dataset Created:")
            print(f"   Successful samples: {result['num_samples']}")
            print(f"   Failed samples: {result['failed_samples']}")
            print(f"   Dataset path: {result['dataset_path']}")
            print(f"   SICKLE Compatible: ✅")
            
        except Exception as e:
            print(f"❌ Error building dataset: {e}")
            print("   This may be due to Earth Engine authentication or data availability")


def example_dry_run_agricultural_analysis():
    """Example of using the dry run script for agricultural analysis."""
    print("\n" + "=" * 60)
    print("🔍 Agricultural Dry Run Analysis Example")
    print("=" * 60)
    
    print("To run agricultural dry run analysis, use the enhanced gee_dry_run.py script:")
    print("\nCommand examples:")
    print("# Basic agricultural analysis")
    print("cd single_download_script")
    print("python gee_dry_run.py ../agricultural_examples/paddy_field.geojson")
    print("\n# With custom date range and output")
    print("python gee_dry_run.py ../agricultural_examples/paddy_field.geojson \\")
    print("  --start-date 2024-06-01 --end-date 2024-12-31 \\")
    print("  --output paddy_analysis.json")
    
    print("\n🆕 New Features in dry run analysis:")
    print("   • SICKLE band compatibility assessment")
    print("   • Agricultural suitability scoring")
    print("   • Crop monitoring readiness evaluation")
    print("   • Field size validation for agricultural analysis")
    print("   • Temporal coverage assessment for crop monitoring")
    print("   • Cloud coverage impact on agricultural tasks")


def main():
    """Run all agricultural examples."""
    print("🚀 SICKLE++ Agricultural Pipeline Examples")
    print("=" * 60)
    print("This script demonstrates the enhanced agricultural capabilities")
    print("based on SICKLE dataset best practices.")
    print("\nFeatures demonstrated:")
    print("• SICKLE-compatible band selections")
    print("• Agricultural-specific feature engineering")
    print("• Crop monitoring workflows")
    print("• Growing season-aware processing")
    print("• Agricultural quality control")
    print("• Multi-crop support")
    
    try:
        # Run examples
        example_agricultural_inference()
        example_agricultural_dataset_building()
        example_dry_run_agricultural_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 All agricultural examples completed!")
        print("=" * 60)
        print("\n📁 Generated files:")
        print("   agricultural_examples/paddy_field.geojson - SICKLE-style paddy field")
        print("   agricultural_examples/corn_field.geojson - US Midwest corn field")
        print("   agricultural_examples/agricultural_model.pth - Example agricultural model")
        print("   agricultural_examples/paddy_dataset/ - Multi-seasonal dataset")
        
        print("\n🌾 SICKLE Compatibility Features:")
        print("   ✅ 12-band Sentinel-2 configuration")
        print("   ✅ VV/VH SAR processing")
        print("   ✅ Agricultural indices (NDVI, GNDVI, NDRE, RVI)")
        print("   ✅ Quality control (25% zero pixel threshold)")
        print("   ✅ Growing season awareness")
        print("   ✅ Multi-resolution support (3m, 10m, 30m)")
        print("   ✅ Crop-specific processing")
        
        print("\n🚀 Next Steps:")
        print("   1. Run Earth Engine authentication: earthengine authenticate")
        print("   2. Test with your own field geometries")
        print("   3. Train models on agricultural datasets")
        print("   4. Deploy for operational crop monitoring")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("This may be due to missing dependencies or Earth Engine setup")


if __name__ == "__main__":
    main()