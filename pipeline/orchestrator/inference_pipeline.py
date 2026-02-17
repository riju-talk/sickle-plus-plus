import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import logging

# Import pipeline components
from ..geometry.loader import load_geometry, validate_geometry_size
from ..download.earth_engine import EarthEngineDownloader
from ..features.engineering import FeatureEngineer
from ..models.inference import ModelInference, SICKLEInference


class InferencePipeline:
    """
    Simplified inference-only pipeline for SICKLE++ agricultural analysis.
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_type: str = 'utae',
                 config_path: Optional[str] = None,
                 log_level: str = 'INFO'):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to trained model file
            model_type: Type of model ('utae', 'unet3d', 'pastis', etc.)
            config_path: Path to configuration file
            log_level: Logging level
        """
        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)
        
        # Initialize components for inference only
        self.feature_engineer = FeatureEngineer()
        self.model_inference = None
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.model_inference = ModelInference(model_path, model_type)
        
        # Load configuration with inference-focused defaults
        self.config = self._load_config(config_path)
        
        self.logger.info("🔮 Inference pipeline initialized")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load pipeline configuration optimized for inference."""
        # Default configuration for inference
        default_config = {
            'features': {
                'include_vegetation': True,
                'include_sar': True,
                'include_texture': False,
                'include_spectral': False,
                'crop_focus': True,
                'quality_control': True,
                'zero_threshold': 0.25,
                'normalization': 'minmax'
            },
            'inference': {
                'task_type': 'crop_classification',
                'confidence_threshold': 0.5,
                'batch_size': 8,
                'output_probabilities': True
            },
            'validation': {
                'max_area_km2': 10000,
                'min_area_km2': 0.1,
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def load_model(self, model_path: str, model_type: str = 'utae'):
        """Load inference model."""
        self.model_inference = ModelInference(model_path, model_type)
        self.logger.info(f"📦 Model loaded: {model_path}")
    
    def run_inference_from_data(self,
                               satellite_data: np.ndarray,
                               output_dir: Optional[str] = None) -> Dict:
        """
        Run inference on pre-loaded satellite data.
        
        Args:
            satellite_data: Satellite data array (T, C, H, W)
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with inference results
        """
        self.logger.info("🔮 Starting inference on satellite data...")
        
        try:
            # Step 1: Feature engineering
            self.logger.info("🛠️ Computing features...")
            features = self.feature_engineer.create_agricultural_feature_stack(
                satellite_data,
                **self.config['features']
            )
            
            # Step 2: Run inference
            if self.model_inference is None:
                raise ValueError("No model loaded. Call load_model() first.")
            
            self.logger.info("🤖 Running model inference...")
            results = self.model_inference.predict_crop_classification(features)
            
            # Step 3: Compile results
            inference_results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'agricultural_inference',
                'input_shape': satellite_data.shape,
                'feature_shape': features.shape,
                'predictions': results,
                'model_info': self.model_inference.get_model_info(),
                'config': self.config
            }
            
            # Step 4: Save if requested
            if output_dir:
                self._save_results(inference_results, output_dir)
            
            self.logger.info("✅ Inference completed successfully!")
            return inference_results
            
        except Exception as e:
            self.logger.error(f"❌ Inference failed: {str(e)}")
            raise
    
    def run_inference_from_downloaded_data(self, 
                                         downloads_dir: str,
                                         output_dir: Optional[str] = None) -> Dict:
        """
        Run inference on data from the download script.
        
        Args:
            downloads_dir: Directory containing downloaded satellite data
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with inference results
        """
        self.logger.info(f"📂 Loading data from: {downloads_dir}")
        
        # Load metadata
        metadata_path = os.path.join(downloads_dir, 'metadata', 'download_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            download_metadata = json.load(f)
        
        # TODO: Implement data loading from downloaded files
        # This would need to load the actual satellite data files from Google Drive
        # For now, return structure showing what would happen
        
        results = {
            'status': 'ready_for_inference',
            'downloads_dir': downloads_dir,
            'download_metadata': download_metadata,
            'message': 'Data loaded. Ready for inference when satellite files are available.',
            'next_steps': [
                '1. Download files from Google Drive folder "sickle_downloads"',
                '2. Place Sentinel-1 files in sentinel1/ directory', 
                '3. Place Sentinel-2 files in sentinel2/ directory',
                '4. Run inference with loaded data'
            ]
        }
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results_path = os.path.join(output_dir, 'inference_setup.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"💾 Setup info saved to: {results_path}")
        
        return results
    
    def run_field_analysis(self,
                          geojson_path: str,
                          satellite_data: np.ndarray,
                          output_dir: Optional[str] = None) -> Dict:
        """
        Run complete field analysis with geometry and predictions.
        
        Args:
            geojson_path: Path to field geometry file
            satellite_data: Preprocessed satellite data
            output_dir: Directory to save outputs
            
        Returns:
            Complete analysis results
        """
        self.logger.info("🌾 Starting field analysis...")
        
        # Load and validate geometry
        geometry, bbox, metadata = load_geometry(geojson_path)
        
        if not validate_geometry_size(bbox, self.config['validation']['max_area_km2']):
            raise ValueError("Field area too large for processing")
        if metadata['area_km2'] < self.config['validation']['min_area_km2']:
            raise ValueError("Field area too small for reliable analysis")
        
        # Run inference
        inference_results = self.run_inference_from_data(satellite_data, output_dir)
        
        # Create SICKLE-specific analysis
        sickle_inference = SICKLEInference(
            self.model_inference.model_path, 
            self.model_inference.model_type
        )
        
        agricultural_analysis = sickle_inference.predict_field(satellite_data)
        
        # Combine all results
        field_results = {
            'field_info': {
                **metadata,
                'geojson_path': geojson_path,
                'suitable_for_agriculture': metadata['area_km2'] >= self.config['validation']['min_area_km2']
            },
            'technical_inference': inference_results,
            'agricultural_analysis': agricultural_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        if output_dir:
            self._save_field_results(field_results, output_dir)
        
        self.logger.info("🎆 Field analysis completed!")
        return field_results
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save inference results."""
        os.makedirs(output_dir, exist_ok=True)
        
        results_path = os.path.join(output_dir, 'inference_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"💾 Results saved to: {results_path}")
    
    def _save_field_results(self, results: Dict, output_dir: str):
        """Save field analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save complete results
        results_path = os.path.join(output_dir, 'field_analysis_results.json')
        json_results = self._prepare_for_json(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save crop map if available
        if 'agricultural_analysis' in results and 'class_predictions' in results['agricultural_analysis']:
            crop_map = results['agricultural_analysis']['class_predictions']
            if isinstance(crop_map, np.ndarray):
                crop_map_path = os.path.join(output_dir, 'crop_map.npy')
                np.save(crop_map_path, crop_map)
                self.logger.info(f"🗺️ Crop map saved to: {crop_map_path}")
        
        self.logger.info(f"📊 Field results saved to: {results_path}")
    
    def _prepare_for_json(self, data):
        """Prepare data for JSON serialization by converting numpy arrays."""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        else:
            return data


def run_simple_inference(model_path: str, 
                        satellite_data: np.ndarray,
                        model_type: str = 'utae',
                        output_dir: Optional[str] = None) -> Dict:
    """
    Simple function to run inference without pipeline setup.
    
    Args:
        model_path: Path to trained model
        satellite_data: Satellite data array
        model_type: Model architecture type
        output_dir: Optional output directory
        
    Returns:
        Inference results
    """
    # Initialize pipeline
    pipeline = InferencePipeline(model_path, model_type)
    
    # Run inference
    results = pipeline.run_inference_from_data(satellite_data, output_dir)
    
    return results