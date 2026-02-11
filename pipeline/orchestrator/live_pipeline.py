import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging

# Import pipeline components
from ..geometry.loader import load_geometry, validate_geometry_size
from ..download.earth_engine import EarthEngineDownloader
from ..features.engineering import FeatureEngineer
from ..models.inference import ModelInference


class LiveInferencePipeline:
    """
    Complete pipeline for live satellite-based crop inference.
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 log_level: str = 'INFO'):
        """
        Initialize the live inference pipeline.
        
        Args:
            model_path: Path to trained model file
            config_path: Path to configuration file
            log_level: Logging level
        """
        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.downloader = EarthEngineDownloader()
        self.feature_engineer = FeatureEngineer()
        self.model_inference = None
        
        # Load model if provided
        if model_path:
            self.model_inference = ModelInference(model_path)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Pipeline state
        self.last_geometry = None
        self.last_features = None
        self.cache = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load pipeline configuration."""
        default_config = {
            'download': {
                'cloud_threshold': 20.0,
                'scale': 10,
                'include_s1': True,
                'include_s2': True,
                'include_l8': False
            },
            'features': {
                'include_vegetation': True,
                'include_sar': True,
                'include_texture': False,
                'include_spectral': False,
                'normalization': 'minmax'
            },
            'inference': {
                'task_type': 'yield_prediction',  # or 'classification'
                'confidence_threshold': 0.5
            },
            'validation': {
                'max_area_km2': 10000
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
    
    def load_model(self, model_path: str):
        """Load inference model."""
        self.model_inference = ModelInference(model_path)
        self.logger.info(f"Model loaded: {model_path}")
    
    def run_live_inference(self,
                          geometry_file: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          output_dir: Optional[str] = None) -> Dict:
        """
        Run complete live inference pipeline.
        
        Args:
            geometry_file: Path to KML or GeoJSON file
            start_date: Start date (YYYY-MM-DD), defaults to 30 days ago
            end_date: End date (YYYY-MM-DD), defaults to today
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with inference results
        """
        self.logger.info("Starting live inference pipeline...")
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        try:
            # Step 1: Load and validate geometry
            self.logger.info("Loading geometry...")
            geometry, bbox, metadata = load_geometry(geometry_file)
            
            if not validate_geometry_size(bbox, self.config['validation']['max_area_km2']):
                raise ValueError("Geometry too large for processing")
            
            self.last_geometry = geometry
            
            # Step 2: Check data availability
            self.logger.info("Checking data availability...")
            data_info = self.downloader.get_collection_info(geometry, start_date, end_date)
            
            # Step 3: Download satellite data
            self.logger.info("Downloading satellite data...")
            satellite_image = self.downloader.download_multisensor_stack(
                geometry=geometry,
                start_date=start_date,
                end_date=end_date,
                **self.config['download']
            )
            
            # Step 4: Feature engineering
            self.logger.info("Computing features...")
            feature_image = self.feature_engineer.create_feature_stack(
                satellite_image,
                **self.config['features']
            )
            
            # Step 5: Convert to numpy array
            self.logger.info("Preparing features for inference...")
            features, feature_names = self.feature_engineer.to_numpy_features(
                feature_image, geometry, scale=self.config['download']['scale']
            )
            
            self.last_features = features
            
            # Step 6: Run inference if model is loaded
            inference_results = None
            if self.model_inference:
                self.logger.info("Running model inference...")
                
                if self.config['inference']['task_type'] == 'yield_prediction':
                    inference_results = self.model_inference.predict_crop_yield(features)
                else:
                    inference_results = self.model_inference.predict_crop_classification(features)
            
            # Step 7: Compile results
            results = {
                'timestamp': datetime.now().isoformat(),
                'geometry_info': metadata,
                'date_range': {'start': start_date, 'end': end_date},
                'data_availability': data_info,
                'feature_info': {
                    'shape': features.shape,
                    'feature_names': feature_names,
                    'num_features': len(feature_names)
                },
                'inference_results': inference_results,
                'config': self.config
            }
            
            # Step 8: Save outputs if requested
            if output_dir:
                self._save_results(results, features, output_dir)
            
            self.logger.info("Live inference completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def run_dataset_builder(self,
                           geometry_file: str,
                           date_ranges: List[Tuple[str, str]],
                           output_dir: str) -> Dict:
        """
        Build dataset from multiple time periods.
        
        Args:
            geometry_file: Path to KML or GeoJSON file
            date_ranges: List of (start_date, end_date) tuples
            output_dir: Directory to save dataset
            
        Returns:
            Dictionary with dataset info
        """
        self.logger.info("Starting dataset builder...")
        
        # Load geometry
        geometry, bbox, metadata = load_geometry(geometry_file)
        
        dataset = []
        dataset_info = []
        
        for i, (start_date, end_date) in enumerate(date_ranges):
            self.logger.info(f"Processing period {i+1}/{len(date_ranges)}: {start_date} to {end_date}")
            
            try:
                # Download data for this period
                satellite_image = self.downloader.download_multisensor_stack(
                    geometry=geometry,
                    start_date=start_date,
                    end_date=end_date,
                    **self.config['download']
                )
                
                # Feature engineering
                feature_image = self.feature_engineer.create_feature_stack(
                    satellite_image,
                    **self.config['features']
                )
                
                # Convert to numpy
                features, feature_names = self.feature_engineer.to_numpy_features(
                    feature_image, geometry, scale=self.config['download']['scale']
                )
                
                dataset.append(features)
                dataset_info.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'shape': features.shape,
                    'success': True
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process period {start_date}-{end_date}: {str(e)}")
                dataset_info.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'shape': None,
                    'success': False,
                    'error': str(e)
                })
        
        # Save dataset
        os.makedirs(output_dir, exist_ok=True)
        
        dataset_path = os.path.join(output_dir, 'dataset.npz')
        np.savez_compressed(dataset_path, 
                           data=np.array(dataset),
                           feature_names=feature_names)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'dataset_info.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'geometry_info': metadata,
                'dataset_info': dataset_info,
                'feature_names': feature_names,
                'config': self.config
            }, f, indent=2)
        
        self.logger.info(f"Dataset saved to {output_dir}")
        
        return {
            'dataset_path': dataset_path,
            'metadata_path': metadata_path,
            'num_samples': len([info for info in dataset_info if info['success']]),
            'failed_samples': len([info for info in dataset_info if not info['success']]),
            'dataset_info': dataset_info
        }
    
    def _save_results(self, results: Dict, features: np.ndarray, output_dir: str):
        """Save pipeline results to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results JSON
        results_path = os.path.join(output_dir, 'results.json')
        # Create a copy without numpy arrays for JSON serialization
        json_results = results.copy()
        if 'inference_results' in json_results and json_results['inference_results']:
            for key, value in json_results['inference_results'].items():
                if isinstance(value, np.ndarray):
                    json_results['inference_results'][key] = value.tolist()
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save features
        features_path = os.path.join(output_dir, 'features.npz')
        np.savez_compressed(features_path, 
                           features=features,
                           feature_names=results['feature_info']['feature_names'])
        
        # Save inference outputs if available
        if results['inference_results']:
            inference_path = os.path.join(output_dir, 'inference_outputs.npz')
            np.savez_compressed(inference_path, **results['inference_results'])
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            'earth_engine_initialized': hasattr(self.downloader, 'initialized'),
            'model_loaded': self.model_inference is not None,
            'last_geometry_loaded': self.last_geometry is not None,
            'last_features_shape': self.last_features.shape if self.last_features is not None else None,
            'config': self.config
        }


# Convenience functions for quick usage
def quick_inference(geometry_file: str,
                   model_path: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> Dict:
    """
    Quick inference for a single geometry file.
    
    Args:
        geometry_file: Path to KML or GeoJSON file
        model_path: Path to trained model
        start_date: Start date (defaults to 30 days ago)
        end_date: End date (defaults to today)
        
    Returns:
        Inference results
    """
    pipeline = LiveInferencePipeline(model_path=model_path)
    return pipeline.run_live_inference(
        geometry_file=geometry_file,
        start_date=start_date,
        end_date=end_date
    )


def quick_dataset_build(geometry_file: str,
                       date_ranges: List[Tuple[str, str]],
                       output_dir: str) -> Dict:
    """
    Quick dataset building for a geometry file.
    
    Args:
        geometry_file: Path to KML or GeoJSON file
        date_ranges: List of (start_date, end_date) tuples
        output_dir: Output directory
        
    Returns:
        Dataset info
    """
    pipeline = LiveInferencePipeline()
    return pipeline.run_dataset_builder(
        geometry_file=geometry_file,
        date_ranges=date_ranges,
        output_dir=output_dir
    )