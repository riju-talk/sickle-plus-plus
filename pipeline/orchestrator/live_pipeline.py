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
        """Load pipeline configuration with SICKLE-compatible defaults."""
        # SICKLE-compatible default configuration
        default_config = {
            'download': {
                'cloud_threshold': 20.0,
                'scale': 10,  # SICKLE default resolution
                'include_s1': True,
                'include_s2': True, 
                'include_l8': False,
                'sickle_compatible': True,
                'agricultural_focus': True
            },
            'features': {
                'include_vegetation': True,
                'include_sar': True,
                'include_texture': False,
                'include_spectral': False,
                'sickle_compatible': True,
                'crop_focus': True,
                'quality_control': True,
                'zero_threshold': 0.25,  # SICKLE quality control standard
                'normalization': 'minmax'
            },
            'inference': {
                'task_type': 'crop_monitoring',  # SICKLE tasks: crop_type, yield_prediction, phenology
                'confidence_threshold': 0.5,
                'agricultural_tasks': {
                    'crop_type_mapping': True,
                    'yield_prediction': True,
                    'sowing_date': False,
                    'transplanting_date': False,
                    'harvesting_date': False
                }
            },
            'validation': {
                'max_area_km2': 10000,
                'min_area_km2': 0.1,  # Minimum field size for analysis
            },
            'temporal': {
                'growing_season_aware': True,
                'season_start_month': 6,  # Crop calendar dependent
                'season_duration_months': 6,
                'temporal_compositing': 'median'
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
    
    def run_agricultural_inference(self,
                                  geometry_file: str,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  output_dir: Optional[str] = None,
                                  crop_type: str = 'general',
                                  growing_season: bool = True) -> Dict:
        """
        Run agricultural inference pipeline optimized for crop monitoring.
        
        Args:
            geometry_file: Path to KML or GeoJSON file
            start_date: Start date (YYYY-MM-DD), defaults to growing season start
            end_date: End date (YYYY-MM-DD), defaults to growing season end
            output_dir: Directory to save outputs
            crop_type: Type of crop for specialized processing
            growing_season: Use growing season dates
            
        Returns:
            Dictionary with agricultural inference results
        """
        self.logger.info("Starting agricultural inference pipeline...")
        
        # Set growing season dates if requested
        if growing_season and (start_date is None or end_date is None):
            start_date, end_date = self._get_growing_season_dates()
        elif end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')  # Crop season
        
        try:
            # Step 1: Load and validate geometry
            self.logger.info("Loading agricultural field geometry...")
            geometry, bbox, metadata = load_geometry(geometry_file)
            
            # Agricultural area validation
            if not validate_geometry_size(bbox, self.config['validation']['max_area_km2']):
                raise ValueError("Field area too large for processing")
            if metadata['area_km2'] < self.config['validation']['min_area_km2']:
                raise ValueError("Field area too small for reliable analysis")
            
            self.last_geometry = geometry
            
            # Step 2: Check agricultural data availability
            self.logger.info("Checking satellite data availability for crop monitoring...")
            data_info = self.downloader.get_collection_info(geometry, start_date, end_date)
            
            # Step 3: Download agricultural satellite data
            self.logger.info("Downloading multi-sensor agricultural data...")
            satellite_image = self.downloader.download_multisensor_stack(
                geometry=geometry,
                start_date=start_date,
                end_date=end_date,
                **self.config['download']
            )
            
            # Step 4: Agricultural feature engineering
            self.logger.info("Computing agricultural features and indices...") 
            feature_image = self.feature_engineer.create_agricultural_feature_stack(
                satellite_image,
                **self.config['features']
            )
            
            # Apply SICKLE-style quality control
            if self.config['features']['quality_control']:
                feature_image = self.feature_engineer.apply_sickle_quality_control(
                    feature_image, self.config['features']['zero_threshold']
                )
            
            # Step 5: Convert to numpy array
            self.logger.info("Preparing features for agricultural analysis...")
            features, feature_names = self.feature_engineer.to_numpy_features(
                feature_image, geometry, scale=self.config['download']['scale']
            )
            
            self.last_features = features
            
            # Step 6: Run agricultural inference
            agricultural_results = None
            if self.model_inference:
                self.logger.info("Running agricultural model inference...")
                agricultural_results = self._run_agricultural_tasks(features)
            
            # Step 7: Compile agricultural results
            results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'agricultural_monitoring',
                'crop_type': crop_type,
                'growing_season': {
                    'start': start_date, 
                    'end': end_date,
                    'season_aware': growing_season
                },
                'field_info': {
                    **metadata,
                    'suitable_for_agriculture': metadata['area_km2'] >= self.config['validation']['min_area_km2']
                },
                'data_availability': data_info,
                'feature_info': {
                    'shape': features.shape,
                    'feature_names': feature_names,
                    'agricultural_features': len([f for f in feature_names if any(x in f for x in ['ndvi', 'gndvi', 'evi', 'savi', 'rvi'])]),
                    'sar_features': len([f for f in feature_names if any(x in f for x in ['vv', 'vh', 'ratio'])]),
                    'quality_controlled': self.config['features']['quality_control']
                },
                'agricultural_results': agricultural_results,
                'config': self.config
            }
            
            # Step 8: Save agricultural outputs
            if output_dir:
                self._save_agricultural_results(results, features, output_dir)
            
            self.logger.info("Agricultural inference completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"Agricultural pipeline failed: {str(e)}")
            raise
    
    def _get_growing_season_dates(self) -> Tuple[str, str]:
        """Get growing season dates based on configuration."""
        current_date = datetime.now()
        
        # Calculate season start/end based on configuration
        start_month = self.config['temporal']['season_start_month']
        duration = self.config['temporal']['season_duration_months']
        
        # Determine current or previous season
        if current_date.month >= start_month:
            # Current year season
            season_start = datetime(current_date.year, start_month, 1)
        else:
            # Previous year season
            season_start = datetime(current_date.year - 1, start_month, 1)
        
        season_end = season_start + timedelta(days=duration * 30)  # Approximate months to days
        
        return season_start.strftime('%Y-%m-%d'), season_end.strftime('%Y-%m-%d')
    
    def _run_agricultural_tasks(self, features: np.ndarray) -> Dict:
        """Run agricultural-specific inference tasks."""
        task_config = self.config['inference']['agricultural_tasks']
        results = {}
        
        # Crop type mapping (SICKLE task)
        if task_config.get('crop_type_mapping', False):
            if self.config['inference']['task_type'] == 'crop_monitoring':
                crop_results = self.model_inference.predict_crop_classification(features)
                results['crop_type'] = crop_results
        
        # Yield prediction (SICKLE task)
        if task_config.get('yield_prediction', False):
            if hasattr(self.model_inference, 'predict_crop_yield'):
                yield_results = self.model_inference.predict_crop_yield(features)
                results['yield'] = yield_results
        
        # Phenology prediction (SICKLE tasks - dates)
        phenology_tasks = ['sowing_date', 'transplanting_date', 'harvesting_date']
        for task in phenology_tasks:
            if task_config.get(task, False):
                # This would require specialized models trained for phenology prediction
                self.logger.info(f"Phenology task {task} configured but requires specialized model")
                
        return results
    
    def _save_agricultural_results(self, results: Dict, features: np.ndarray, output_dir: str):
        """Save agricultural pipeline results with crop-specific organization."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save agricultural results JSON
        results_path = os.path.join(output_dir, 'agricultural_results.json')
        json_results = results.copy()
        if 'agricultural_results' in json_results and json_results['agricultural_results']:
            for task, task_results in json_results['agricultural_results'].items():
                for key, value in task_results.items():
                    if isinstance(value, np.ndarray):
                        json_results['agricultural_results'][task][key] = value.tolist()
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save agricultural features with SICKLE-compatible format
        features_path = os.path.join(output_dir, 'agricultural_features.npz')
        feature_metadata = {
            'features': features,
            'feature_names': results['feature_info']['feature_names'],
            'agricultural_indices': [f for f in results['feature_info']['feature_names'] 
                                   if any(x in f for x in ['ndvi', 'evi', 'savi', 'gndvi'])],
            'sar_indices': [f for f in results['feature_info']['feature_names']
                           if any(x in f for x in ['vv', 'vh', 'ratio', 'rvi'])],
            'crop_type': results['crop_type'],
            'sickle_compatible': True
        }
        np.savez_compressed(features_path, **feature_metadata)
        
        # Save agricultural inference outputs
        if results['agricultural_results']:
            inference_path = os.path.join(output_dir, 'agricultural_predictions.npz')
            inference_data = {}
            for task, task_results in results['agricultural_results'].items():
                for key, value in task_results.items():
                    if isinstance(value, np.ndarray):
                        inference_data[f"{task}_{key}"] = value
            if inference_data:
                np.savez_compressed(inference_path, **inference_data)
        
        self.logger.info(f"Agricultural results saved to {output_dir}")
    
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
def quick_agricultural_inference(geometry_file: str,
                               model_path: str,
                               crop_type: str = 'general',
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               growing_season: bool = True) -> Dict:
    """
    Quick agricultural inference for crop monitoring (SICKLE-compatible).
    
    Args:
        geometry_file: Path to KML or GeoJSON file
        model_path: Path to trained agricultural model
        crop_type: Type of crop ('paddy', 'corn', 'wheat', etc.)
        start_date: Start date (defaults to growing season)
        end_date: End date (defaults to growing season)
        growing_season: Use growing season dates
        
    Returns:
        Agricultural inference results
    """
    pipeline = LiveInferencePipeline(model_path=model_path)
    return pipeline.run_agricultural_inference(
        geometry_file=geometry_file,
        start_date=start_date,
        end_date=end_date,
        crop_type=crop_type,
        growing_season=growing_season
    )

def quick_inference(geometry_file: str,
                   model_path: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   agricultural: bool = True) -> Dict:
    """
    Quick inference with optional agricultural optimization.
    
    Args:
        geometry_file: Path to KML or GeoJSON file
        model_path: Path to trained model
        start_date: Start date (defaults based on agricultural setting)
        end_date: End date (defaults based on agricultural setting)  
        agricultural: Use agricultural-optimized pipeline
        
    Returns:
        Inference results
    """
    if agricultural:
        return quick_agricultural_inference(
            geometry_file, model_path, start_date=start_date, end_date=end_date
        )
    else:
        pipeline = LiveInferencePipeline(model_path=model_path)
        return pipeline.run_live_inference(
            geometry_file=geometry_file,
            start_date=start_date,
            end_date=end_date
        )


def quick_agricultural_dataset_build(geometry_file: str,
                                    crop_type: str,
                                    num_seasons: int = 3,
                                    output_dir: str = './agricultural_dataset',
                                    season_start_month: int = 6) -> Dict:
    """
    Quick agricultural dataset building for multiple growing seasons (SICKLE-compatible).
    
    Args:
        geometry_file: Path to KML or GeoJSON file
        crop_type: Type of crop for specialized processing
        num_seasons: Number of growing seasons to include
        output_dir: Output directory
        season_start_month: Month when growing season typically starts
        
    Returns:
        Agricultural dataset info
    """
    # Generate growing season date ranges
    from datetime import datetime, timedelta
    import calendar
    
    end_year = datetime.now().year
    date_ranges = []
    
    for i in range(num_seasons):
        year = end_year - i
        season_start = datetime(year, season_start_month, 1)
        
        # 6-month growing season (can be adjusted per crop)
        season_end = season_start + timedelta(days=180)
        
        date_ranges.append((
            season_start.strftime('%Y-%m-%d'),
            season_end.strftime('%Y-%m-%d')
        ))
    
    # Use agricultural pipeline configuration
    pipeline = LiveInferencePipeline()
    pipeline.config['temporal']['growing_season_aware'] = True
    pipeline.config['temporal']['season_start_month'] = season_start_month
    
    return pipeline.run_dataset_builder(
        geometry_file=geometry_file,
        date_ranges=date_ranges,
        output_dir=output_dir
    )

def quick_dataset_build(geometry_file: str,
                       date_ranges: List[Tuple[str, str]],
                       output_dir: str,
                       agricultural: bool = True) -> Dict:
    """
    Quick dataset building with optional agricultural optimization.
    
    Args:
        geometry_file: Path to KML or GeoJSON file
        date_ranges: List of (start_date, end_date) tuples
        output_dir: Output directory
        agricultural: Use agricultural-optimized pipeline
        
    Returns:
        Dataset info
    """
    if agricultural:
        # Use SICKLE-compatible configuration
        pipeline = LiveInferencePipeline()
        pipeline.config['download']['sickle_compatible'] = True
        pipeline.config['features']['sickle_compatible'] = True
    else:
        pipeline = LiveInferencePipeline()
    
    return pipeline.run_dataset_builder(
        geometry_file=geometry_file,
        date_ranges=date_ranges,
        output_dir=output_dir
    )