#!/usr/bin/env python3
"""
SICKLE++ Baseline Inference Script

Run inference on satellite data for:
- Crop classification
- Phenology prediction
- Yield prediction
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from pipeline.models.inference import ModelInference, SICKLEInference
from pipeline.evaluation import EvaluationPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(
        description='Run SICKLE++ baseline inference'
    )
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='utae',
                       choices=['utae', 'unet3d', 'unet3d_multitask', 'pastis'],
                       help='Model architecture')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to input satellite data (NPY file)')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='Path to ground truth labels (optional, for evaluation)')
    parser.add_argument('--task', type=str, default='crop_classification',
                       choices=['crop_classification', 'phenology', 'yield'],
                       help='Task type')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device')
    parser.add_argument('--return_probabilities', action='store_true',
                       help='Return class probabilities')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Output directory')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"🎯 Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        logger.info(f"📦 Loading model from {args.model_path}")
        
        inference = ModelInference(
            model_path=args.model_path,
            model_type=args.model_type,
            device=device
        )
        
        # Load satellite data
        logger.info(f"📂 Loading satellite data from {args.data_path}")
        
        if args.data_path.endswith('.npy'):
            satellite_data = np.load(args.data_path)
        elif args.data_path.endswith('.npz'):
            with np.load(args.data_path) as data:
                # Assume 'satellite_data' key
                satellite_data = data['satellite_data'] if 'satellite_data' in data else data[list(data.keys())[0]]
        else:
            raise ValueError(f"Unsupported file format: {args.data_path}")
        
        logger.info(f"   Data shape: {satellite_data.shape}")
        
        # Run inference
        logger.info(f"🔮 Running inference...")
        
        results = inference.predict(
            satellite_data,
            return_probabilities=args.return_probabilities
        )
        
        # Additional agricultural analysis if crop classification
        if args.task == 'crop_classification':
            logger.info(f"🌾 Computing agricultural analysis...")
            
            sickle_inference = SICKLEInference(
                model_path=args.model_path,
                model_type=args.model_type
            )
            
            agricultural_results = sickle_inference.predict_field(satellite_data)
            results['agricultural_analysis'] = agricultural_results['agricultural_analysis']
        
        # Evaluate if labels provided
        if args.labels_path:
            logger.info(f"📊 Evaluating predictions...")
            
            # Load labels
            if args.labels_path.endswith('.npy'):
                labels = np.load(args.labels_path)
            elif args.labels_path.endswith('.json'):
                with open(args.labels_path, 'r') as f:
                    labels = json.load(f)
            else:
                labels = np.loadtxt(args.labels_path)
            
            # Create evaluator
            evaluator = EvaluationPipeline(
                output_dir=output_dir,
                verbose=True
            )
            
            # Evaluate
            if args.task == 'crop_classification':
                metrics = evaluator.evaluate_crop_classification(
                    labels,
                    results['class_predictions'],
                    save_results=True
                )
            elif args.task == 'yield':
                metrics = evaluator.evaluate_yield_prediction(
                    labels,
                    results['raw_predictions'],
                    save_results=True
                )
            else:
                metrics = evaluator.evaluate_phenology_prediction(
                    labels,
                    results['raw_predictions'],
                    save_results=True
                )
            
            results['evaluation'] = metrics
        
        # Save predictions
        if args.save_predictions:
            logger.info(f"💾 Saving predictions...")
            
            # Prepare for JSON serialization
            def prepare_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: prepare_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                else:
                    return obj
            
            serializable_results = prepare_for_json(results)
            
            # Save to JSON
            results_path = output_dir / 'inference_results.json'
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"   Results: {results_path}")
            
            # Save predictions as NPY
            predictions_path = output_dir / 'predictions.npy'
            np.save(predictions_path, results['class_predictions'])
            logger.info(f"   Predictions: {predictions_path}")
            
            # Save probabilities if available
            if 'probabilities' in results:
                probs_path = output_dir / 'probabilities.npy'
                np.save(probs_path, results['probabilities'])
                logger.info(f"   Probabilities: {probs_path}")
        
        # Print summary
        logger.info(f"✅ Inference complete!")
        logger.info(f"\nResults Summary:")
        logger.info(f"  Input shape: {results['input_shape']}")
        logger.info(f"  Predictions shape: {results['output_shape']}")
        logger.info(f"  Mean confidence: {results['mean_confidence']:.3f}")
        
        if 'agricultural_analysis' in results:
            agr = results['agricultural_analysis']
            logger.info(f"\nAgricultural Analysis:")
            logger.info(f"  Dominant crop: {agr['dominant_crop']}")
            logger.info(f"  Dominance: {agr['dominant_crop_percentage']:.1f}%")
            logger.info(f"  Field diversity: {agr['field_diversity']} crop types")
        
        if 'evaluation' in results:
            logger.info(f"\nEvaluation Results:")
            for key, value in results['evaluation'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
        
        logger.info(f"\nOutput directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
