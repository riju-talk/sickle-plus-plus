"""
Evaluation metrics and utilities for SICKLE++ baseline models.

Implements evaluation for:
- Crop classification (accuracy, F1, confusion matrix)
- Phenology prediction (MAE in days)
- Yield prediction (RMSE, R² score)
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, mean_squared_error, mean_absolute_error, r2_score
)
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SICKLEMetrics:
    """
    SICKLE baseline evaluation metrics.
    Includes classification, regression, and temporal prediction metrics.
    """
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None) -> Dict:
        """
        Compute classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: Optional class names for reporting
            
        Returns:
            Dict with accuracy, F1 score, and per-class metrics
        """
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Macro and weighted F1
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Get classification report
        if class_names:
            report = classification_report(
                y_true, y_pred, 
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
        else:
            report = classification_report(
                y_true, y_pred,
                output_dict=True,
                zero_division=0
            )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'n_samples': len(y_true),
            'n_unique_classes': len(np.unique(y_true))
        }
    
    @staticmethod
    def phenology_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict:
        """
        Compute phenology prediction metrics (predicting dates).
        
        Args:
            y_true: True dates (as day-of-year or unix timestamp)
            y_pred: Predicted dates
            
        Returns:
            Dict with MAE and RMSE
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Also compute percentage error
        pct_error = np.abs(y_true - y_pred) / np.abs(y_true + 1e-8) * 100
        mean_pct_error = np.mean(pct_error)
        
        return {
            'mae_days': float(mae),
            'rmse_days': float(rmse),
            'mean_percentage_error': float(mean_pct_error),
            'n_samples': len(y_true)
        }
    
    @staticmethod
    def yield_prediction_metrics(y_true: np.ndarray,
                                y_pred: np.ndarray) -> Dict:
        """
        Compute yield prediction metrics.
        
        Args:
            y_true: Ground truth yield values
            y_pred: Predicted yield values
            
        Returns:
            Dict with RMSE, MAE, R², and MAPE
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE - Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Scale-aware metrics
        yield_range = np.max(y_true) - np.min(y_true)
        nrmse = rmse / yield_range if yield_range > 0 else rmse
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape),
            'nrmse': float(nrmse),
            'yield_range': float(yield_range),
            'n_samples': len(y_true),
            'min_yield': float(np.min(y_true)),
            'max_yield': float(np.max(y_true))
        }
    
    @staticmethod
    def segmentation_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray) -> Dict:
        """
        Compute pixel-level segmentation metrics (e.g., crop type maps).
        
        Args:
            y_true: Ground truth segmentation map (H, W)
            y_pred: Predicted segmentation map (H, W)
            
        Returns:
            Dict with pixel accuracy and per-class metrics
        """
        # Flatten maps
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Pixel accuracy
        pixel_accuracy = accuracy_score(y_true_flat, y_pred_flat)
        
        # Per-class IoU (Intersection over Union)
        classes = np.unique(y_true_flat)
        iou_scores = {}
        
        for cls in classes:
            intersection = np.sum((y_true_flat == cls) & (y_pred_flat == cls))
            union = np.sum((y_true_flat == cls) | (y_pred_flat == cls))
            iou = intersection / (union + 1e-8) if union > 0 else 0
            iou_scores[int(cls)] = float(iou)
        
        mean_iou = np.mean(list(iou_scores.values())) if iou_scores else 0
        
        return {
            'pixel_accuracy': float(pixel_accuracy),
            'mean_iou': float(mean_iou),
            'per_class_iou': iou_scores,
            'n_classes': len(classes),
            'image_shape': y_true.shape
        }
    
    @staticmethod
    def temporal_metrics(y_true: List[np.ndarray],
                        y_pred: List[np.ndarray],
                        metric_type: str = 'classification') -> Dict:
        """
        Compute metrics for temporal sequences.
        
        Args:
            y_true: List of temporal ground truth arrays
            y_pred: List of temporal predictions
            metric_type: 'classification' or 'regression'
            
        Returns:
            Temporal metrics with per-timestep breakdowns
        """
        per_timestep = {}
        
        for t, (yt, yp) in enumerate(zip(y_true, y_pred)):
            if metric_type == 'classification':
                metrics = SICKLEMetrics.classification_metrics(yt, yp)
            else:
                metrics = SICKLEMetrics.yield_prediction_metrics(yt, yp)
            
            per_timestep[f't_{t}'] = metrics
        
        # Aggregate
        if metric_type == 'classification':
            all_true = np.concatenate(y_true)
            all_pred = np.concatenate(y_pred)
            aggregate = SICKLEMetrics.classification_metrics(all_true, all_pred)
        else:
            all_true = np.concatenate(y_true)
            all_pred = np.concatenate(y_pred)
            aggregate = SICKLEMetrics.yield_prediction_metrics(all_true, all_pred)
        
        return {
            'per_timestep': per_timestep,
            'aggregate': aggregate,
            'n_timesteps': len(y_true)
        }


class EvaluationPipeline:
    """
    Complete evaluation pipeline for SICKLE++ models.
    """
    
    def __init__(self, 
                 output_dir: Optional[Union[str, Path]] = None,
                 verbose: bool = True):
        """
        Initialize evaluation pipeline.
        
        Args:
            output_dir: Directory to save results
            verbose: Print progress
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = verbose
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_crop_classification(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    class_names: Optional[List[str]] = None,
                                    save_results: bool = True) -> Dict:
        """
        Evaluate crop classification task.
        
        Args:
            y_true: Ground truth crop classes
            y_pred: Predicted crop classes
            class_names: Crop class names
            save_results: Save results to file
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("🌾 Evaluating crop classification...")
        
        metrics = SICKLEMetrics.classification_metrics(y_true, y_pred, class_names)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("CROP CLASSIFICATION RESULTS")
            print(f"{'='*60}")
            print(f"Accuracy:        {metrics['accuracy']:.4f}")
            print(f"F1 (macro):      {metrics['f1_macro']:.4f}")
            print(f"F1 (weighted):   {metrics['f1_weighted']:.4f}")
            print(f"Samples:         {metrics['n_samples']}")
            print(f"Classes:         {metrics['n_unique_classes']}")
        
        if save_results and self.output_dir:
            results_path = self.output_dir / 'crop_classification_results.json'
            self._save_metrics(metrics, results_path)
            
            # Save confusion matrix plot
            self._plot_confusion_matrix(
                np.array(metrics['confusion_matrix']),
                self.output_dir / 'confusion_matrix.png',
                class_names
            )
        
        return metrics
    
    def evaluate_phenology_prediction(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     save_results: bool = True) -> Dict:
        """
        Evaluate phenology prediction task.
        
        Args:
            y_true: Ground truth dates
            y_pred: Predicted dates
            save_results: Save results to file
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("📅 Evaluating phenology prediction...")
        
        metrics = SICKLEMetrics.phenology_metrics(y_true, y_pred)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("PHENOLOGY PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"MAE (days):              {metrics['mae_days']:.2f}")
            print(f"RMSE (days):             {metrics['rmse_days']:.2f}")
            print(f"Mean Percentage Error:   {metrics['mean_percentage_error']:.2f}%")
            print(f"Samples:                 {metrics['n_samples']}")
        
        if save_results and self.output_dir:
            results_path = self.output_dir / 'phenology_results.json'
            self._save_metrics(metrics, results_path)
            
            # Save prediction vs ground truth plot
            self._plot_predictions(y_true, y_pred, 
                                   self.output_dir / 'phenology_predictions.png',
                                   xlabel='Day of Year')
        
        return metrics
    
    def evaluate_yield_prediction(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 save_results: bool = True) -> Dict:
        """
        Evaluate yield prediction task.
        
        Args:
            y_true: Ground truth yield values
            y_pred: Predicted yield values
            save_results: Save results to file
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("🎯 Evaluating yield prediction...")
        
        metrics = SICKLEMetrics.yield_prediction_metrics(y_true, y_pred)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("YIELD PREDICTION RESULTS (BASELINE TARGETS)")
            print(f"{'='*60}")
            print(f"RMSE:          {metrics['rmse']:.4f}  (target: 0.4-0.6)")
            print(f"MAE:           {metrics['mae']:.4f}")
            print(f"R² Score:      {metrics['r2_score']:.4f}")
            print(f"MAPE:          {metrics['mape']:.2f}%")
            print(f"NRMSE:         {metrics['nrmse']:.4f}")
            print(f"Yield Range:   {metrics['min_yield']:.2f} - {metrics['max_yield']:.2f}")
            print(f"Samples:       {metrics['n_samples']}")
        
        if save_results and self.output_dir:
            results_path = self.output_dir / 'yield_results.json'
            self._save_metrics(metrics, results_path)
            
            # Save prediction vs ground truth plot
            self._plot_predictions(y_true, y_pred,
                                   self.output_dir / 'yield_predictions.png',
                                   xlabel='Yield (kg/acre)')
        
        return metrics
    
    def evaluate_crop_map(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         save_results: bool = True) -> Dict:
        """
        Evaluate crop type segmentation map.
        
        Args:
            y_true: Ground truth crop map (H, W)
            y_pred: Predicted crop map (H, W)
            save_results: Save results to file
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("🗺️ Evaluating crop map segmentation...")
        
        metrics = SICKLEMetrics.segmentation_metrics(y_true, y_pred)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("CROP MAP RESULTS")
            print(f"{'='*60}")
            print(f"Pixel Accuracy:  {metrics['pixel_accuracy']:.4f}")
            print(f"Mean IoU:        {metrics['mean_iou']:.4f}")
            print(f"Classes:         {metrics['n_classes']}")
            print(f"Image Shape:     {metrics['image_shape']}")
        
        if save_results and self.output_dir:
            results_path = self.output_dir / 'crop_map_results.json'
            self._save_metrics(metrics, results_path)
            
            # Save map visualizations
            self._plot_crop_maps(y_true, y_pred, self.output_dir)
        
        return metrics
    
    @staticmethod
    def _save_metrics(metrics: Dict, output_path: Path):
        """Save metrics to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        metrics_serializable = convert_to_serializable(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"✅ Metrics saved: {output_path}")
    
    @staticmethod
    def _plot_confusion_matrix(cm: np.ndarray,
                               output_path: Path,
                               class_names: Optional[List[str]] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        if class_names and len(class_names) == cm.shape[0]:
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                       yticklabels=class_names, cmap='Blues')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"✅ Confusion matrix saved: {output_path}")
    
    @staticmethod
    def _plot_predictions(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         output_path: Path,
                         xlabel: str = 'Value'):
        """Plot predictions vs ground truth."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        axes[0].set_xlabel(f'True {xlabel}')
        axes[0].set_ylabel(f'Predicted {xlabel}')
        axes[0].set_title('Predictions vs Ground Truth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"✅ Prediction plot saved: {output_path}")
    
    @staticmethod
    def _plot_crop_maps(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       output_dir: Path):
        """Plot crop type maps."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # True map
        im1 = axes[0].imshow(y_true, cmap='tab20')
        axes[0].set_title('Ground Truth Crop Map')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Predicted map
        im2 = axes[1].imshow(y_pred, cmap='tab20')
        axes[1].set_title('Predicted Crop Map')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'crop_maps.png', dpi=150)
        plt.close()
        
        logger.info(f"✅ Crop maps saved: {output_dir / 'crop_maps.png'}")
