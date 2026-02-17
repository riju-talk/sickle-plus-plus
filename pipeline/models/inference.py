import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from pathlib import Path
import sys

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'models'))

try:
    from unet3d import UNet3D
    from unet3d_multitask import UNet3D as UNet3DMultitask
    from utae import UTAE
    from pastis_unet3d import PaSTiSUNet3D
except ImportError as e:
    print(f"Warning: Could not import all models: {e}")


class ModelInference:
    """
    Simplified inference wrapper for SICKLE++ agricultural models.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_type: str = 'utae',
                 device: str = 'auto'):
        """
        Initialize model inference.
        
        Args:
            model_path: Path to trained model file (.pth or .pt)
            model_type: Type of model ('utae', 'unet3d', 'unet3d_multitask', 'pastis')
            device: Device for inference ('cpu', 'cuda', 'auto')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        self.model = None
        self.model_type = model_type
        self.model_path = model_path
        self.input_channels = None
        self.num_classes = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from file."""
        print(f"🔄 Loading model: {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                self.input_channels = config.get('input_channels', 13)  # S1 + S2 default
                self.num_classes = config.get('num_classes', 20)  # PASTIS default
            elif 'config' in checkpoint:
                config = checkpoint['config']
                self.input_channels = config.get('input_channels', 13)
                self.num_classes = config.get('num_classes', 20)
            else:
                # Default configuration for agricultural analysis
                self.input_channels = 13  # S2 (10) + S1 (2) + derived (1)
                self.num_classes = 20     # Crop classes
            
            # Initialize model architecture
            self.model = self._create_model_architecture()
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully: {self.model_type}")
            print(f"   Input channels: {self.input_channels}")
            print(f"   Output classes: {self.num_classes}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _create_model_architecture(self):
        """Create model architecture based on model type."""
        try:
            if self.model_type == 'utae':
                model = UTAE(
                    input_dim=self.input_channels,
                    encoder_widths=[64, 64, 64, 128],
                    decoder_widths=[32, 32, 64, 128],
                    out_conv=[32, self.num_classes],
                    n_head=16,
                    d_model=256
                )
            elif self.model_type == 'unet3d':
                model = UNet3D(
                    in_channel=self.input_channels,
                    n_classes=self.num_classes,
                    timesteps=61,  # PASTIS default
                    dropout=0.2
                )
            elif self.model_type == 'unet3d_multitask':
                model = UNet3DMultitask(
                    in_channel=self.input_channels,
                    n_classes=self.num_classes,
                    timesteps=61,
                    dropout=0.2
                )
            elif self.model_type == 'pastis':
                model = PaSTiSUNet3D(
                    input_dim=self.input_channels,
                    num_classes=self.num_classes
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            return model
        
        except NameError as e:
            print(f"⚠️  Model class not available: {e}")
            print("   Using basic CNN architecture instead")
            return self._create_basic_model()
    
    def _create_basic_model(self):
        """Create a basic CNN model when specific architectures are not available."""
        class BasicCNN(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(input_dim, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((8, 8))
                self.classifier = nn.Linear(256 * 64, num_classes)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                if len(x.shape) == 5:  # (B, T, C, H, W) -> use last time step
                    x = x[:, -1, :, :, :]
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.classifier(x)
                return x
        
        return BasicCNN(self.input_channels, self.num_classes)
    
    def predict(self, data: np.ndarray, return_probabilities: bool = True) -> Dict:
        """
        Run inference on input data.
        
        Args:
            data: Input data array of shape (T, C, H, W) or (B, T, C, H, W)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with predictions and metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"🔮 Running inference...")
        print(f"   Input shape: {data.shape}")
        
        # Prepare input tensor
        if len(data.shape) == 4:
            # Add batch dimension: (T, C, H, W) -> (1, T, C, H, W)
            data = data[np.newaxis, ...]
        
        # Convert to tensor
        input_tensor = torch.from_numpy(data).float().to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(input_tensor)
            
            # Process outputs based on model type
            if isinstance(outputs, tuple):
                # Multi-task models may return multiple outputs
                predictions = outputs[0]
            else:
                predictions = outputs
            
            # Move to CPU and convert to numpy
            predictions = predictions.cpu().numpy()
            
            # Process predictions
            if return_probabilities and self.num_classes > 1:
                probabilities = torch.softmax(torch.from_numpy(predictions), dim=1).numpy()
                # Get class predictions
                class_predictions = np.argmax(predictions, axis=1)
                # Calculate confidence scores
                max_probs = np.max(probabilities, axis=1)
            else:
                # Binary or regression case
                probabilities = torch.sigmoid(torch.from_numpy(predictions)).numpy()
                class_predictions = (probabilities > 0.5).astype(int).squeeze()
                max_probs = np.maximum(probabilities, 1 - probabilities).squeeze()
            
            mean_confidence = np.mean(max_probs)
            
        results = {
            'class_predictions': class_predictions,
            'confidence_scores': max_probs,
            'mean_confidence': mean_confidence,
            'input_shape': data.shape,
            'output_shape': predictions.shape,
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'raw_predictions': predictions
        }
        
        if return_probabilities:
            results['probabilities'] = probabilities
        
        print(f"✅ Inference completed")
        print(f"   Mean confidence: {mean_confidence:.3f}")
        if len(class_predictions.shape) > 0:
            print(f"   Unique classes predicted: {len(np.unique(class_predictions))}")
        
        return results
    
    def predict_crop_classification(self, data: np.ndarray) -> Dict:
        """Agricultural crop classification prediction."""
        return self.predict(data, return_probabilities=True)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {'status': 'No model loaded'}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'device': str(self.device),
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Approximate size in MB
        }


class SICKLEInference:
    """
    SICKLE-specific inference wrapper with agricultural focus.
    """
    
    def __init__(self, model_path: str, model_type: str = 'utae'):
        """Initialize SICKLE inference system."""
        self.inference = ModelInference(model_path, model_type)
        
        # SICKLE crop class mapping (example for PASTIS dataset)
        self.crop_classes = {
            0: 'Background',
            1: 'Winter Wheat', 2: 'Winter Barley', 3: 'Winter Rapeseed',
            4: 'Spring Oats', 5: 'Corn', 6: 'Sunflower', 7: 'Winter Triticale',
            8: 'Durum Wheat', 9: 'Soybean', 10: 'Sugar Beet', 11: 'Potato',
            12: 'Winter Pea', 13: 'Spring Barley', 14: 'Meadow',
            15: 'Rapeseed', 16: 'Spring Wheat', 17: 'Fallow',
            18: 'Sorghum', 19: 'Other'
        }
    
    def predict_field(self, satellite_data: np.ndarray) -> Dict:
        """Predict crop types for agricultural field."""
        results = self.inference.predict_crop_classification(satellite_data)
        
        # Add agricultural-specific analysis
        class_pred = results['class_predictions']
        if len(class_pred.shape) > 0 and class_pred.shape[0] == 1:
            class_pred = class_pred[0]  # Remove batch dimension
        
        # Calculate crop area distribution
        unique_classes, counts = np.unique(class_pred, return_counts=True)
        total_pixels = class_pred.size
        
        crop_distribution = {}
        for class_id, count in zip(unique_classes, counts):
            crop_name = self.crop_classes.get(class_id, f'Unknown_{class_id}')
            percentage = (count / total_pixels) * 100
            crop_distribution[crop_name] = {
                'pixels': int(count),
                'percentage': round(percentage, 2)
            }
        
        # Find dominant crop
        dominant_class = unique_classes[np.argmax(counts)]
        dominant_crop = self.crop_classes.get(dominant_class, f'Unknown_{dominant_class}')
        dominant_percentage = max(crop_distribution.values(), key=lambda x: x['percentage'])['percentage']
        
        agricultural_results = {
            'dominant_crop': dominant_crop,
            'dominant_crop_percentage': dominant_percentage,
            'crop_distribution': crop_distribution,
            'field_diversity': len(unique_classes),
            'field_classification_confidence': results['mean_confidence']
        }
        
        # Combine with technical results
        results['agricultural_analysis'] = agricultural_results
        
        return results