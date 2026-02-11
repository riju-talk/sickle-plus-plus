import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import pickle
from pathlib import Path


class SICKLEModel(nn.Module):
    """
    SICKLE++ style model for crop monitoring and yield prediction.
    """
    
    def __init__(self, 
                 input_channels: int,
                 num_classes: int = 1,
                 hidden_dim: int = 128,
                 dropout: float = 0.3):
        super(SICKLEModel, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Dropout2d(dropout)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


class ModelInference:
    """
    Model inference wrapper for SICKLE++ predictions.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        Initialize model inference.
        
        Args:
            model_path: Path to trained model file
            device: Device for inference ('cpu', 'cuda', 'auto')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_metadata = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model from file.
        
        Args:
            model_path: Path to model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model state
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.model = SICKLEModel(**config)
        else:
            # Default configuration
            input_channels = checkpoint.get('input_channels', 10)
            num_classes = checkpoint.get('num_classes', 1)
            self.model = SICKLEModel(input_channels, num_classes)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessing parameters
        if 'scaler' in checkpoint:
            self.scaler = checkpoint['scaler']
        if 'feature_names' in checkpoint:
            self.feature_names = checkpoint['feature_names']
        if 'metadata' in checkpoint:
            self.model_metadata = checkpoint['metadata']
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model config: {self.model_metadata}")
    
    def preprocess_features(self, features: np.ndarray) -> torch.Tensor:
        """
        Preprocess features for model input.
        
        Args:
            features: Input feature array (H, W, C)
            
        Returns:
            Preprocessed tensor
        """
        # Handle different input shapes
        if features.ndim == 3:
            # Single image (H, W, C) -> (1, C, H, W)
            features = features.transpose(2, 0, 1)  # (C, H, W)
            features = np.expand_dims(features, 0)  # (1, C, H, W)
        elif features.ndim == 4:
            # Batch of images (B, H, W, C) -> (B, C, H, W)
            features = features.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported feature shape: {features.shape}")
        
        # Convert to tensor
        tensor = torch.FloatTensor(features)
        
        # Apply scaling if available
        if self.scaler is not None:
            # Reshape for scaling
            original_shape = tensor.shape
            tensor_flat = tensor.view(-1, tensor.shape[1])
            
            # Apply scaling per channel
            for i in range(tensor.shape[1]):
                if hasattr(self.scaler, 'transform'):
                    # Sklearn scaler
                    tensor_flat[:, i] = torch.FloatTensor(
                        self.scaler.transform(tensor_flat[:, i:i+1].numpy()).flatten()
                    )
                elif isinstance(self.scaler, dict):
                    # Custom scaling parameters
                    mean = self.scaler.get('mean', 0)
                    std = self.scaler.get('std', 1)
                    if isinstance(mean, (list, np.ndarray)):
                        mean = mean[i] if i < len(mean) else 0
                    if isinstance(std, (list, np.ndarray)):
                        std = std[i] if i < len(std) else 1
                    tensor_flat[:, i] = (tensor_flat[:, i] - mean) / (std + 1e-8)
            
            tensor = tensor_flat.view(original_shape)
        
        # Handle NaN values
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        return tensor.to(self.device)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run model inference on features.
        
        Args:
            features: Feature array (H, W, C) or (B, H, W, C)
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        input_tensor = self.preprocess_features(features)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Apply appropriate activation
            if self.model.num_classes == 1:
                # Regression or binary classification
                predictions = outputs
            else:
                # Multi-class classification
                predictions = torch.softmax(outputs, dim=1)
            
            # Convert to numpy
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def predict_crop_yield(self, features: np.ndarray) -> Dict:
        """
        Predict crop yield with confidence metrics.
        
        Args:
            features: Feature array
            
        Returns:
            Dictionary with yield predictions and metadata
        """
        predictions = self.predict(features)
        
        # Calculate statistics
        mean_yield = float(np.mean(predictions))
        std_yield = float(np.std(predictions))
        min_yield = float(np.min(predictions))
        max_yield = float(np.max(predictions))
        
        result = {
            'mean_yield': mean_yield,
            'std_yield': std_yield,
            'min_yield': min_yield,
            'max_yield': max_yield,
            'yield_map': predictions.squeeze(),
            'confidence': 1.0 / (1.0 + std_yield),  # Simple confidence metric
            'pixel_count': predictions.size,
            'model_info': self.model_metadata
        }
        
        return result
    
    def predict_crop_classification(self, features: np.ndarray) -> Dict:
        """
        Predict crop types with class probabilities.
        
        Args:
            features: Feature array
            
        Returns:
            Dictionary with classification results
        """
        predictions = self.predict(features)
        
        if predictions.shape[-1] == 1:
            # Binary classification
            probabilities = torch.sigmoid(torch.FloatTensor(predictions)).numpy()
            class_predictions = (probabilities > 0.5).astype(int)
        else:
            # Multi-class classification
            probabilities = predictions
            class_predictions = np.argmax(predictions, axis=-1)
        
        result = {
            'class_predictions': class_predictions.squeeze(),
            'class_probabilities': probabilities.squeeze(),
            'confidence': np.max(probabilities, axis=-1).squeeze(),
            'pixel_count': predictions.shape[0] * predictions.shape[1] if predictions.ndim > 2 else predictions.size,
            'model_info': self.model_metadata
        }
        
        return result
    
    def batch_predict(self, 
                     feature_list: List[np.ndarray],
                     batch_size: int = 8) -> List[np.ndarray]:
        """
        Run batch predictions on multiple feature arrays.
        
        Args:
            feature_list: List of feature arrays
            batch_size: Batch size for processing
            
        Returns:
            List of prediction arrays
        """
        predictions = []
        
        for i in range(0, len(feature_list), batch_size):
            batch = feature_list[i:i+batch_size]
            
            # Stack batch
            if len(batch) == 1:
                batch_array = batch[0]
            else:
                batch_array = np.stack(batch, axis=0)
            
            # Predict
            batch_predictions = self.predict(batch_array)
            
            # Split batch results
            if len(batch) == 1:
                predictions.append(batch_predictions)
            else:
                for j in range(len(batch)):
                    predictions.append(batch_predictions[j])
        
        return predictions


def create_dummy_model(input_channels: int = 10, 
                      num_classes: int = 1,
                      save_path: str = None) -> ModelInference:
    """
    Create a dummy model for testing purposes.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        save_path: Optional path to save the dummy model
        
    Returns:
        ModelInference instance with dummy model
    """
    # Create model
    model = SICKLEModel(input_channels, num_classes)
    
    # Create dummy metadata
    metadata = {
        'model_type': 'SICKLEModel',
        'input_channels': input_channels,
        'num_classes': num_classes,
        'training_date': '2024-01-01',
        'version': '1.0.0'
    }
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'input_channels': input_channels,
            'num_classes': num_classes
        },
        'metadata': metadata
    }
    
    # Save if path provided
    if save_path:
        torch.save(checkpoint, save_path)
        print(f"Dummy model saved to {save_path}")
    
    # Create inference instance
    inference = ModelInference()
    inference.model = model
    inference.model_metadata = metadata
    
    return inference