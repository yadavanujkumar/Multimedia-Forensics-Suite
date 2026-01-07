"""
Audio deepfake detection model using CNN/ResNet architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class AudioDeepfakeDetector(nn.Module):
    """
    CNN-based model for audio deepfake detection
    Inspired by ResNet architecture with focus on temporal patterns
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        """
        Initialize the audio deepfake detector
        
        Args:
            input_channels: Number of input channels (MFCC + deltas)
            num_classes: Number of output classes (2 for fake/real)
        """
        super(AudioDeepfakeDetector, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-style blocks
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels: int, out_channels: int, 
                   blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a residual layer"""
        layers = []
        
        # First block may have stride > 1
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Output tensor with class logits
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for the audio detector"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class AudioAnalyzer:
    """High-level audio analysis class"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """
        Initialize audio analyzer
        
        Args:
            model_path: Path to pretrained model weights
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = AudioDeepfakeDetector()
        self.model.to(device)
        
        # Load pretrained weights if available
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.eval()
                self.is_trained = True
            except FileNotFoundError:
                print(f"Warning: Model weights not found at {model_path}")
                print("Using untrained model for demonstration purposes")
                self.is_trained = False
        else:
            self.is_trained = False
    
    def preprocess_features(self, mfcc: np.ndarray, 
                          target_length: int = 500) -> torch.Tensor:
        """
        Preprocess MFCC features for model input
        
        Args:
            mfcc: MFCC features as numpy array
            target_length: Target temporal length
            
        Returns:
            Preprocessed tensor
        """
        # Pad or truncate to fixed length
        if mfcc.shape[1] < target_length:
            # Pad with zeros
            pad_width = target_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            mfcc = mfcc[:, :target_length]
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # Convert to tensor and add batch and channel dimensions
        tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def predict(self, mfcc: np.ndarray) -> Dict:
        """
        Predict if audio is fake or real
        
        Args:
            mfcc: MFCC features
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        input_tensor = self.preprocess_features(mfcc).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get prediction
            fake_prob = probabilities[0, 1].item()
            real_prob = probabilities[0, 0].item()
            is_fake = fake_prob > 0.5
        
        return {
            'is_fake': is_fake,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'confidence': max(fake_prob, real_prob)
        }
    
    def analyze_temporal_anomalies(self, mfcc: np.ndarray, 
                                  window_size: int = 50) -> list:
        """
        Analyze temporal windows to detect anomalous regions
        
        Args:
            mfcc: MFCC features
            window_size: Size of temporal window
            
        Returns:
            List of anomalous time regions
        """
        anomaly_regions = []
        num_windows = mfcc.shape[1] // window_size
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            
            # Extract window
            window = mfcc[:, start_idx:end_idx]
            
            # Quick prediction on window
            result = self.predict(window)
            
            if result['is_fake'] and result['confidence'] > 0.7:
                # Convert indices to time (assuming hop_length=512, sr=16000)
                start_time = start_idx * 512 / 16000
                end_time = end_idx * 512 / 16000
                anomaly_regions.append((start_time, end_time))
        
        return anomaly_regions
    
    def compute_integrity_score(self, features: Dict, 
                               prediction: Dict) -> float:
        """
        Compute audio integrity score based on multiple factors
        
        Args:
            features: Dictionary of extracted audio features
            prediction: Prediction dictionary from model
            
        Returns:
            Integrity score (0-100)
        """
        # Start with model confidence
        base_score = prediction['real_probability'] * 100
        
        # Adjust based on silence patterns
        silence_ratio = features['silence_analysis']['silence_ratio']
        if silence_ratio > 0.3:  # Excessive silence
            base_score *= 0.9
        
        # Adjust based on high-frequency analysis
        hf_ratio = features['high_freq_analysis']['high_freq_ratio']
        if hf_ratio < 0.05:  # Suspiciously low high-frequency content
            base_score *= 0.85
        
        # Adjust based on sudden transitions
        transitions = features['silence_analysis']['sudden_transitions']
        if transitions > 50:  # Many sudden transitions
            base_score *= 0.9
        
        return max(0.0, min(100.0, base_score))
