"""
Video deepfake detection model using EfficientNet/Xception-inspired architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import cv2


class VideoDeepfakeDetector(nn.Module):
    """
    CNN-based model for video deepfake detection
    Inspired by EfficientNet with attention mechanisms
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize the video deepfake detector
        
        Args:
            num_classes: Number of output classes (2 for fake/real)
        """
        super(VideoDeepfakeDetector, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Inverted residual blocks (MobileNet-style)
        self.block1 = self._make_block(64, 128, num_blocks=2, stride=2)
        self.block2 = self._make_block(128, 256, num_blocks=3, stride=2)
        self.block3 = self._make_block(256, 512, num_blocks=3, stride=2)
        self.block4 = self._make_block(512, 1024, num_blocks=2, stride=2)
        
        # Attention mechanism
        self.attention = SpatialAttention()
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
    
    def _make_block(self, in_channels: int, out_channels: int,
                   num_blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a block of inverted residual layers"""
        layers = []
        
        # First block with stride
        layers.append(InvertedResidual(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(InvertedResidual(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 3, height, width)
            
        Returns:
            Output tensor with class logits
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(InvertedResidual, self).__init__()
        
        hidden_dim = in_channels * 4
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        layers = []
        
        # Expand
        if in_channels != hidden_dim:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Project
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        
        return x * attention


class VideoAnalyzer:
    """High-level video analysis class"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """
        Initialize video analyzer
        
        Args:
            model_path: Path to pretrained model weights
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = VideoDeepfakeDetector()
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
    
    def preprocess_frame(self, frame: np.ndarray,
                        target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        Preprocess frame for model input
        
        Args:
            frame: RGB frame as numpy array
            target_size: Target size for resizing
            
        Returns:
            Preprocessed tensor
        """
        # Resize
        frame_resized = cv2.resize(frame, target_size)
        
        # Normalize to [0, 1]
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Standardize (ImageNet statistics)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_standardized = (frame_normalized - mean) / std
        
        # Convert to tensor (C, H, W)
        tensor = torch.FloatTensor(frame_standardized).permute(2, 0, 1)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict_frame(self, frame: np.ndarray) -> Dict:
        """
        Predict if a frame contains deepfake
        
        Args:
            frame: RGB frame
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        input_tensor = self.preprocess_frame(frame).to(self.device)
        
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
    
    def analyze_video_sequence(self, frames: List[np.ndarray], 
                               faces: List[np.ndarray]) -> Dict:
        """
        Analyze a sequence of video frames
        
        Args:
            frames: List of video frames
            faces: List of detected faces
            
        Returns:
            Dictionary with analysis results
        """
        frame_predictions = []
        face_predictions = []
        
        # Analyze frames
        for frame in frames[:30]:  # Analyze first 30 frames
            pred = self.predict_frame(frame)
            frame_predictions.append(pred)
        
        # Analyze faces
        for face in faces[:20]:  # Analyze first 20 faces
            pred = self.predict_frame(face)
            face_predictions.append(pred)
        
        # Aggregate results
        avg_frame_fake_prob = np.mean([p['fake_probability'] for p in frame_predictions]) if frame_predictions else 0.5
        avg_face_fake_prob = np.mean([p['fake_probability'] for p in face_predictions]) if face_predictions else 0.5
        
        # Combine predictions (faces are more important)
        combined_fake_prob = 0.3 * avg_frame_fake_prob + 0.7 * avg_face_fake_prob
        
        return {
            'is_fake': combined_fake_prob > 0.5,
            'fake_probability': combined_fake_prob,
            'real_probability': 1 - combined_fake_prob,
            'confidence': abs(combined_fake_prob - 0.5) * 2,
            'frame_predictions': frame_predictions,
            'face_predictions': face_predictions
        }
    
    def identify_manipulated_frames(self, frames: List[np.ndarray],
                                   threshold: float = 0.6) -> List[int]:
        """
        Identify which frames appear to be manipulated
        
        Args:
            frames: List of frames
            threshold: Probability threshold for manipulation
            
        Returns:
            List of frame indices that appear manipulated
        """
        manipulated_indices = []
        
        for idx, frame in enumerate(frames):
            pred = self.predict_frame(frame)
            
            if pred['fake_probability'] > threshold:
                manipulated_indices.append(idx)
        
        return manipulated_indices
    
    def compute_integrity_score(self, video_data: Dict, 
                               prediction: Dict) -> float:
        """
        Compute video integrity score
        
        Args:
            video_data: Dictionary with video analysis data
            prediction: Prediction dictionary
            
        Returns:
            Integrity score (0-100)
        """
        # Start with model confidence
        base_score = prediction['real_probability'] * 100
        
        # Adjust based on blink analysis
        if 'blink_analysis' in video_data:
            blink_data = video_data['blink_analysis']
            if blink_data.get('is_unnatural', False):
                base_score *= 0.85
        
        # Adjust based on artifact analysis
        if 'artifact_analysis' in video_data:
            artifacts = video_data['artifact_analysis']
            num_with_artifacts = sum(1 for a in artifacts if a.get('has_artifacts', False))
            artifact_ratio = num_with_artifacts / len(artifacts) if artifacts else 0
            
            if artifact_ratio > 0.3:
                base_score *= (1 - artifact_ratio * 0.5)
        
        # Adjust based on face detection consistency
        total_faces = video_data.get('total_faces_detected', 0)
        num_frames = len(video_data.get('frames', []))
        if num_frames > 0:
            face_detection_rate = total_faces / num_frames
            if face_detection_rate < 0.5:  # Low face detection might indicate issues
                base_score *= 0.95
        
        return max(0.0, min(100.0, base_score))
