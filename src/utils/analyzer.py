"""
Integrated multimedia forensics analyzer
"""

import os
import sys
from typing import Dict, Optional, Tuple
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.preprocessing import AudioPreprocessor
from audio.detector import AudioAnalyzer
from video.preprocessing import VideoPreprocessor
from video.detector import VideoAnalyzer
from config.config import (
    AUDIO_CONFIG, VIDEO_CONFIG, MODEL_CONFIG, 
    THRESHOLD_CONFIG, VIZ_CONFIG
)


class MultimediaForensicsAnalyzer:
    """
    Main analyzer that integrates audio and video forensics pipelines
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the multimedia forensics analyzer
        
        Args:
            device: Device to run models on ('cpu' or 'cuda')
        """
        # Initialize audio pipeline
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=AUDIO_CONFIG['sample_rate'],
            n_mfcc=AUDIO_CONFIG['n_mfcc'],
            n_fft=AUDIO_CONFIG['n_fft'],
            hop_length=AUDIO_CONFIG['hop_length'],
            n_mels=AUDIO_CONFIG['n_mels']
        )
        
        self.audio_analyzer = AudioAnalyzer(
            model_path=MODEL_CONFIG.get('audio_model_path'),
            device=device
        )
        
        # Initialize video pipeline
        self.video_preprocessor = VideoPreprocessor(
            frame_sample_rate=VIDEO_CONFIG['frame_sample_rate'],
            face_detection_threshold=VIDEO_CONFIG['face_detection_threshold']
        )
        
        self.video_analyzer = VideoAnalyzer(
            model_path=MODEL_CONFIG.get('video_model_path'),
            device=device
        )
        
        self.device = device
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Perform complete audio analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio analysis results
        """
        print(f"Analyzing audio: {audio_path}")
        
        # Extract features
        features = self.audio_preprocessor.extract_all_features(audio_path)
        
        # Run deepfake detection
        prediction = self.audio_analyzer.predict(features['mfcc'])
        
        # Identify anomalous regions
        anomaly_regions = self.audio_analyzer.analyze_temporal_anomalies(
            features['mfcc']
        )
        
        # Compute integrity score
        integrity_score = self.audio_analyzer.compute_integrity_score(
            features, prediction
        )
        
        return {
            'features': features,
            'prediction': prediction,
            'anomaly_regions': anomaly_regions,
            'integrity_score': integrity_score,
            'is_authentic': integrity_score >= 50.0,
            'analysis_summary': self._create_audio_summary(
                prediction, features, integrity_score
            )
        }
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        Perform complete video analysis
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video analysis results
        """
        print(f"Analyzing video: {video_path}")
        
        # Process video
        video_data = self.video_preprocessor.process_video(video_path)
        
        # Run deepfake detection
        prediction = self.video_analyzer.analyze_video_sequence(
            video_data['frames'],
            video_data['extracted_faces']
        )
        
        # Identify manipulated frames
        manipulated_frames = self.video_analyzer.identify_manipulated_frames(
            video_data['frames'][:30]  # Check first 30 frames
        )
        
        # Compute integrity score
        integrity_score = self.video_analyzer.compute_integrity_score(
            video_data, prediction
        )
        
        return {
            'video_data': video_data,
            'prediction': prediction,
            'manipulated_frames': manipulated_frames,
            'integrity_score': integrity_score,
            'is_authentic': integrity_score >= 50.0,
            'analysis_summary': self._create_video_summary(
                prediction, video_data, integrity_score
            )
        }
    
    def analyze_multimedia(self, video_path: str, 
                          audio_path: Optional[str] = None) -> Dict:
        """
        Perform integrated audio and video analysis
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file (optional, can be extracted from video)
            
        Returns:
            Dictionary with complete analysis results
        """
        print("=" * 60)
        print("MULTIMEDIA FORENSICS ANALYSIS")
        print("=" * 60)
        
        # Analyze video
        video_results = self.analyze_video(video_path)
        
        # Analyze audio
        if audio_path:
            audio_results = self.analyze_audio(audio_path)
        else:
            # Extract audio from video if not provided
            audio_results = self._extract_and_analyze_audio(video_path)
        
        # Compute combined integrity score
        combined_score = self._compute_combined_score(
            audio_results['integrity_score'],
            video_results['integrity_score']
        )
        
        # Determine final verdict
        is_authentic = combined_score >= 50.0
        
        return {
            'audio_analysis': audio_results,
            'video_analysis': video_results,
            'combined_integrity_score': combined_score,
            'is_authentic': is_authentic,
            'verdict': 'AUTHENTIC' if is_authentic else 'MANIPULATED',
            'confidence': abs(combined_score - 50.0) * 2,  # 0-100 scale
            'detailed_summary': self._create_detailed_summary(
                audio_results, video_results, combined_score
            )
        }
    
    def _extract_and_analyze_audio(self, video_path: str) -> Dict:
        """
        Extract audio from video and analyze it
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio analysis results
        """
        # This is a simplified version - in production, use ffmpeg or moviepy
        # For now, return placeholder results
        print("Note: Audio extraction from video not fully implemented")
        print("Using video-only analysis")
        
        return {
            'features': None,
            'prediction': {'is_fake': False, 'fake_probability': 0.5},
            'anomaly_regions': [],
            'integrity_score': 50.0,
            'is_authentic': True,
            'analysis_summary': 'Audio analysis unavailable'
        }
    
    def _compute_combined_score(self, audio_score: float, 
                                video_score: float) -> float:
        """
        Compute combined integrity score from audio and video
        
        Args:
            audio_score: Audio integrity score (0-100)
            video_score: Video integrity score (0-100)
            
        Returns:
            Combined score (0-100)
        """
        weights = THRESHOLD_CONFIG['integrity_weights']
        
        combined = (
            audio_score * weights['audio'] +
            video_score * weights['video']
        )
        
        return max(0.0, min(100.0, combined))
    
    def _create_audio_summary(self, prediction: Dict, 
                             features: Dict, score: float) -> str:
        """Create human-readable audio analysis summary"""
        
        verdict = "FAKE" if prediction['is_fake'] else "REAL"
        confidence = prediction['confidence'] * 100
        
        summary = f"Audio Analysis: Classified as {verdict} "
        summary += f"(Confidence: {confidence:.1f}%)\n"
        summary += f"Integrity Score: {score:.1f}%\n"
        
        # Add feature insights
        silence_ratio = features['silence_analysis']['silence_ratio']
        if silence_ratio > 0.3:
            summary += f"⚠ High silence ratio detected ({silence_ratio:.1%})\n"
        
        hf_ratio = features['high_freq_analysis']['high_freq_ratio']
        if hf_ratio < 0.05:
            summary += f"⚠ Anomalous high-frequency content ({hf_ratio:.3f})\n"
        
        return summary
    
    def _create_video_summary(self, prediction: Dict,
                             video_data: Dict, score: float) -> str:
        """Create human-readable video analysis summary"""
        
        verdict = "FAKE" if prediction['is_fake'] else "REAL"
        confidence = prediction['confidence'] * 100
        
        summary = f"Video Analysis: Classified as {verdict} "
        summary += f"(Confidence: {confidence:.1f}%)\n"
        summary += f"Integrity Score: {score:.1f}%\n"
        
        # Add detection insights
        total_faces = video_data['total_faces_detected']
        summary += f"Faces detected: {total_faces}\n"
        
        if video_data.get('blink_analysis', {}).get('is_unnatural'):
            summary += "⚠ Unnatural blinking pattern detected\n"
        
        artifact_count = sum(
            1 for a in video_data.get('artifact_analysis', [])
            if a.get('has_artifacts', False)
        )
        if artifact_count > 0:
            summary += f"⚠ Blending artifacts detected in {artifact_count} faces\n"
        
        return summary
    
    def _create_detailed_summary(self, audio_results: Dict,
                                video_results: Dict,
                                combined_score: float) -> str:
        """Create detailed analysis summary"""
        
        summary = "\n" + "=" * 60 + "\n"
        summary += "FORENSICS ANALYSIS REPORT\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"OVERALL INTEGRITY SCORE: {combined_score:.1f}%\n"
        summary += f"VERDICT: {'AUTHENTIC' if combined_score >= 50 else 'MANIPULATED'}\n\n"
        
        summary += "AUDIO ANALYSIS:\n"
        summary += "-" * 40 + "\n"
        summary += audio_results['analysis_summary'] + "\n"
        
        summary += "VIDEO ANALYSIS:\n"
        summary += "-" * 40 + "\n"
        summary += video_results['analysis_summary'] + "\n"
        
        summary += "=" * 60 + "\n"
        
        return summary
