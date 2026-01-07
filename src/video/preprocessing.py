"""
Video preprocessing and frame extraction module
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import os

# Try to import MTCNN, fall back to OpenCV Haar Cascades if not available
try:
    from mtcnn import MTCNN
    USE_MTCNN = True
except ImportError:
    USE_MTCNN = False
    print("MTCNN not available, using OpenCV Haar Cascade face detector")


class VideoPreprocessor:
    """Handles video frame extraction and face detection"""
    
    def __init__(self, frame_sample_rate: int = 5, 
                 face_detection_threshold: float = 0.9):
        """
        Initialize video preprocessor
        
        Args:
            frame_sample_rate: Process every Nth frame
            face_detection_threshold: Confidence threshold for face detection
        """
        self.frame_sample_rate = frame_sample_rate
        self.face_detection_threshold = face_detection_threshold
        
        if USE_MTCNN:
            self.face_detector = MTCNN(min_face_size=50)
            self.detector_type = 'mtcnn'
        else:
            # Use OpenCV's Haar Cascade face detector as fallback
            self.face_detector = None
            self.detector_type = 'opencv'
            # Initialize OpenCV face detector
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames list, video metadata)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        metadata = {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': duration
        }
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on sample rate
            if frame_idx % self.frame_sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        
        return frames, metadata
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a frame using MTCNN or OpenCV
        
        Args:
            frame: RGB frame
            
        Returns:
            List of face detections with bounding boxes and confidence
        """
        if self.detector_type == 'mtcnn':
            detections = self.face_detector.detect_faces(frame)
            
            # Filter by confidence threshold
            filtered_detections = [
                d for d in detections 
                if d['confidence'] >= self.face_detection_threshold
            ]
            
            return filtered_detections
        else:
            # Use OpenCV Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            
            # Convert to MTCNN-like format
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'box': [int(x), int(y), int(w), int(h)],
                    'confidence': 0.95,  # Haar cascades don't provide confidence
                    'keypoints': {}  # No keypoints from Haar
                })
            
            return detections
    
    def extract_face_regions(self, frame: np.ndarray, 
                            detections: List[Dict],
                            target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
        """
        Extract and resize face regions from frame
        
        Args:
            frame: RGB frame
            detections: List of face detections
            target_size: Target size for face crops
            
        Returns:
            List of cropped and resized face images
        """
        face_regions = []
        
        for detection in detections:
            x, y, w, h = detection['box']
            
            # Add padding
            padding = int(0.2 * max(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            # Extract face region
            face = frame[y1:y2, x1:x2]
            
            # Resize to target size
            if face.size > 0:
                face_resized = cv2.resize(face, target_size)
                face_regions.append(face_resized)
        
        return face_regions
    
    def analyze_blink_patterns(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze blinking patterns in video frames
        
        Args:
            frames: List of frames
            
        Returns:
            Dictionary with blink analysis
        """
        blink_count = 0
        eye_aspect_ratios = []
        
        for frame in frames:
            detections = self.detect_faces(frame)
            
            for detection in detections:
                # Check if landmarks are available
                if 'keypoints' in detection:
                    keypoints = detection['keypoints']
                    
                    # Calculate eye aspect ratio (simplified)
                    if 'left_eye' in keypoints and 'right_eye' in keypoints:
                        # Placeholder for actual EAR calculation
                        # In production, would use facial landmarks
                        ear = 0.3  # Dummy value
                        eye_aspect_ratios.append(ear)
                        
                        # Detect blink (EAR threshold)
                        if ear < 0.2:
                            blink_count += 1
        
        # Calculate blink rate
        num_frames = len(frames)
        blink_rate = blink_count / num_frames if num_frames > 0 else 0
        
        return {
            'blink_count': blink_count,
            'blink_rate': blink_rate,
            'avg_eye_aspect_ratio': np.mean(eye_aspect_ratios) if eye_aspect_ratios else 0.3,
            'is_unnatural': blink_rate < 0.01 or blink_rate > 0.15
        }
    
    def detect_blending_artifacts(self, face: np.ndarray) -> Dict:
        """
        Detect blending inconsistencies in face region
        
        Args:
            face: Face image
            
        Returns:
            Dictionary with artifact analysis
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        
        # Compute Laplacian (edge detection) - high values indicate sharp transitions
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Compute frequency domain features
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Analyze frequency distribution
        high_freq_energy = np.sum(magnitude[magnitude > np.percentile(magnitude, 90)])
        total_energy = np.sum(magnitude)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Detect color inconsistencies (RGB channels)
        r, g, b = face[:, :, 0], face[:, :, 1], face[:, :, 2]
        color_std = np.std([r.std(), g.std(), b.std()])
        
        return {
            'laplacian_variance': laplacian_var,
            'high_freq_ratio': high_freq_ratio,
            'color_inconsistency': color_std,
            'has_artifacts': laplacian_var > 500 or high_freq_ratio > 0.15
        }
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process entire video for deepfake detection
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with all extracted information
        """
        # Extract frames
        frames, metadata = self.extract_frames(video_path)
        
        # Process frames
        all_faces = []
        face_detections_per_frame = []
        
        for frame in frames:
            detections = self.detect_faces(frame)
            face_detections_per_frame.append(detections)
            
            # Extract face regions
            faces = self.extract_face_regions(frame, detections)
            all_faces.extend(faces)
        
        # Analyze blink patterns
        blink_analysis = self.analyze_blink_patterns(frames)
        
        # Analyze artifacts in faces
        artifact_scores = []
        for face in all_faces[:10]:  # Analyze first 10 faces
            artifacts = self.detect_blending_artifacts(face)
            artifact_scores.append(artifacts)
        
        return {
            'frames': frames,
            'metadata': metadata,
            'face_detections': face_detections_per_frame,
            'extracted_faces': all_faces,
            'blink_analysis': blink_analysis,
            'artifact_analysis': artifact_scores,
            'total_faces_detected': len(all_faces)
        }
