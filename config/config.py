"""
Configuration settings for the Multimedia Forensics Suite
"""

# Audio Configuration
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'n_mfcc': 40,
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 128,
    'supported_formats': ['.wav', '.mp3'],
    'max_duration': 300,  # seconds
}

# Video Configuration
VIDEO_CONFIG = {
    'supported_formats': ['.mp4', '.avi', '.mov'],
    'frame_sample_rate': 5,  # Process every Nth frame
    'face_detection_threshold': 0.9,
    'input_size': (224, 224),
    'max_duration': 600,  # seconds
}

# Model Configuration
MODEL_CONFIG = {
    'audio_model_path': 'models/audio_deepfake_detector.pth',
    'video_model_path': 'models/video_deepfake_detector.pth',
    'device': 'cpu',  # 'cuda' if available
}

# Analysis Thresholds
THRESHOLD_CONFIG = {
    'audio_fake_threshold': 0.5,
    'video_fake_threshold': 0.5,
    'integrity_weights': {
        'audio': 0.4,
        'video': 0.6,
    }
}

# Visualization
VIZ_CONFIG = {
    'bbox_color_real': (0, 255, 0),  # Green
    'bbox_color_fake': (255, 0, 0),  # Red
    'bbox_thickness': 2,
    'spectrogram_cmap': 'viridis',
}
