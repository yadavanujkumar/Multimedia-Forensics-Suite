"""
Visualization utilities for explainability layer
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from typing import List, Tuple, Dict
import io
from PIL import Image


class VideoVisualizer:
    """Handles video visualization with bounding boxes"""
    
    def __init__(self, bbox_color_real: Tuple[int, int, int] = (0, 255, 0),
                 bbox_color_fake: Tuple[int, int, int] = (255, 0, 0),
                 bbox_thickness: int = 2):
        """
        Initialize video visualizer
        
        Args:
            bbox_color_real: Color for real faces (RGB)
            bbox_color_fake: Color for fake faces (RGB)
            bbox_thickness: Thickness of bounding box
        """
        self.bbox_color_real = bbox_color_real
        self.bbox_color_fake = bbox_color_fake
        self.bbox_thickness = bbox_thickness
    
    def draw_detection_box(self, frame: np.ndarray, 
                          detection: Dict, 
                          is_fake: bool,
                          confidence: float) -> np.ndarray:
        """
        Draw bounding box on frame
        
        Args:
            frame: RGB frame
            detection: Face detection dictionary with 'box' key
            is_fake: Whether detection is classified as fake
            confidence: Confidence score
            
        Returns:
            Frame with bounding box drawn
        """
        frame_copy = frame.copy()
        
        # Get bounding box coordinates
        x, y, w, h = detection['box']
        
        # Choose color based on classification
        color = self.bbox_color_fake if is_fake else self.bbox_color_real
        
        # Draw rectangle
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), 
                     color, self.bbox_thickness)
        
        # Add label
        label = f"{'FAKE' if is_fake else 'REAL'}: {confidence:.2%}"
        label_bg_color = color
        
        # Calculate label size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw label background
        cv2.rectangle(frame_copy, (x, y - label_h - 10), 
                     (x + label_w + 10, y), label_bg_color, -1)
        
        # Draw label text
        cv2.putText(frame_copy, label, (x + 5, y - 5),
                   font, font_scale, (255, 255, 255), font_thickness)
        
        return frame_copy
    
    def annotate_frame(self, frame: np.ndarray,
                      detections: List[Dict],
                      predictions: List[Dict]) -> np.ndarray:
        """
        Annotate frame with all detections
        
        Args:
            frame: RGB frame
            detections: List of face detections
            predictions: List of predictions for each detection
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for detection, prediction in zip(detections, predictions):
            annotated = self.draw_detection_box(
                annotated,
                detection,
                prediction['is_fake'],
                prediction['confidence']
            )
        
        return annotated
    
    def create_annotated_video(self, frames: List[np.ndarray],
                              detections_per_frame: List[List[Dict]],
                              predictions_per_frame: List[List[Dict]],
                              output_path: str = None,
                              fps: float = 30.0) -> List[np.ndarray]:
        """
        Create annotated video with bounding boxes
        
        Args:
            frames: List of video frames
            detections_per_frame: List of detections for each frame
            predictions_per_frame: List of predictions for each frame
            output_path: Path to save video (optional)
            fps: Frames per second
            
        Returns:
            List of annotated frames
        """
        annotated_frames = []
        
        for frame, detections, predictions in zip(frames, detections_per_frame, predictions_per_frame):
            if detections and predictions:
                annotated = self.annotate_frame(frame, detections, predictions)
            else:
                annotated = frame.copy()
            
            annotated_frames.append(annotated)
        
        # Save video if output path is provided
        if output_path and annotated_frames:
            height, width = annotated_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in annotated_frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
        
        return annotated_frames


class AudioVisualizer:
    """Handles audio visualization with anomaly highlighting"""
    
    def __init__(self, cmap: str = 'viridis'):
        """
        Initialize audio visualizer
        
        Args:
            cmap: Colormap for spectrogram
        """
        self.cmap = cmap
    
    def create_spectrogram_with_highlights(self, 
                                          mel_spec: np.ndarray,
                                          anomaly_regions: List[Tuple[float, float]],
                                          sample_rate: int = 16000,
                                          hop_length: int = 512) -> plt.Figure:
        """
        Create spectrogram visualization with anomaly highlights
        
        Args:
            mel_spec: Mel spectrogram (already in dB)
            anomaly_regions: List of (start_time, end_time) tuples
            sample_rate: Audio sample rate
            hop_length: Hop length used in STFT
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Display spectrogram
        img = librosa.display.specshow(
            mel_spec,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap=self.cmap
        )
        
        # Highlight anomaly regions
        for start, end in anomaly_regions:
            ax.axvspan(start, end, alpha=0.4, color='red', 
                      label='Synthetic Artifact' if start == anomaly_regions[0][0] else '')
        
        # Styling
        ax.set_title('Audio Spectrogram - Deepfake Detection Analysis', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time (seconds)', fontsize=13)
        ax.set_ylabel('Frequency (Hz)', fontsize=13)
        
        # Add colorbar
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Intensity (dB)', fontsize=12)
        
        # Add legend if there are anomalies
        if anomaly_regions:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                     loc='upper right', fontsize=11)
        
        plt.tight_layout()
        
        return fig
    
    def create_waveform_plot(self, audio: np.ndarray, 
                            sample_rate: int = 16000,
                            anomaly_regions: List[Tuple[float, float]] = None) -> plt.Figure:
        """
        Create waveform visualization
        
        Args:
            audio: Audio time series
            sample_rate: Sample rate
            anomaly_regions: List of anomalous time regions
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # Create time axis
        time = np.arange(len(audio)) / sample_rate
        
        # Plot waveform
        ax.plot(time, audio, color='steelblue', linewidth=0.5)
        ax.fill_between(time, audio, alpha=0.3, color='steelblue')
        
        # Highlight anomaly regions
        if anomaly_regions:
            for start, end in anomaly_regions:
                ax.axvspan(start, end, alpha=0.3, color='red',
                          label='Suspicious Region' if start == anomaly_regions[0][0] else '')
        
        ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if anomaly_regions:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        return fig
    
    def create_mfcc_heatmap(self, mfcc: np.ndarray) -> plt.Figure:
        """
        Create MFCC coefficient heatmap
        
        Args:
            mfcc: MFCC features
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot heatmap
        im = ax.imshow(mfcc, aspect='auto', origin='lower', 
                      cmap='coolwarm', interpolation='nearest')
        
        ax.set_title('MFCC Features Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Frame', fontsize=12)
        ax.set_ylabel('MFCC Coefficient', fontsize=12)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Coefficient Value', fontsize=11)
        
        plt.tight_layout()
        
        return fig


class IntegrityScoreVisualizer:
    """Visualize integrity scores"""
    
    @staticmethod
    def create_score_gauge(score: float, 
                          title: str = "Integrity Score") -> plt.Figure:
        """
        Create a gauge visualization for integrity score
        
        Args:
            score: Score value (0-100)
            title: Title for the gauge
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Background arcs
        ax.plot(theta, [1]*100, color='lightgray', linewidth=20, alpha=0.3)
        
        # Score arc
        score_theta = theta[:int(score)]
        
        # Color based on score
        if score >= 70:
            color = 'green'
        elif score >= 40:
            color = 'orange'
        else:
            color = 'red'
        
        ax.plot(score_theta, [1]*len(score_theta), color=color, 
               linewidth=20, alpha=0.8)
        
        # Configure axes
        ax.set_ylim(0, 1.5)
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        
        # Add score text
        ax.text(np.pi/2, 0.5, f'{score:.1f}%', 
               ha='center', va='center', fontsize=36, fontweight='bold')
        
        # Add title
        ax.text(np.pi/2, 1.3, title,
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Add labels
        ax.text(0, 1.1, 'FAKE', ha='right', va='center', fontsize=12, color='red')
        ax.text(np.pi, 1.1, 'REAL', ha='left', va='center', fontsize=12, color='green')
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def create_breakdown_chart(audio_score: float, 
                              video_score: float,
                              combined_score: float) -> plt.Figure:
        """
        Create a breakdown chart showing component scores
        
        Args:
            audio_score: Audio integrity score
            video_score: Video integrity score
            combined_score: Combined integrity score
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        components = ['Audio', 'Video', 'Combined']
        scores = [audio_score, video_score, combined_score]
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        bars = ax1.bar(components, scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax1.set_ylabel('Integrity Score (%)', fontsize=12)
        ax1.set_title('Component Scores Breakdown', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart for weights
        weights = [40, 60]  # Audio: 40%, Video: 60%
        labels = ['Audio\n(40%)', 'Video\n(60%)']
        colors_pie = ['skyblue', 'lightcoral']
        
        ax2.pie(weights, labels=labels, colors=colors_pie, autopct='',
               startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Weight Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return fig


def fig_to_array(fig: plt.Figure) -> np.ndarray:
    """
    Convert matplotlib figure to numpy array
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        RGB numpy array
    """
    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Load as PIL image and convert to numpy
    img = Image.open(buf)
    img_array = np.array(img)
    
    buf.close()
    plt.close(fig)
    
    return img_array
