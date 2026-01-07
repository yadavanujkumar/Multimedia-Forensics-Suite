"""
Audio preprocessing and feature extraction module for deepfake detection
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import io
from typing import Tuple, Dict
import soundfile as sf


class AudioPreprocessor:
    """Handles audio preprocessing and feature extraction"""
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 40, 
                 n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
        """
        Initialize audio preprocessor
        
        Args:
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients to extract
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of Mel bands
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Mel-frequency cepstral coefficients (MFCCs)
        
        Args:
            audio: Audio time series
            
        Returns:
            MFCC features as numpy array
        """
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Add delta and delta-delta features for better temporal modeling
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack features
        mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        return mfcc_features
    
    def generate_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Generate Mel spectrogram
        
        Args:
            audio: Audio time series
            
        Returns:
            Mel spectrogram as numpy array
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def detect_silence_patterns(self, audio: np.ndarray, 
                                threshold: float = 0.01) -> Dict:
        """
        Detect unnatural silence patterns in audio
        
        Args:
            audio: Audio time series
            threshold: Energy threshold for silence detection
            
        Returns:
            Dictionary with silence statistics
        """
        # Calculate short-time energy
        frame_length = 2048
        hop_length = 512
        
        energy = np.array([
            sum(abs(audio[i:i+frame_length]**2))
            for i in range(0, len(audio), hop_length)
        ])
        
        # Normalize energy
        energy = energy / np.max(energy) if np.max(energy) > 0 else energy
        
        # Detect silence frames
        silence_frames = energy < threshold
        silence_ratio = np.sum(silence_frames) / len(silence_frames)
        
        # Detect sudden transitions (potential artifacts)
        energy_diff = np.abs(np.diff(energy))
        sudden_transitions = np.sum(energy_diff > 0.5)
        
        return {
            'silence_ratio': silence_ratio,
            'sudden_transitions': sudden_transitions,
            'silence_frames': silence_frames
        }
    
    def analyze_high_frequency(self, audio: np.ndarray) -> Dict:
        """
        Analyze high-frequency components for synthetic artifacts
        
        Args:
            audio: Audio time series
            
        Returns:
            Dictionary with high-frequency analysis
        """
        # Compute STFT
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        
        # Analyze frequency bands
        nyquist = self.sample_rate / 2
        high_freq_start = int(magnitude.shape[0] * 0.7)  # Top 30% frequencies
        
        high_freq_energy = np.mean(magnitude[high_freq_start:, :])
        low_freq_energy = np.mean(magnitude[:high_freq_start, :])
        
        # Calculate ratio (synthetic audio often has anomalies in high frequencies)
        hf_ratio = high_freq_energy / (low_freq_energy + 1e-8)
        
        return {
            'high_freq_ratio': hf_ratio,
            'high_freq_energy': high_freq_energy,
            'low_freq_energy': low_freq_energy
        }
    
    def visualize_spectrogram(self, audio: np.ndarray, 
                             anomaly_regions: list = None) -> plt.Figure:
        """
        Create spectrogram visualization with anomaly highlighting
        
        Args:
            audio: Audio time series
            anomaly_regions: List of time regions with anomalies [(start, end), ...]
            
        Returns:
            Matplotlib figure
        """
        mel_spec_db = self.generate_mel_spectrogram(audio)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        img = librosa.display.specshow(
            mel_spec_db,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap='viridis'
        )
        
        # Highlight anomaly regions
        if anomaly_regions:
            for start, end in anomaly_regions:
                ax.axvspan(start, end, alpha=0.3, color='red', 
                          label='Potential Manipulation')
        
        ax.set_title('Mel Spectrogram with Anomaly Detection', fontsize=14)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        if anomaly_regions:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def extract_all_features(self, audio_path: str) -> Dict:
        """
        Extract all audio features for deepfake detection
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Extract features
        mfcc = self.extract_mfcc(audio)
        mel_spec = self.generate_mel_spectrogram(audio)
        silence_info = self.detect_silence_patterns(audio)
        hf_info = self.analyze_high_frequency(audio)
        
        return {
            'audio': audio,
            'sample_rate': sr,
            'mfcc': mfcc,
            'mel_spectrogram': mel_spec,
            'silence_analysis': silence_info,
            'high_freq_analysis': hf_info,
            'duration': len(audio) / sr
        }
