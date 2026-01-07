"""
Streamlit Web Application for Multimedia Forensics Suite
A drag-and-drop interface for deepfake detection
"""

import streamlit as st
import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.analyzer import MultimediaForensicsAnalyzer
from utils.visualization import (
    VideoVisualizer, AudioVisualizer, 
    IntegrityScoreVisualizer, fig_to_array
)
from audio.preprocessing import AudioPreprocessor
from config.config import AUDIO_CONFIG, VIDEO_CONFIG

# Page configuration
st.set_page_config(
    page_title="Multimedia Forensics Suite",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #2980b9, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 3rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer():
    """Load the multimedia forensics analyzer"""
    return MultimediaForensicsAnalyzer(device='cpu')


def display_header():
    """Display the application header"""
    st.markdown('<h1 class="main-header">🔍 Multimedia Forensics Suite</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Deepfake Detection & Truth Engine</p>', 
                unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with information"""
    with st.sidebar:
        st.header("About")
        st.info("""
        This platform uses advanced AI to detect manipulated 
        multimedia content including:
        
        🎵 **Audio Analysis**
        - Synthetic voice detection
        - MFCC & Spectrogram analysis
        - Temporal anomaly detection
        
        🎥 **Video Analysis**
        - Face-swap detection
        - Deepfake identification
        - Blinking pattern analysis
        
        📊 **Explainability**
        - Visual anomaly highlighting
        - Detailed forensic reports
        - Confidence scoring
        """)
        
        st.header("Supported Formats")
        st.write("**Audio:** WAV, MP3")
        st.write("**Video:** MP4, AVI, MOV")
        
        st.header("How It Works")
        st.markdown("""
        1. Upload audio/video files
        2. AI analyzes for manipulation
        3. View detailed forensic report
        4. Get integrity score (0-100%)
        """)


def analyze_audio_file(audio_file, analyzer):
    """Analyze uploaded audio file"""
    
    with st.spinner("🎵 Analyzing audio..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Run analysis
            results = analyzer.analyze_audio(tmp_path)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Audio Integrity Score",
                    f"{results['integrity_score']:.1f}%",
                    delta=f"{'Authentic' if results['is_authentic'] else 'Suspicious'}"
                )
            
            with col2:
                st.metric(
                    "Fake Probability",
                    f"{results['prediction']['fake_probability']:.1%}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Confidence",
                    f"{results['prediction']['confidence']:.1%}",
                    delta=None
                )
            
            # Display verdict
            if results['is_authentic']:
                st.markdown(f'<div class="success-box">✅ <strong>Verdict:</strong> Audio appears AUTHENTIC</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="danger-box">⚠️ <strong>Verdict:</strong> Audio appears MANIPULATED</div>', 
                           unsafe_allow_html=True)
            
            # Visualizations
            st.subheader("📊 Audio Analysis Visualizations")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Spectrogram", "Waveform", "MFCC Features"])
            
            with tab1:
                # Spectrogram with anomalies
                audio_viz = AudioVisualizer()
                spec_fig = audio_viz.create_spectrogram_with_highlights(
                    results['features']['mel_spectrogram'],
                    results['anomaly_regions'],
                    AUDIO_CONFIG['sample_rate'],
                    AUDIO_CONFIG['hop_length']
                )
                st.pyplot(spec_fig)
                
                if results['anomaly_regions']:
                    st.warning(f"⚠️ {len(results['anomaly_regions'])} suspicious time regions detected")
            
            with tab2:
                # Waveform
                waveform_fig = audio_viz.create_waveform_plot(
                    results['features']['audio'],
                    AUDIO_CONFIG['sample_rate'],
                    results['anomaly_regions']
                )
                st.pyplot(waveform_fig)
            
            with tab3:
                # MFCC heatmap
                mfcc_fig = audio_viz.create_mfcc_heatmap(
                    results['features']['mfcc'][:40]  # Show first 40 coefficients
                )
                st.pyplot(mfcc_fig)
            
            # Analysis summary
            st.subheader("📋 Analysis Summary")
            st.text(results['analysis_summary'])
            
            return results
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def analyze_video_file(video_file, analyzer):
    """Analyze uploaded video file"""
    
    with st.spinner("🎥 Analyzing video... This may take a moment..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_file:
            tmp_file.write(video_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Run analysis
            results = analyzer.analyze_video(tmp_path)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Video Integrity Score",
                    f"{results['integrity_score']:.1f}%",
                    delta=f"{'Authentic' if results['is_authentic'] else 'Suspicious'}"
                )
            
            with col2:
                st.metric(
                    "Fake Probability",
                    f"{results['prediction']['fake_probability']:.1%}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Faces Detected",
                    f"{results['video_data']['total_faces_detected']}",
                    delta=None
                )
            
            # Display verdict
            if results['is_authentic']:
                st.markdown(f'<div class="success-box">✅ <strong>Verdict:</strong> Video appears AUTHENTIC</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="danger-box">⚠️ <strong>Verdict:</strong> Video appears MANIPULATED</div>', 
                           unsafe_allow_html=True)
            
            # Video metadata
            st.subheader("📹 Video Information")
            metadata = results['video_data']['metadata']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**Duration:** {metadata['duration']:.1f}s")
            with col2:
                st.write(f"**FPS:** {metadata['fps']:.1f}")
            with col3:
                st.write(f"**Resolution:** {metadata['width']}x{metadata['height']}")
            with col4:
                st.write(f"**Frames:** {metadata['total_frames']}")
            
            # Sample frames with detection boxes
            if results['manipulated_frames']:
                st.warning(f"⚠️ {len(results['manipulated_frames'])} potentially manipulated frames detected")
            
            # Display sample frames
            st.subheader("🖼️ Sample Frame Analysis")
            frames = results['video_data']['frames']
            
            if len(frames) > 0:
                # Show first few frames
                num_display = min(3, len(frames))
                cols = st.columns(num_display)
                
                for i, col in enumerate(cols):
                    with col:
                        frame_idx = i * (len(frames) // num_display)
                        st.image(frames[frame_idx], caption=f"Frame {frame_idx}", use_column_width=True)
            
            # Analysis summary
            st.subheader("📋 Analysis Summary")
            st.text(results['analysis_summary'])
            
            return results
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def main():
    """Main application"""
    
    # Display header and sidebar
    display_header()
    display_sidebar()
    
    # Initialize analyzer
    try:
        analyzer = load_analyzer()
    except Exception as e:
        st.error(f"Error loading analyzer: {str(e)}")
        st.info("The models will run in demonstration mode without pre-trained weights.")
        analyzer = load_analyzer()
    
    # Main content
    st.markdown("---")
    
    # Analysis mode selection
    analysis_mode = st.radio(
        "Select Analysis Mode:",
        ["Audio Only", "Video Only", "Audio + Video (Integrated)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # File upload section
    if analysis_mode == "Audio Only":
        st.subheader("🎵 Upload Audio File")
        audio_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=['wav', 'mp3'],
            help="Upload an audio file for deepfake detection"
        )
        
        if audio_file:
            st.success(f"✅ File uploaded: {audio_file.name}")
            
            if st.button("🔍 Analyze Audio", type="primary"):
                results = analyze_audio_file(audio_file, analyzer)
                
                # Integrity score gauge
                st.markdown("---")
                st.subheader("🎯 Integrity Score Visualization")
                score_viz = IntegrityScoreVisualizer()
                gauge_fig = score_viz.create_score_gauge(
                    results['integrity_score'],
                    "Audio Integrity Score"
                )
                st.pyplot(gauge_fig)
    
    elif analysis_mode == "Video Only":
        st.subheader("🎥 Upload Video File")
        video_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video file for deepfake detection"
        )
        
        if video_file:
            st.success(f"✅ File uploaded: {video_file.name}")
            
            if st.button("🔍 Analyze Video", type="primary"):
                results = analyze_video_file(video_file, analyzer)
                
                # Integrity score gauge
                st.markdown("---")
                st.subheader("🎯 Integrity Score Visualization")
                score_viz = IntegrityScoreVisualizer()
                gauge_fig = score_viz.create_score_gauge(
                    results['integrity_score'],
                    "Video Integrity Score"
                )
                st.pyplot(gauge_fig)
    
    else:  # Audio + Video
        st.subheader("🎬 Upload Files for Integrated Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Video File**")
            video_file = st.file_uploader(
                "Upload video",
                type=['mp4', 'avi', 'mov'],
                key='video'
            )
        
        with col2:
            st.write("**Audio File (Optional)**")
            audio_file = st.file_uploader(
                "Upload audio or leave empty to extract from video",
                type=['wav', 'mp3'],
                key='audio'
            )
        
        if video_file:
            st.success(f"✅ Video uploaded: {video_file.name}")
            if audio_file:
                st.success(f"✅ Audio uploaded: {audio_file.name}")
            
            if st.button("🔍 Analyze Both", type="primary"):
                # Analyze video
                st.markdown("### Video Analysis")
                video_results = analyze_video_file(video_file, analyzer)
                
                st.markdown("---")
                
                # Analyze audio if provided
                if audio_file:
                    st.markdown("### Audio Analysis")
                    audio_results = analyze_audio_file(audio_file, analyzer)
                else:
                    st.info("ℹ️ Audio analysis from video extraction not fully implemented in demo mode")
                    audio_results = {
                        'integrity_score': 50.0,
                        'is_authentic': True
                    }
                
                st.markdown("---")
                
                # Combined analysis
                st.header("🎯 Combined Integrity Analysis")
                
                # Calculate combined score
                audio_score = audio_results['integrity_score']
                video_score = video_results['integrity_score']
                combined_score = 0.4 * audio_score + 0.6 * video_score
                
                # Display combined metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Audio Score", f"{audio_score:.1f}%")
                with col2:
                    st.metric("Video Score", f"{video_score:.1f}%")
                with col3:
                    st.metric("Combined Score", f"{combined_score:.1f}%", 
                             delta="Final Verdict")
                
                # Final verdict
                is_authentic = combined_score >= 50.0
                if is_authentic:
                    st.markdown(f'<div class="success-box">✅ <strong>FINAL VERDICT:</strong> Content appears AUTHENTIC</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="danger-box">⚠️ <strong>FINAL VERDICT:</strong> Content appears MANIPULATED</div>', 
                               unsafe_allow_html=True)
                
                # Score breakdown
                st.subheader("📊 Score Breakdown")
                score_viz = IntegrityScoreVisualizer()
                breakdown_fig = score_viz.create_breakdown_chart(
                    audio_score, video_score, combined_score
                )
                st.pyplot(breakdown_fig)
                
                # Combined gauge
                gauge_fig = score_viz.create_score_gauge(
                    combined_score,
                    "Combined Integrity Score"
                )
                st.pyplot(gauge_fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
            <p><strong>Multimedia Forensics Suite</strong> - AI-Powered Truth Engine</p>
            <p>Built with PyTorch, OpenCV, Librosa, and Streamlit</p>
            <p style='font-size: 0.8rem;'>⚠️ This tool provides analysis assistance. Always verify important findings through multiple sources.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
