# 🔍 Multimedia Forensics Suite

A comprehensive AI-powered platform for detecting deepfakes and analyzing multimedia authenticity. This "Truth Engine" provides forensic analysis of both audio and video content with explainable visualizations.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

### Audio Forensics Pipeline
- **MFCC & Spectrogram Analysis**: Extract Mel-frequency cepstral coefficients and spectrograms
- **CNN-Based Detector**: ResNet-inspired architecture for synthetic voice detection
- **Temporal Anomaly Detection**: Identify suspicious time regions in audio
- **High-Frequency Analysis**: Detect artifacts in frequency domain
- **Silence Pattern Analysis**: Identify unnatural silence patterns

### Video Forensics Pipeline
- **Face Detection**: MTCNN-based face detection and extraction
- **Deepfake Detection**: EfficientNet-inspired model for face-swap detection
- **Frame-by-Frame Analysis**: Comprehensive video analysis
- **Blending Artifact Detection**: Identify face manipulation inconsistencies
- **Blink Pattern Analysis**: Detect unnatural blinking patterns

### Explainability & Visualization
- **Video Bounding Boxes**: Real-time overlay showing fake/real classification
- **Spectrogram Highlighting**: Visual identification of manipulated audio segments
- **Integrity Score**: 0-100% confidence score combining all analysis factors
- **Detailed Reports**: Human-readable forensic analysis summaries

### Web Interface
- **Drag-and-Drop Upload**: User-friendly Streamlit interface
- **Multi-Modal Analysis**: Analyze audio, video, or both together
- **Interactive Visualizations**: Real-time charts and graphs
- **Comprehensive Reports**: Detailed breakdown of analysis results

## 🏗️ Architecture

```
Multimedia-Forensics-Suite/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── config/
│   └── config.py              # Configuration settings
├── src/
│   ├── audio/
│   │   ├── preprocessing.py   # Audio feature extraction
│   │   └── detector.py        # Audio deepfake detector
│   ├── video/
│   │   ├── preprocessing.py   # Video frame processing
│   │   └── detector.py        # Video deepfake detector
│   ├── models/                # Model architectures
│   └── utils/
│       ├── analyzer.py        # Integrated analyzer
│       └── visualization.py   # Visualization utilities
└── README.md
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yadavanujkumar/Multimedia-Forensics-Suite.git
cd Multimedia-Forensics-Suite
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser**
Navigate to `http://localhost:8501`

## 📖 Usage

### Audio Analysis
1. Select "Audio Only" mode
2. Upload a WAV or MP3 file
3. Click "Analyze Audio"
4. View results including:
   - Integrity score
   - Spectrogram with anomaly highlights
   - MFCC features
   - Detailed analysis report

### Video Analysis
1. Select "Video Only" mode
2. Upload an MP4, AVI, or MOV file
3. Click "Analyze Video"
4. View results including:
   - Face detection results
   - Frame-by-frame analysis
   - Manipulation probability
   - Sample frame visualizations

### Integrated Analysis
1. Select "Audio + Video" mode
2. Upload both video and audio files
3. Click "Analyze Both"
4. View comprehensive results:
   - Combined integrity score
   - Component breakdown
   - Weighted final verdict

## 🧠 Technical Details

### Audio Processing
- **MFCC Extraction**: 40 coefficients with delta and delta-delta features
- **Spectrogram**: Mel-scaled with 128 bands
- **Model**: ResNet-based CNN with residual blocks
- **Input**: 16kHz audio, normalized and preprocessed

### Video Processing
- **Face Detection**: MTCNN with 0.9 confidence threshold
- **Frame Sampling**: Every 5th frame for efficiency
- **Model**: EfficientNet-inspired architecture with attention
- **Input**: 224x224 RGB images, ImageNet normalized

### Scoring System
- Audio weight: 40%
- Video weight: 60%
- Combined integrity score: Weighted average
- Threshold: 50% for authentic/manipulated classification

## 🔬 Model Training

The platform is designed to work with models trained on:
- **Audio**: ASVspoof dataset for synthetic voice detection
- **Video**: FaceForensics++ or Celeb-DF datasets for deepfake detection

Note: This implementation includes the full architecture but ships without pre-trained weights. For production use, train models on appropriate datasets or use your own pre-trained weights.

## 📊 Example Output

### Integrity Score Visualization
The system provides a gauge-style visualization showing:
- Overall integrity score (0-100%)
- Color-coded indicator (green=real, red=fake)
- Confidence level

### Audio Spectrogram
- Time-frequency representation
- Highlighted suspicious regions in red
- dB scale intensity mapping

### Video Frame Analysis
- Bounding boxes around detected faces
- Color coding: Green (real), Red (fake)
- Confidence scores per detection

## 🛠️ Configuration

Edit `config/config.py` to customize:
- Audio processing parameters
- Video analysis settings
- Model paths
- Detection thresholds
- Visualization preferences

## ⚠️ Limitations

- Models run in demonstration mode without pre-trained weights
- Audio extraction from video requires additional tools (ffmpeg)
- Processing time depends on file size and hardware
- Best results require GPU acceleration (CUDA-enabled PyTorch)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Dataset integration
- Model improvements

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ASVspoof Dataset** for audio deepfake detection research
- **FaceForensics++** for video manipulation detection
- **MTCNN** for robust face detection
- **Librosa** for audio processing
- **OpenCV** for computer vision operations
- **Streamlit** for the web interface

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🔮 Future Enhancements

- [ ] Pre-trained model weights
- [ ] Real-time webcam analysis
- [ ] Batch processing support
- [ ] API endpoint for integration
- [ ] Multi-language support
- [ ] Advanced temporal analysis
- [ ] GAN-specific detection methods
- [ ] Blockchain-based verification

---

**⚠️ Disclaimer**: This tool provides analysis assistance for multimedia forensics. Always verify important findings through multiple sources and methods. The tool should be used as part of a comprehensive verification process.