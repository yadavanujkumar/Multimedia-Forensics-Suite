"""
Demo script to test the Multimedia Forensics Suite components
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from audio.preprocessing import AudioPreprocessor
        print("✓ Audio preprocessing module")
    except Exception as e:
        print(f"✗ Audio preprocessing module: {e}")
        return False
    
    try:
        from audio.detector import AudioAnalyzer, AudioDeepfakeDetector
        print("✓ Audio detector module")
    except Exception as e:
        print(f"✗ Audio detector module: {e}")
        return False
    
    try:
        from video.preprocessing import VideoPreprocessor
        print("✓ Video preprocessing module")
    except Exception as e:
        print(f"✗ Video preprocessing module: {e}")
        return False
    
    try:
        from video.detector import VideoAnalyzer, VideoDeepfakeDetector
        print("✓ Video detector module")
    except Exception as e:
        print(f"✗ Video detector module: {e}")
        return False
    
    try:
        from utils.analyzer import MultimediaForensicsAnalyzer
        print("✓ Integrated analyzer module")
    except Exception as e:
        print(f"✗ Integrated analyzer module: {e}")
        return False
    
    try:
        from utils.visualization import (
            VideoVisualizer, AudioVisualizer, IntegrityScoreVisualizer
        )
        print("✓ Visualization module")
    except Exception as e:
        print(f"✗ Visualization module: {e}")
        return False
    
    try:
        from config.config import AUDIO_CONFIG, VIDEO_CONFIG
        print("✓ Configuration module")
    except Exception as e:
        print(f"✗ Configuration module: {e}")
        return False
    
    return True


def test_audio_components():
    """Test audio processing components"""
    print("\nTesting audio components...")
    
    try:
        from audio.preprocessing import AudioPreprocessor
        from audio.detector import AudioDeepfakeDetector
        
        # Initialize components
        preprocessor = AudioPreprocessor()
        model = AudioDeepfakeDetector()
        
        print("✓ Audio components initialized successfully")
        print(f"  - MFCC coefficients: {preprocessor.n_mfcc}")
        print(f"  - Sample rate: {preprocessor.sample_rate} Hz")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ Audio components test failed: {e}")
        return False


def test_video_components():
    """Test video processing components"""
    print("\nTesting video components...")
    
    try:
        from video.preprocessing import VideoPreprocessor
        from video.detector import VideoDeepfakeDetector
        
        # Initialize components
        preprocessor = VideoPreprocessor()
        model = VideoDeepfakeDetector()
        
        print("✓ Video components initialized successfully")
        print(f"  - Frame sample rate: {preprocessor.frame_sample_rate}")
        print(f"  - Face detection threshold: {preprocessor.face_detection_threshold}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ Video components test failed: {e}")
        return False


def test_integrated_analyzer():
    """Test integrated analyzer"""
    print("\nTesting integrated analyzer...")
    
    try:
        from utils.analyzer import MultimediaForensicsAnalyzer
        
        # Initialize analyzer
        analyzer = MultimediaForensicsAnalyzer(device='cpu')
        
        print("✓ Integrated analyzer initialized successfully")
        print("  - Audio pipeline: Ready")
        print("  - Video pipeline: Ready")
        print("  - Device: CPU")
        
        return True
    except Exception as e:
        print(f"✗ Integrated analyzer test failed: {e}")
        return False


def test_visualizations():
    """Test visualization components"""
    print("\nTesting visualization components...")
    
    try:
        from utils.visualization import (
            VideoVisualizer, AudioVisualizer, IntegrityScoreVisualizer
        )
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Test audio visualizer
        audio_viz = AudioVisualizer()
        
        # Test video visualizer
        video_viz = VideoVisualizer()
        
        # Test score visualizer
        score_viz = IntegrityScoreVisualizer()
        
        print("✓ Visualization components initialized successfully")
        print("  - Audio visualizer: Ready")
        print("  - Video visualizer: Ready")
        print("  - Score visualizer: Ready")
        
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("MULTIMEDIA FORENSICS SUITE - COMPONENT TEST")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("Audio Components", test_audio_components()))
    results.append(("Video Components", test_video_components()))
    results.append(("Integrated Analyzer", test_integrated_analyzer()))
    results.append(("Visualizations", test_visualizations()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nTo start the web application, run:")
        print("  streamlit run app.py")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
