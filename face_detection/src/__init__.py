"""
Face Detection module for Face Recognition pipeline.

Main classes:
    - FaceProcessor: Main class for face detection, landmarks, and alignment
    - FaceDetector: Low-level face detection using MTCNN
    - LandmarkPredictor: Facial landmark prediction

Quick start:
    >>> from face_detection import FaceProcessor
    >>> 
    >>> # From config file
    >>> processor = FaceProcessor.from_config('config.json')
    >>> 
    >>> # Or with explicit path
    >>> processor = FaceProcessor(checkpoint_path='model.pth')
    >>> 
    >>> # Process image
    >>> result = processor.process(image)
    >>> aligned_faces = result.get_aligned_faces()
"""

# Main processor class
from .processor import FaceProcessor, FaceResult, ProcessingResult

# Configuration
from .config import Config, load_config, get_config, set_config, reset_config

# Low-level components
from .detector import FaceDetector, get_detector, detect_faces_from_image
from .landmark_model import (
    LandmarkPredictor, 
    StackedHourglassNetwork,
    HourglassModule,
    ResidualBlock,
    get_landmark_predictor
)

# Web UI
from .app import create_demo, main as run_webui


__all__ = [
    # Main API
    'FaceProcessor',
    'FaceResult', 
    'ProcessingResult',
    
    # Configuration
    'Config',
    'load_config',
    'get_config',
    'set_config',
    'reset_config',
    
    # Detection
    'FaceDetector',
    'get_detector',
    'detect_faces_from_image',
    
    # Landmarks
    'LandmarkPredictor',
    'StackedHourglassNetwork',
    'HourglassModule', 
    'ResidualBlock',
    'get_landmark_predictor',
    
    # Web UI
    'create_demo',
    'run_webui',
]
