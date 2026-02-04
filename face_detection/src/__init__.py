"""
Face Detection and Recognition module.

Main classes:
    - FaceProcessor: Main class for detection, recognition, and search
    - FaceDatabase: Storage and search for face embeddings
    - FaceDetector: Low-level face detection using MTCNN
    - LandmarkPredictor: Facial landmark prediction
    - FaceRecognitionModel: Face embedding extraction

Quick start:
    >>> from face_detection import FaceProcessor
    >>> 
    >>> # From config file
    >>> processor = FaceProcessor.from_config('config.json')
    >>> 
    >>> # Process image
    >>> result = processor.process(image, compute_embeddings=True, find_matches=True)
    >>> 
    >>> # Get matches
    >>> for face in result.faces:
    >>>     if face.best_match:
    >>>         print(f"Match: {face.best_match.path} ({face.best_match.similarity:.2f})")
"""

# Main processor class
from .processor import FaceProcessor, FaceResult, ProcessingResult

# Database
from .database import FaceDatabase, SearchResult

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
from .recognizer import FaceRecognitionModel, ArcFaceLoss

# Web UI
from .app import create_demo, main as run_webui


__all__ = [
    # Main API
    'FaceProcessor',
    'FaceResult', 
    'ProcessingResult',
    
    # Database
    'FaceDatabase',
    'SearchResult',
    
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
    
    # Recognition
    'FaceRecognitionModel',
    'ArcFaceLoss',
    
    # Web UI
    'create_demo',
    'run_webui',
]
