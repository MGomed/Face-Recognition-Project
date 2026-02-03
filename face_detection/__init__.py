"""
Face Detection Package

Provides face detection, landmark prediction, and face alignment.

Usage:
    from face_detection import FaceProcessor
    
    # From config
    processor = FaceProcessor.from_config('config.json')
    
    # Or explicit
    processor = FaceProcessor(checkpoint_path='model.pth')
    
    # Process
    result = processor.process(image)
    aligned_faces = result.get_aligned_faces()
"""

from .src import (
    # Main API
    FaceProcessor,
    FaceResult,
    ProcessingResult,
    
    # Configuration
    Config,
    load_config,
    get_config,
    set_config,
    reset_config,
    
    # Detection
    FaceDetector,
    get_detector,
    detect_faces_from_image,
    
    # Landmarks  
    LandmarkPredictor,
    StackedHourglassNetwork,
    HourglassModule,
    ResidualBlock,
    get_landmark_predictor,
    
    # Web UI
    create_demo,
    run_webui,
)

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

__version__ = '1.0.0'
