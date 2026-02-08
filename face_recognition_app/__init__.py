"""
Face Detection and Recognition Package

Provides:
    - Face detection
    - Landmark prediction
    - Face alignment
    - Face recognition (embedding extraction)
    - Face database (similarity search)

Usage:
    from face_detection import FaceProcessor
    
    # From config
    processor = FaceProcessor.from_config('config.json')
    
    # Add faces to database
    processor.add_image_to_database('photo.jpg')
    
    # Find similar faces
    matches = processor.find_similar_faces(query_image)
"""

from .src import (
    # Main API
    FaceProcessor,
    FaceResult,
    ProcessingResult,
    
    # Database
    FaceDatabase,
    SearchResult,
    
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
    
    # Recognition
    FaceRecognitionModel,
    ArcFaceLoss,
    
    # Web UI
    create_demo,
    run_webui,
)

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

__version__ = '1.0.0'
