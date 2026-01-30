"""
Face Detection Module using MTCNN
Detects faces in images and crops them to 128x128 pixels
"""

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import torch


class FaceDetector:
    """Face detector using MTCNN from facenet-pytorch"""
    
    def __init__(self, output_size: int = 128, margin: int = 20, device: Optional[str] = None):
        """
        Initialize the face detector.
        
        Args:
            output_size: Size of the output cropped face images (default: 128x128)
            margin: Margin to add around detected faces before cropping (default: 20)
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.output_size = output_size
        self.margin = margin
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize MTCNN detector
        # keep_all=True to detect all faces, not just the largest one
        self.mtcnn = MTCNN(
            image_size=output_size,
            margin=margin,
            keep_all=True,
            device=self.device,
            post_process=False  # Return PIL images, not normalized tensors
        )
        
        print(f"Face detector initialized on device: {self.device}")
    
    def detect_faces(self, image: Image.Image) -> Tuple[List[Image.Image], List[List[float]], List[float]]:
        """
        Detect faces in an image and return cropped face images.
        
        Args:
            image: PIL Image to detect faces in
            
        Returns:
            tuple containing:
                - List of cropped face images (128x128 PIL Images)
                - List of bounding boxes [x1, y1, x2, y2]
                - List of detection confidences
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Detect faces - get boxes and probabilities
        boxes, probs = self.mtcnn.detect(image)
        
        if boxes is None:
            return [], [], []
        
        cropped_faces = []
        valid_boxes = []
        valid_probs = []
        
        for box, prob in zip(boxes, probs):
            if prob is None or prob < 0.9:  # Skip low confidence detections
                continue
            
            # Extract and validate bounding box
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Add margin
            x1 = max(0, x1 - self.margin)
            y1 = max(0, y1 - self.margin)
            x2 = min(image.width, x2 + self.margin)
            y2 = min(image.height, y2 + self.margin)
            
            # Crop face from image
            face_crop = image.crop((x1, y1, x2, y2))
            
            # Resize to output_size x output_size
            face_crop = face_crop.resize((self.output_size, self.output_size), Image.LANCZOS)
            
            cropped_faces.append(face_crop)
            valid_boxes.append([x1, y1, x2, y2])
            valid_probs.append(float(prob))
        
        return cropped_faces, valid_boxes, valid_probs
    
    def detect_and_draw(self, image: Image.Image) -> Tuple[Image.Image, List[Image.Image]]:
        """
        Detect faces and return image with drawn bounding boxes + cropped faces.
        
        Args:
            image: PIL Image to process
            
        Returns:
            tuple containing:
                - Original image with bounding boxes drawn
                - List of cropped face images (128x128)
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Detect faces
        cropped_faces, boxes, probs = self.detect_faces(image)
        
        # Draw bounding boxes on a copy of the image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            x1, y1, x2, y2 = box
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
            
            # Draw label with confidence
            label = f"Face {i+1}: {prob:.2f}"
            
            # Draw label background
            text_bbox = draw.textbbox((x1, y1 - 20), label)
            draw.rectangle(text_bbox, fill='lime')
            draw.text((x1, y1 - 20), label, fill='black')
        
        return annotated_image, cropped_faces


# Global detector instance (lazy initialization)
_detector: Optional[FaceDetector] = None


def get_detector() -> FaceDetector:
    """Get or create the global face detector instance."""
    global _detector
    if _detector is None:
        _detector = FaceDetector(output_size=128, margin=20)
    return _detector


def detect_faces_from_image(image: Image.Image) -> Tuple[Image.Image, List[Image.Image]]:
    """
    Convenience function to detect faces from an image.
    
    Args:
        image: PIL Image
        
    Returns:
        tuple of (annotated_image, list_of_cropped_faces)
    """
    detector = get_detector()
    return detector.detect_and_draw(image)
