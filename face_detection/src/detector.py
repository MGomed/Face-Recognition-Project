from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import torch

TARGET_SIZE = (128, 128)
MARGIN = 20

class FaceDetector:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.mtcnn = MTCNN(
            image_size=TARGET_SIZE,
            margin=MARGIN,
            keep_all=True,
            device=self.device,
            post_process=False
        )
    
    def detect_faces(self, image: Image.Image) -> Tuple[List[Image.Image], List[List[float]], List[float]]:
        """
        Args:
            image: PIL Image для детекции лиц
            
        Returns:
            tuple содержит:
                - Список кропнутых лиц (128x128 PIL Images)
                - Список bounding boxes [x1, y1, x2, y2]
                - Список confidence детекции
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        boxes, probs = self.mtcnn.detect(image)

        if boxes is None:
            return [], [], []
      
        cropped_faces = []
        valid_boxes = []
        valid_probs = []

        for box, prob in zip(boxes, probs):
            if prob is None or prob < 0.9:
                continue

            x1, y1, x2, y2 = [int(b) for b in box]

            x1 = max(0, x1 - MARGIN)
            y1 = max(0, y1 - MARGIN)
            x2 = min(image.width, x2 + MARGIN)
            y2 = min(image.height, y2 + MARGIN)

            face_crop = image.crop((x1, y1, x2, y2))

            face_crop = face_crop.resize(TARGET_SIZE, Image.LANCZOS)

            cropped_faces.append(face_crop)
            valid_boxes.append([x1, y1, x2, y2])
            valid_probs.append(float(prob))
        
        return cropped_faces, valid_boxes, valid_probs
    
    def detect_and_draw(self, image: Image.Image) -> Tuple[Image.Image, List[Image.Image]]:
        """
        Детектирует лица и рисует bounding boxes на изображении
        
        Args:
            image: PIL Image для обработки
            
        Returns:
            tuple содержит:
                - Оригинальное изображение с нарисованными bounding boxes
                - Список кропнутых лиц (128x128)
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        cropped_faces, boxes, probs = self.detect_faces(image)

        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for i, (box, prob) in enumerate(zip(boxes, probs)):
            x1, y1, x2, y2 = box

            draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)

            label = f"Face {i+1}: {prob:.2f}"

            text_bbox = draw.textbbox((x1, y1 - 20), label)
            draw.rectangle(text_bbox, fill='lime')
            draw.text((x1, y1 - 20), label, fill='black')
        
        return annotated_image, cropped_faces


_detector: Optional[FaceDetector] = None


def get_detector() -> FaceDetector:
    global _detector

    if _detector is None:
        _detector = FaceDetector(output_size=128, margin=20)

    return _detector


def detect_faces_from_image(image: Image.Image) -> Tuple[Image.Image, List[Image.Image]]:
    detector = get_detector()

    return detector.detect_and_draw(image)
