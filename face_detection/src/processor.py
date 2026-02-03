from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import os

import numpy as np
from PIL import Image
import torch

from .detector import FaceDetector
from .landmark_model import LandmarkPredictor


@dataclass
class FaceResult:
    """Результат для одного обнаруженного лица."""
    
    face: Image.Image

    bbox: List[float]

    confidence: float

    landmarks: Optional[List[Tuple[int, int]]] = None

    aligned_face: Optional[Image.Image] = None

    aligned_landmarks: Optional[List[Tuple[int, int]]] = None

    index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'landmarks': self.landmarks,
            'aligned_landmarks': self.aligned_landmarks,
        }


@dataclass
class ProcessingResult:
    """Результат обработки всего изображения."""

    annotated_image: Image.Image

    faces: List[FaceResult]

    num_faces: int

    status: str

    def get_cropped_faces(self) -> List[Image.Image]:
        return [f.face for f in self.faces]
    
    def get_aligned_faces(self) -> List[Image.Image]:
        return [f.aligned_face for f in self.faces if f.aligned_face is not None]
    
    def get_all_landmarks(self) -> List[List[Tuple[int, int]]]:
        return [f.landmarks for f in self.faces if f.landmarks is not None]


class FaceProcessor:
    def __init__(
        self,
        landmark_checkpoint_path: Optional[str] = None,
        output_size: int = 128,
        margin: int = 20,
        device: Optional[str] = None,
        confidence_threshold: float = 0.9
    ):
        """        
        Args:
            landmark_checkpoint_path: Путь к весам модели для предсказания ключевых точек
            output_size: Размер выходного лица (default: 128x128)
            margin: Отступ вокруг обнаруженных лиц (default: 20)
            device: Устройство для использования ('cuda' или 'cpu')
            confidence_threshold: Минимальное confidence для обнаружения лица (default: 0.9)
        """
        self.output_size = output_size
        self.margin = margin
        self.confidence_threshold = confidence_threshold

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self._detector = FaceDetector(
            output_size=output_size,
            margin=margin,
            device=self.device
        )

        self._landmark_predictor: Optional[LandmarkPredictor] = None
        self.checkpoint_path = landmark_checkpoint_path
        
        if landmark_checkpoint_path is not None and os.path.exists(landmark_checkpoint_path):
            self._landmark_predictor = LandmarkPredictor(
                checkpoint_path=landmark_checkpoint_path,
                device=self.device
            )
            self._has_landmarks = True
        else:
            self._has_landmarks = False
            if landmark_checkpoint_path is not None:
                print(f"Warning: Checkpoint not found at {landmark_checkpoint_path}")
        
        print(f"FaceProcessor initialized on {self.device}")
        print(f"  - Face detection: enabled")
        print(f"  - Landmark prediction: {'enabled' if self._has_landmarks else 'disabled'}")
    
    @property
    def has_landmarks(self) -> bool:
        return self._has_landmarks
    
    def detect_faces(
        self, 
        image: Union[Image.Image, np.ndarray]
    ) -> Tuple[List[Image.Image], List[List[float]], List[float]]:
        """
        Args:
            image: PIL Image or numpy array
        Returns:
            tuple: (cropped_faces, bounding_boxes, confidences)
        """
        pil_image = self._to_pil(image)

        return self._detector.detect_faces(pil_image)
    
    def predict_landmarks(
        self, 
        face_image: Union[Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Args:
            face_image: Кропнутые изображение лица (должно быть 128x128)
        Returns:
            tuple: (heatmaps, keypoints)
        Raises:
            RuntimeError: Если модель для предсказания ключевых точек не инициализирована
        """
        if not self._has_landmarks:
            raise RuntimeError(
                "Landmark predictor not available. "
                "Initialize FaceProcessor with a valid checkpoint_path."
            )
        
        pil_image = self._to_pil(face_image)

        return self._landmark_predictor.predict(pil_image)
    
    def align_face(
        self, 
        face_image: Union[Image.Image, np.ndarray],
        landmarks: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[Image.Image, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Args:
            face_image: Кропнутые изображение лица
            landmarks: Опциональные предобработанные координаты ключевых точек. Если None, будут предсказаны.
        Returns:
            tuple: (aligned_face, original_landmarks, aligned_landmarks)
        Raises:
            RuntimeError: Если координаты ключевых точек не предоставлены и модель для предсказания ключевых точек не инициализирована
        """
        pil_image = self._to_pil(face_image)
        
        if landmarks is None:
            if not self._has_landmarks:
                raise RuntimeError(
                    "Cannot align face: no landmarks provided and predictor not available."
                )
            return self._landmark_predictor.align_and_predict(pil_image)

        M = self._landmark_predictor.compute_affine_transform(landmarks)
        aligned = self._landmark_predictor.align_face(pil_image, landmarks)
        aligned_landmarks = self._landmark_predictor.transform_keypoints(landmarks, M)
        
        return aligned, landmarks, aligned_landmarks
    
    def draw_landmarks(
        self,
        image: Union[Image.Image, np.ndarray],
        landmarks: List[Tuple[int, int]]
    ) -> Image.Image:
        """
        Args:
            image: Изображение для рисования
            landmarks: Список координат (x, y)
        Returns:
            Изображение с нарисованными точками
        """
        pil_image = self._to_pil(image)

        return self._landmark_predictor.draw_landmarks(pil_image, landmarks)
    
    def process(
        self, 
        image: Union[Image.Image, np.ndarray],
        align_faces: bool = True,
        draw_landmarks: bool = True
    ) -> ProcessingResult:
        """
        Args:
            image: Входное изображение (PIL Image или numpy array)
            align_faces: Нужно ли выравнивать лица (требуется координаты ключевых точек)
            draw_landmarks: Нужно ли рисовать точки на изображении
        Returns:
            ProcessingResult с всеми обнаруженными лицами и их информацией
        """
        pil_image = self._to_pil(image)

        annotated_image, cropped_faces = self._detector.detect_and_draw(pil_image)
        cropped_faces_list, boxes, probs = self._detector.detect_faces(pil_image)

        face_results: List[FaceResult] = []

        for i, (face, bbox, conf) in enumerate(zip(cropped_faces_list, boxes, probs)):
            result = FaceResult(
                face=face,
                bbox=bbox,
                confidence=conf,
                index=i
            )

            if self._has_landmarks:
                try:
                    aligned, orig_landmarks, aligned_landmarks = \
                        self._landmark_predictor.align_and_predict(face)

                    result.landmarks = orig_landmarks

                    if align_faces:
                        result.aligned_face = aligned
                        result.aligned_landmarks = aligned_landmarks

                except Exception as e:
                    print(f"Error processing face {i}: {e}")

            face_results.append(result)

        num_faces = len(face_results)
        if num_faces == 0:
            status = "No faces detected."
        elif num_faces == 1:
            status = "1 face detected."
        else:
            status = f"{num_faces} faces detected."
        
        if self._has_landmarks:
            status += " Landmarks predicted."
            if align_faces:
                status += " Faces aligned."
        else:
            status += " (Landmarks not available)"
        
        return ProcessingResult(
            annotated_image=annotated_image,
            faces=face_results,
            num_faces=num_faces,
            status=status
        )
    
    def process_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        align_faces: bool = True
    ) -> List[ProcessingResult]:
        """
        Args:
            images: Список изображений
            align_faces: Нужно ли выравнивать лица
        Returns:
            Список ProcessingResult для каждого изображения
        """
        return [self.process(img, align_faces=align_faces) for img in images]
    
    def get_embeddings_ready_faces(
        self,
        image: Union[Image.Image, np.ndarray]
    ) -> List[Image.Image]:
        """
        Args:
            image: Входное изображение
        Returns:
            Список выровненных лиц (128x128)
        """
        result = self.process(image, align_faces=True)

        return result.get_aligned_faces()
    
    def _to_pil(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)

        return image
    
    def __repr__(self) -> str:
        return (
            f"FaceProcessor("
            f"device={self.device}, "
            f"output_size={self.output_size}, "
            f"has_landmarks={self._has_landmarks})"
        )
