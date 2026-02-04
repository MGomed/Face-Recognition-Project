from typing import List, Optional, Tuple, Union, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from .detector import FaceDetector
from .landmark_model import LandmarkPredictor
from .recognizer import FaceRecognitionModel
from .database import FaceDatabase, SearchResult
from .config import load_config

if TYPE_CHECKING:
    from .config import Config


@dataclass
class FaceResult:
    """Результат для одного обнаруженного лица."""
    
    face: Image.Image

    bbox: List[float]

    confidence: float

    landmarks: Optional[List[Tuple[int, int]]] = None

    aligned_face: Optional[Image.Image] = None

    aligned_landmarks: Optional[List[Tuple[int, int]]] = None

    embedding: Optional[np.ndarray] = None

    best_match: Optional[SearchResult] = None

    matches: List[SearchResult] = field(default_factory=list)

    index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'landmarks': self.landmarks,
            'aligned_landmarks': self.aligned_landmarks,
            'has_embedding': self.embedding is not None,
            'best_match': {
                'path': self.best_match.path,
                'similarity': self.best_match.similarity
            } if self.best_match else None,
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
    _transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def __init__(
        self,
        landmark_checkpoint_path: Optional[str] = None,
        recognition_checkpoint_path: Optional[str] = None,
        cache_path: Optional[str] = None,
        images_directory: Optional[str] = None,
        output_size: int = 128,
        margin: int = 20,
        device: Optional[str] = None,
        confidence_threshold: float = 0.9,
        auto_save: bool = True,
        auto_load: bool = True
    ):
        """        
        Args:
            landmark_checkpoint_path: Путь к весам модели для предсказания ключевых точек
            recognition_checkpoint_path: Путь к весам модели для распознавания лиц
            cache_path: Путь к файлу кеша эмбеддингов (.pkl)
            images_directory: Каталог с выровненными фото лицами
            output_size: Размер выходного лица (по умолчанию: 128)
            margin: Отступ вокруг обнаруженных лиц (по умолчанию: 20)
            device: Устройство ('cuda' или 'cpu')
            confidence_threshold: Минимальное сходство для обнаружения (по умолчанию: 0.9)
            auto_save: Автоматическое сохранение базы данных после изменений
            auto_load: Автоматическая загрузка базы данных при запуске
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
        self._has_landmarks = False
        
        if landmark_checkpoint_path and os.path.exists(landmark_checkpoint_path):
            self._landmark_predictor = LandmarkPredictor(
                checkpoint_path=landmark_checkpoint_path,
                device=self.device
            )
            self._has_landmarks = True
        elif landmark_checkpoint_path:
            print(f"Warning: Landmark model not found: {landmark_checkpoint_path}")

        self._recognition_model: Optional[FaceRecognitionModel] = None
        self._has_recognition = False
        
        if recognition_checkpoint_path and os.path.exists(recognition_checkpoint_path):
            self._load_recognition_model(recognition_checkpoint_path)
            self._has_recognition = True
        elif recognition_checkpoint_path:
            print(f"Warning: Recognition model not found: {recognition_checkpoint_path}")

        self._database: Optional[FaceDatabase] = None
        
        if cache_path or images_directory:
            self._database = FaceDatabase(
                cache_path=cache_path,
                images_directory=images_directory,
                auto_save=auto_save,
                auto_load=auto_load
            )

            if self._has_recognition:
                self._database.set_embedding_function(self._compute_embedding_from_path)

            if auto_load and self._has_recognition and self._database.needs_rebuild():
                print("Building embeddings database from images directory...")
                self._database.build_from_directory()

        print(f"FaceProcessor initialized on {self.device}")
        print(f"  - Face detection: enabled")
        print(f"  - Landmarks: {'enabled' if self._has_landmarks else 'disabled'}")
        print(f"  - Recognition: {'enabled' if self._has_recognition else 'disabled'}")
        print(f"  - Database: {self._database.size if self._database else 0} faces")
    
    def _compute_embedding_from_path(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')

        return self.get_embedding(image)
    
    def _load_recognition_model(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))

        num_classes = 1000
        for key in state_dict:
            if 'loss_fn.weight' in key:
                num_classes = state_dict[key].shape[0]
                break
            elif 'classifier.weight' in key:
                num_classes = state_dict[key].shape[0]
                break

        self._recognition_model = FaceRecognitionModel(
            num_classes=num_classes,
            embedding_size=512,
            loss_type='arcface'
        )

        self._recognition_model.load_state_dict(state_dict)
        self._recognition_model.to(self.device)
        self._recognition_model.eval()
        
        print(f"Loaded recognition model from {checkpoint_path}")
    
    @classmethod
    def from_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        override: Optional[Dict[str, Any]] = None
    ) -> 'FaceProcessor':
        config = load_config(config_path, override)
        
        return cls(
            landmark_checkpoint_path=config.get_landmark_model_path(),
            recognition_checkpoint_path=config.get_face_recognition_model_path(),
            cache_path=config.get_embeddings_cache_path(),
            images_directory=config.get_images_directory(),
            device=config.device,
            auto_save=config.auto_save,
            auto_load=config.auto_load
        )
    
    @property
    def has_landmarks(self) -> bool:
        return self._has_landmarks
    
    @property
    def has_recognition(self) -> bool:
        return self._has_recognition
    
    @property
    def database(self) -> Optional[FaceDatabase]:
        return self._database
    
    @property
    def database_size(self) -> int:
        return self._database.size if self._database else 0
    
    def detect_faces(
        self, 
        image: Union[Image.Image, np.ndarray]
    ) -> Tuple[List[Image.Image], List[List[float]], List[float]]:
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
        result = self.process(image, align_faces=True)

        return result.get_aligned_faces()
    
    def get_embedding(
        self,
        face_image: Union[Image.Image, np.ndarray]
    ) -> np.ndarray:
        """        
        Args:
            face_image: Фото лица (должно быть выровнено 128x128)
            
        Returns:
            Вектор эмбеддинга (нормализованный)
            
        Raises:
            RuntimeError: Если модель для распознавания не инициализирована
        """
        if not self._has_recognition:
            raise RuntimeError("Recognition model not available")
        
        pil_image = self._to_pil(face_image)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Transform and get embedding
        tensor = self._transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding, _ = self._recognition_model(tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy().flatten()
    
    def get_embeddings_batch(
        self,
        face_images: List[Union[Image.Image, np.ndarray]]
    ) -> np.ndarray:
        """
        Args:
            face_images: Список фото лиц
            
        Returns:
            Массив эмбеддингов [N, embedding_dim]
        """
        if not self._has_recognition:
            raise RuntimeError("Recognition model not available")
        
        if not face_images:
            return np.array([])
        
        # Transform all images
        tensors = []
        for img in face_images:
            pil_img = self._to_pil(img)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            tensors.append(self._transform(pil_img))
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            embeddings, _ = self._recognition_model(batch)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def add_face_to_database(
        self,
        image_path: str,
        face_image: Optional[Union[Image.Image, np.ndarray]] = None,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Args:
            image_path: Путь для ассоциирования с фото лицом
            face_image: Фото лица (будет извлечен эмбеддинг, если не предоставлен embedding)
            embedding: Предвычисленный эмбеддинг (опционально)
            
        Returns:
            True если добавлено успешно
        """
        if self._database is None:
            raise RuntimeError("База данных не инициализирована. Укажите embeddings_path в конфиге.")
        
        if embedding is None:
            if face_image is None:
                raise ValueError("Необходимо предоставить либо face_image, либо embedding")
            embedding = self.get_embedding(face_image)
        
        return self._database.add_face(image_path, embedding)
    
    def add_image_to_database(
        self,
        image_path: str,
        image: Optional[Union[Image.Image, np.ndarray]] = None
    ) -> int:
        """
        Args:
            image_path: Путь к изображению (используется как основа для путей фото лиц)
            image: Изображение для обработки (если None, будет загружено из image_path)
            
        Returns:
            Количество добавленных лиц
        """
        if self._database is None:
            raise RuntimeError("База данных не инициализирована")
        
        if image is None:
            image = Image.open(image_path)

        result = self.process(image, align_faces=True, compute_embeddings=True)

        added = 0
        base_path = Path(image_path)
        
        for face_result in result.faces:
            if face_result.embedding is not None:
                if result.num_faces == 1:
                    face_path = str(base_path)
                else:
                    face_path = str(base_path.with_stem(f"{base_path.stem}_face{face_result.index}"))
                
                if self._database.add_face(face_path, face_result.embedding):
                    added += 1
        
        return added
    
    def find_similar_faces(
        self,
        query: Union[Image.Image, np.ndarray, str],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """        
        Args:
            query: Фото лица, эмбеддинг или путь к изображению
            top_k: Количество результатов для возврата
            threshold: Минимальное сходство для рассмотрения
            
        Returns:
            Список SearchResult, отсортированный по сходству
        """
        if self._database is None:
            raise RuntimeError("База данных не инициализирована")

        if isinstance(query, str):
            image = Image.open(query)
            faces = self.get_embeddings_ready_faces(image)
            if not faces:
                return []
            embedding = self.get_embedding(faces[0])
        elif isinstance(query, np.ndarray) and query.ndim == 1:
            embedding = query
        else:
            embedding = self.get_embedding(query)
        
        return self._database.find_similar(embedding, top_k=top_k, threshold=threshold)
    
    def identify_face(
        self,
        face_image: Union[Image.Image, np.ndarray],
        threshold: float = 0.5
    ) -> Optional[SearchResult]:
        """
        Args:
            face_image: Фото лица для идентификации
            threshold: Минимальное сходство для рассмотрения
            
        Returns:
            Лучшее совпадение или None, если нет совпадения выше порога
        """
        if self._database is None:
            raise RuntimeError("База данных не инициализирована")
        
        embedding = self.get_embedding(face_image)
        return self._database.find_best_match(embedding, threshold=threshold)
    
    def save_database(self, path: Optional[str] = None):
        """Save face database to cache file."""
        if self._database is None:
            raise RuntimeError("Database not initialized")
        self._database.save_cache(path)
    
    def load_database(self, path: Optional[str] = None):
        """Load face database from cache file."""
        if self._database is None:
            raise RuntimeError("Database not initialized")
        self._database.load_cache(path)
    
    def rebuild_database(self, progress_callback=None) -> int:
        """
        Args:
            progress_callback: callback для отслеживания прогресса
            
        Returns:
            Количество обработанных изображений
        """
        if self._database is None:
            raise RuntimeError("База данных не инициализирована")
        
        if not self._has_recognition:
            raise RuntimeError("Модель распознавания лиц необходима для построения базы данных")
        
        return self._database.build_from_directory(
            embedding_fn=self._compute_embedding_from_path,
            progress_callback=progress_callback
        )
    
    def database_needs_rebuild(self) -> bool:
        if self._database is None:
            return False

        return self._database.needs_rebuild()
    
    def process(
        self,
        image: Union[Image.Image, np.ndarray],
        align_faces: bool = True,
        compute_embeddings: bool = False,
        find_matches: bool = False,
        match_threshold: float = 0.5,
        top_k: int = 5
    ) -> ProcessingResult:
        """
        Args:
            image: Входное изображение
            align_faces: Нужно ли выравнивать лица
            compute_embeddings: Нужно ли вычислять эмбеддинги
            find_matches: Нужно ли найти совпадения в базе данных
            match_threshold: Порог для совпадения
            top_k: Количество совпадений для найденных
            
        Returns:
            ProcessingResult с всей информацией
        """
        pil_image = self._to_pil(image)

        annotated_image, _ = self._detector.detect_and_draw(pil_image)
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
                    print(f"Error processing landmarks for face {i}: {e}")

            if compute_embeddings and self._has_recognition:
                try:
                    face_for_embedding = result.aligned_face if result.aligned_face else face
                    result.embedding = self.get_embedding(face_for_embedding)
                except Exception as e:
                    print(f"Error computing embedding for face {i}: {e}")

            if find_matches and result.embedding is not None and self._database:
                try:
                    result.matches = self._database.find_similar(
                        result.embedding,
                        top_k=top_k,
                        threshold=match_threshold
                    )
                    if result.matches:
                        result.best_match = result.matches[0]
                except Exception as e:
                    print(f"Error finding matches for face {i}: {e}")
            
            face_results.append(result)

        num_faces = len(face_results)
        status_parts = []
        
        if num_faces == 0:
            status_parts.append("No faces detected.")
        else:
            status_parts.append(f"{num_faces} face(s) detected.")
        
        if self._has_landmarks:
            status_parts.append("Landmarks predicted.")
        if compute_embeddings and self._has_recognition:
            status_parts.append("Embeddings computed.")
        if find_matches and self._database:
            matched = sum(1 for f in face_results if f.best_match)
            status_parts.append(f"{matched} face(s) matched.")
        
        return ProcessingResult(
            annotated_image=annotated_image,
            faces=face_results,
            num_faces=num_faces,
            status=" ".join(status_parts)
        )
    
    def _to_pil(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)

        return image
    
    def __repr__(self) -> str:
        return (
            f"FaceProcessor("
            f"device={self.device}, "
            f"landmarks={self._has_landmarks}, "
            f"recognition={self._has_recognition}, "
            f"database={self.database_size} faces)"
        )
