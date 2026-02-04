"""
Face Database - Storage and search for face embeddings.

Supports:
    - Loading embeddings from cache file
    - Building embeddings from directory of images
    - Similarity search using cosine similarity
    - Auto-save and auto-rebuild
"""

from typing import Dict, List, Optional, Tuple, Union, Callable, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
from glob import glob
import pickle
import os
import hashlib

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


@dataclass
class SearchResult:
    path: str
    similarity: float
    embedding: np.ndarray
    
    def __repr__(self) -> str:
        return f"Результат поиска похожих лиц(путь='{self.path}', сходство={self.similarity:.4f})"


class FaceDatabase:
    def __init__(
        self,
        cache_path: Optional[str] = None,
        images_directory: Optional[str] = None,
        auto_save: bool = True,
        auto_load: bool = True
    ):
        """
        Args:
            cache_path: Путь к файлу кеша эмбеддингов (.pkl)
            images_directory: Каталог с фото лицами
            auto_save: Автоматическое сохранение после изменений
            auto_load: Автоматическая загрузка при инициализации (из кеша или перестроение из каталога)
        """
        self.cache_path = cache_path
        self.images_directory = images_directory
        self.auto_save = auto_save

        self._embeddings: Dict[str, np.ndarray] = {}
        self._directory_hash: Optional[str] = None

        self._normalized_cache: Optional[np.ndarray] = None
        self._paths_cache: Optional[List[str]] = None

        self._embedding_fn: Optional[Callable] = None
 
        if auto_load:
            self._auto_load()
    
    def set_embedding_function(self, fn: Callable):
        self._embedding_fn = fn
    
    def _auto_load(self):
        cache_loaded = False

        if self.cache_path and os.path.exists(self.cache_path):
            try:
                self.load_cache()
                cache_loaded = True

                if self.images_directory and os.path.isdir(self.images_directory):
                    current_hash = self._compute_directory_hash()
                    if current_hash != self._directory_hash:
                        print("Directory changed since cache was created. Rebuild needed.")
                        cache_loaded = False
            except Exception as e:
                print(f"Error loading cache: {e}")
                cache_loaded = False

        if not cache_loaded and self.images_directory:
            print(f"Cache not available. Will rebuild from {self.images_directory} when ready.")
    
    @property
    def size(self) -> int:
        return len(self._embeddings)
    
    @property
    def paths(self) -> List[str]:
        return list(self._embeddings.keys())
    
    @property
    def is_ready(self) -> bool:
        return self.size > 0
    
    def needs_rebuild(self) -> bool:
        if not self.images_directory or not os.path.isdir(self.images_directory):
            return False
        
        if self.size == 0:
            return True
        
        current_hash = self._compute_directory_hash()

        return current_hash != self._directory_hash
    
    def _compute_directory_hash(self) -> str:
        if not self.images_directory:
            return ""
        
        files_info = []
        for ext in IMAGE_EXTENSIONS:
            for path in glob(os.path.join(self.images_directory, f"*{ext}")):
                stat = os.stat(path)
                files_info.append(f"{path}:{stat.st_size}:{stat.st_mtime}")
        
        files_info.sort()
        content = "\n".join(files_info)

        return hashlib.md5(content.encode()).hexdigest()
    
    def get_image_files(self) -> List[str]:
        if not self.images_directory or not os.path.isdir(self.images_directory):
            return []
        
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(glob(os.path.join(self.images_directory, f"*{ext}")))
            files.extend(glob(os.path.join(self.images_directory, f"*{ext.upper()}")))
        
        return sorted(set(files))
    
    def build_from_directory(
        self,
        embedding_fn: Optional[Callable] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> int:
        """
        Args:
            embedding_fn: Функция для вычисления эмбеддинга из пути к изображению.
            progress_callback: callback для отслеживания прогресса
            
        Returns:
            Количество обработанных изображений
        """
        fn = embedding_fn or self._embedding_fn
        if fn is None:
            raise RuntimeError(
                "No embedding function. Set it with set_embedding_function() "
                "or pass to build_from_directory()"
            )
        
        if not self.images_directory or not os.path.isdir(self.images_directory):
            raise ValueError(f"Images directory not found: {self.images_directory}")
        
        # Get image files
        image_files = self.get_image_files()
        if not image_files:
            print(f"No images found in {self.images_directory}")
            return 0
        
        print(f"Building database from {len(image_files)} images...")

        self.clear()

        processed = 0
        for i, image_path in enumerate(image_files):
            try:
                if progress_callback:
                    progress_callback(i + 1, len(image_files), image_path)

                embedding = fn(image_path)

                self.add_face(image_path, embedding, auto_save=False)
                processed += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        self._directory_hash = self._compute_directory_hash()

        if self.cache_path:
            self.save_cache()
        
        print(f"Built database with {processed} faces")

        return processed
    
    def add_face(
        self,
        image_path: str,
        embedding: Union[np.ndarray, torch.Tensor],
        overwrite: bool = False,
        auto_save: bool = True
    ) -> bool:
        image_path = str(Path(image_path).resolve())
        
        if image_path in self._embeddings and not overwrite:
            return False

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        
        embedding = embedding.flatten().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self._embeddings[image_path] = embedding

        self._invalidate_cache()

        if auto_save and self.auto_save and self.cache_path:
            self.save_cache()
        
        return True
    
    def add_faces_batch(
        self,
        image_paths: List[str],
        embeddings: Union[np.ndarray, torch.Tensor],
        overwrite: bool = False
    ) -> int:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        old_auto_save = self.auto_save
        self.auto_save = False
        
        added = 0
        for path, emb in zip(image_paths, embeddings):
            if self.add_face(path, emb, overwrite=overwrite, auto_save=False):
                added += 1

        self.auto_save = old_auto_save
        if self.auto_save and self.cache_path:
            self.save_cache()
        
        return added
    
    def remove_face(self, image_path: str) -> bool:
        image_path = str(Path(image_path).resolve())
        
        if image_path in self._embeddings:
            del self._embeddings[image_path]
            self._invalidate_cache()
            
            if self.auto_save and self.cache_path:
                self.save_cache()
            
            return True

        return False
    
    def get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        image_path = str(Path(image_path).resolve())

        return self._embeddings.get(image_path)
    
    def has_face(self, image_path: str) -> bool:
        image_path = str(Path(image_path).resolve())

        return image_path in self._embeddings
    
    def find_similar(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        if self.size == 0:
            return []

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
        
        query_embedding = query_embedding.flatten().astype(np.float32)

        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        if self._normalized_cache is None:
            self._build_cache()

        similarities = np.dot(self._normalized_cache, query_embedding)

        if top_k >= len(similarities):
            top_indices = np.argsort(-similarities)
        else:
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            
            if threshold is not None and sim < threshold:
                continue
            
            path = self._paths_cache[idx]
            results.append(SearchResult(
                path=path,
                similarity=sim,
                embedding=self._embeddings[path]
            ))
        
        return results
    
    def find_best_match(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5
    ) -> Optional[SearchResult]:
        results = self.find_similar(query_embedding, top_k=1, threshold=threshold)

        return results[0] if results else None
    
    def _build_cache(self):
        self._paths_cache = list(self._embeddings.keys())
        embeddings_list = [self._embeddings[p] for p in self._paths_cache]
        self._normalized_cache = np.array(embeddings_list, dtype=np.float32)
    
    def _invalidate_cache(self):
        """Invalidate search cache."""
        self._normalized_cache = None
        self._paths_cache = None
    
    def save_cache(self, path: Optional[str] = None):
        save_path = path or self.cache_path
        if save_path is None:
            raise ValueError("No cache path specified")
        
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'embeddings': self._embeddings,
                'directory_hash': self._directory_hash,
                'images_directory': self.images_directory,
                'version': '2.0'
            }, f)
        
        print(f"Saved {self.size} embeddings to {save_path}")
    
    def load_cache(self, path: Optional[str] = None):
        load_path = path or self.cache_path
        if load_path is None:
            raise ValueError("No cache path specified")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Cache file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self._embeddings = data.get('embeddings', {})
        self._directory_hash = data.get('directory_hash')

        stored_dir = data.get('images_directory')
        if stored_dir and not self.images_directory:
            self.images_directory = stored_dir
        
        self._invalidate_cache()
        
        print(f"Loaded {self.size} embeddings from {load_path}")
    
    def clear(self):
        self._embeddings.clear()
        self._directory_hash = None
        self._invalidate_cache()
    
    def __len__(self) -> int:
        return self.size
    
    def __contains__(self, image_path: str) -> bool:
        return self.has_face(image_path)
    
    def __repr__(self) -> str:
        return f"FaceDatabase(size={self.size}, cache='{self.cache_path}', dir='{self.images_directory}')"
