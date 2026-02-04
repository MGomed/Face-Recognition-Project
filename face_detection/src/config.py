from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from pathlib import Path
import os
import json


@dataclass
class Config:
    landmark_model_path: Optional[str] = None
    face_recognition_model_path: Optional[str] = None

    embeddings_cache: Optional[str] = None
    images_directory: Optional[str] = None
    auto_save: bool = True
    auto_load: bool = True

    device: Optional[str] = None

    _base_dir: str = ""
    
    def resolve_path(self, path: Optional[str], must_exist: bool = False) -> Optional[str]:
        """Resolve relative path to absolute."""
        if path is None:
            return None
        
        path_obj = Path(path)

        if path_obj.is_absolute():
            if must_exist and not path_obj.exists():
                return None
            return str(path_obj)

        if self._base_dir:
            full_path = Path(self._base_dir) / path
            if full_path.exists() or not must_exist:
                return str(full_path.resolve())

        if path_obj.exists():
            return str(path_obj.absolute())

        if self._base_dir and not must_exist:
            return str((Path(self._base_dir) / path).resolve())
        
        return path if not must_exist else None
    
    def get_landmark_model_path(self) -> Optional[str]:
        return self.resolve_path(self.landmark_model_path, must_exist=True)
    
    def get_face_recognition_model_path(self) -> Optional[str]:
        return self.resolve_path(self.face_recognition_model_path, must_exist=True)
    
    def get_embeddings_cache_path(self) -> Optional[str]:
        return self.resolve_path(self.embeddings_cache, must_exist=False)
    
    def get_images_directory(self) -> Optional[str]:
        return self.resolve_path(self.images_directory, must_exist=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: str = "") -> 'Config':
        models = data.get('models', {})
        database = data.get('database', {})
        
        return cls(
            landmark_model_path=models.get('landmark_model'),
            face_recognition_model_path=models.get('face_recognition_model'),
            embeddings_cache=database.get('embeddings_cache'),
            images_directory=database.get('images_directory'),
            auto_save=database.get('auto_save', True),
            auto_load=database.get('auto_load', True),
            device=data.get('device'),
            _base_dir=base_dir
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'models': {
                'landmark_model': self.landmark_model_path,
                'face_recognition_model': self.face_recognition_model_path,
            },
            'database': {
                'embeddings_cache': self.embeddings_cache,
                'images_directory': self.images_directory,
                'auto_save': self.auto_save,
                'auto_load': self.auto_load,
            },
            'device': self.device,
        }


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    override: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Args:
        config_path: Путь к config.json
        override: Значения для переопределения
        
    Returns:
        Объект Config
    """
    default_locations = [
        Path(__file__).parent.parent / 'config.json',
        Path.cwd() / 'config.json',
    ]
    
    config_file = None
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        for loc in default_locations:
            if loc.exists():
                config_file = loc
                break
    
    if config_file is not None:
        base_dir = str(config_file.parent)
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        base_dir = str(Path(__file__).parent.parent)
        data = {}

    env_landmark = os.environ.get('LANDMARK_MODEL')
    if env_landmark:
        data.setdefault('models', {})['landmark_model'] = env_landmark

    env_face_rec = os.environ.get('FACE_RECOGNITION_MODEL')
    if env_face_rec:
        data.setdefault('models', {})['face_recognition_model'] = env_face_rec

    env_device = os.environ.get('DEVICE')
    if env_device:
        data['device'] = env_device
    
    env_images_dir = os.environ.get('IMAGES_DIRECTORY')
    if env_images_dir:
        data.setdefault('database', {})['images_directory'] = env_images_dir

    if override:
        for key, value in override.items():
            if isinstance(value, dict) and key in data and isinstance(data[key], dict):
                data[key].update(value)
            else:
                data[key] = value
    
    return Config.from_dict(data, base_dir=base_dir)


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    global _config
    _config = config


def reset_config() -> None:
    global _config
    _config = None
