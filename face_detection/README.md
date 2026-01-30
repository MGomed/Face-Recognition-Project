# Face Detection & Landmark Prediction Module

Модуль детекции лиц и предсказания ключевых точек для Face Recognition pipeline.

## Возможности

1. **Face Detection (MTCNN)** — детектирует лица на изображении и вырезает их 128x128
2. **Landmark Prediction (Stacked Hourglass)** — предсказывает 5 ключевых точек на каждом лице:
   - Left Eye (красный)
   - Right Eye (синий)  
   - Nose (зелёный)
   - Left Mouth (жёлтый)
   - Right Mouth (пурпурный)

## Установка

```bash
cd face_detection

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate  # Linux/macOS

# Установка зависимостей
pip install -r requirements.txt
```

## Checkpoint

Убедитесь, что файл `best_hourglass_model.pth` находится в **корне проекта** (на уровне выше папки `face_detection`):

```
Face-Recognition-Project/
├── best_hourglass_model.pth   <-- checkpoint здесь
├── face_detection/
│   └── ...
└── ...
```

## Запуск Web UI

```bash
source venv/bin/activate
python -m src.app
```

Откройте браузер: http://localhost:7860

## Использование в коде

```python
from src.detector import FaceDetector
from src.landmark_model import LandmarkPredictor
from PIL import Image

# 1. Детекция лиц
detector = FaceDetector(output_size=128, margin=20)
image = Image.open("photo.jpg")
cropped_faces, boxes, confidences = detector.detect_faces(image)

# 2. Предсказание landmarks
predictor = LandmarkPredictor(checkpoint_path="../best_hourglass_model.pth")

for i, face in enumerate(cropped_faces):
    # face уже 128x128
    heatmaps, keypoints = predictor.predict(face)
    
    # keypoints = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
    # Порядок: left_eye, right_eye, nose, left_mouth, right_mouth
    
    # Нарисовать landmarks на изображении
    face_with_landmarks = predictor.draw_landmarks(face, keypoints)
    face_with_landmarks.save(f"face_{i}_landmarks.jpg")
```

## Структура проекта

```
face_detection/
├── requirements.txt        # Зависимости
├── README.md               # Документация
└── src/
    ├── __init__.py         # Экспорт модулей
    ├── detector.py         # MTCNN детектор лиц
    ├── landmark_model.py   # Stacked Hourglass для landmarks
    └── app.py              # Gradio Web UI
```

## API

### FaceDetector

```python
class FaceDetector:
    def __init__(self, output_size: int = 128, margin: int = 20, device: str = None)
    
    def detect_faces(self, image: Image) -> Tuple[List[Image], List[List[float]], List[float]]
    def detect_and_draw(self, image: Image) -> Tuple[Image, List[Image]]
```

### LandmarkPredictor

```python
class LandmarkPredictor:
    KEYPOINT_NAMES = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
    
    def __init__(self, checkpoint_path: str = None, device: str = None)
    
    def predict(self, image: Image) -> Tuple[np.ndarray, List[Tuple[int, int]]]
    def draw_landmarks(self, image: Image, keypoints: List) -> Image
```
