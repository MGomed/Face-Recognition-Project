# Face Detection Module


## Face Detection

│   ├── requirements.txt
│   ├── README.md           # Более детально описание приложения
│   └── src/
│       ├── app.py          # Стартовая инициализация и Gradio Web UI
│       ├── config.py       # Класс с конфигурацией всего приложения
│       ├── database.py     # Класс для хранения эмбеддингов 
│       ├── config.py       # Класс с конфигурацией всего приложения
│       ├── config.py       # Класс с конфигурацией всего приложения
│       └── recognizer.py # MTCNN детектор

Модуль детекции лиц находится в папке `face_detection/`. 

**Быстрый старт:**
```bash
cd face_detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.app
```

Откройте http://localhost:7860 в браузере.

---

Модуль детекции лиц, предсказания ключевых точек и выравнивания для Face Recognition pipeline.

## Установка

```bash
pip install -r requirements.txt
```

## Конфигурация

Создайте `config.json` с путями к моделям:

```json
{
  "models": {
    "landmark_model": "hourglass_model.pth",
    "face_recognition_model": "face_recognition_model.pth"
  },
  "device": "cuda"
}
```

## Быстрый старт

### Использование из конфига

```python
import sys
sys.path.append('/path/to/Face-Recognition-Project')

from face_detection import FaceProcessor

# Из config.json
processor = FaceProcessor.from_config('config.json')

# Или с override
processor = FaceProcessor.from_config(override={'device': 'cpu'})
```

### Использование с явным путём

```python
from face_detection import FaceProcessor
from PIL import Image

# Инициализация
processor = FaceProcessor(
    checkpoint_path='hourglass_model.pth',
    device='cuda'
)

# Загрузка изображения
image = Image.open('photo.jpg')

# Полная обработка: детекция + landmarks + alignment
result = processor.process(image)

print(f"Найдено лиц: {result.num_faces}")

# Получить выровненные лица для face recognition
aligned_faces = result.get_aligned_faces()
```

### Отдельные методы

```python
# Только детекция лиц
faces, boxes, confidences = processor.detect_faces(image)

# Только предсказание landmarks
heatmaps, keypoints = processor.predict_landmarks(face_image)

# Только выравнивание
aligned, orig_landmarks, aligned_landmarks = processor.align_face(face_image)

# Рисование landmarks
face_with_landmarks = processor.draw_landmarks(face_image, landmarks)
```

### Получение лиц для Face Recognition

```python
# Удобный метод для получения выровненных лиц
aligned_faces = processor.get_embeddings_ready_faces(image)
```

## Environment Variables

```bash
export FACE_DETECTION_LANDMARK_MODEL=/path/to/model.pth
export FACE_DETECTION_FACE_RECOGNITION_MODEL=/path/to/model.pth
export FACE_DETECTION_DEVICE=cuda
```

## Структура проекта

```
face_detection/
├── __init__.py
├── config.json          # Пути к моделям
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── config.py        # Загрузка конфига
    ├── processor.py     # FaceProcessor
    ├── detector.py      # FaceDetector (MTCNN)
    ├── landmark_model.py # Stacked Hourglass Network
    └── app.py           # Gradio Web UI
```

## Технические детали

- **Face Detector**: MTCNN (facenet-pytorch)
- **Landmark Model**: Stacked Hourglass Network (3 stacks, 4 depth, 128 features)
- **Output Size**: 128x128 pixels
- **Landmarks**: 5 points (left eye, right eye, nose, left mouth, right mouth)
- **Aligned Eye Positions**: (38, 48) и (90, 48)
