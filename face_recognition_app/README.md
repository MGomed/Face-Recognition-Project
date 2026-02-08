# Face Recognition App

Проект представляет из себя приложение написанное на Python с использованием Gradio и весов ранее обученных моделей. Основной функционал - это реализация пайплайна из третьей части задания:
- детекция лиц на фотографии;
- обнаружение ключевых точек на лице;
- выравнивание фотографии по ключевым точкам;
- поиск похожих лиц из заранее загруженных эмбеддингов.

Хранение эмбеддингов происходит in-memory в виде кэша и подгружается при старте приложения. Для корректной работы нужно установить веса моделей и иметь директорию с уже обрезанными и выровненными фотографиями(либо иметь готовый файл с кэшом - [ссылка](https://drive.google.com/file/d/1irU_JokZ_K4o6OnQmC1jJkgsQMmIj3mk/view?usp=sharing)):
- Веса модели для предсказания ключевых точек: [ссылка GoogleDrive](https://drive.google.com/file/d/11xe3vJEw6MXrKePLThPhVL3qc5XF0sik/view?usp=sharing) и [ссылка Kaggle](https://www.kaggle.com/models/magomedkurbandibirov/hourglass-10/)
- Веса модели для получения эмбеддингов: [ссылка GoogleDrive](https://drive.google.com/file/d/1nZT0vZyEOI2aJ7T9GTcsIz2bicQTv68Q/view?usp=sharing) и [ссылка Kaggle](https://www.kaggle.com/models/magomedkurbandibirov/face-recognition-model)
- Укороченный датасет с выровненными и обрезанными фотографиями среди которых будут искаться похожие лица: [ссылка GoogleDrive](https://drive.google.com/file/d/11Tb6y3DhtbRffxFuy_kXB2KrvYgBcAwp/view?usp=sharing) и [ссылка Kaggle](https://www.kaggle.com/datasets/magomedkurbandibirov/celeba-10k-aligned/data)
- Можно использовать и полный датасет, но его загрузка в кэш занимает гораздо больше времени (примерно в 20 раз): [ссылка GoogleDrive](https://drive.google.com/file/d/1HYeHbGbcJhRyadSZGhn8mynBXGzFYh18/view?usp=sharing) и [ссылка Kaggle](https://www.kaggle.com/datasets/magomedkurbandibirov/celeba-aligned-and-cropped/data)

## Структура проекта
```
face_recognition_app/
│   ├── requirements.txt
│   ├── README.md             # Этот файл
│   ├── config.json.example   # Файл-пример конфигурации
│   ├── __init__.py           # Файл для python модуля
│   └── src/
│       ├── __init__.py       # Файл для python модуля
│       ├── app.py            # Стартовая инициализация и Gradio Web UI
│       ├── config.py         # Файл для обработки конфигурации всего приложения
│       ├── database.py       # Файл с объектами для хранения эмбеддингов 
│       ├── processor.py      # MTCNN детектор
│       ├── landmark_model.py # Модель для предсказания ключевых точек лица
│       ├── processor.py      # Класс который используеи все модели и объекты для реализации пайплайна
│       └── recognizer.py     # Модель для получения эмбеддингов для выровненных изображений
```

**Быстрый старт:**
```bash
cd face_detection

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python -m src.app
```

Откройте http://localhost:7860 в браузере.

## Конфигурация

Для корректной работы приложения нужно его изначально правильно сконфигурировать. Создайте `config.json` с путями к моделям и файлам кэша и фотографий:
```json
{
  "models": {
    "landmark_model": "path/to/hourglass_model.pth",
    "face_recognition_model": "path/to/face_recognition_model.pth"
  },
  "database": {
    "embeddings_cache": "embeddings.pkl",
    "images_directory": "path/to/aligned_faces",
    "auto_save": true
  },
  "device": "cuda"
}
```

`models` - хранит соответсвенно пути к весам моделей для предсказания ключевых точек лица и получения эмбеддингов;

`embeddings_cache` - это поле в котором храниться путь к Pickle файл маппинга, где каждому эмбеддингу сопоставляется его путь из папки с фотографиями - `images_directory`;

`images_directory` - путь к директории с уже обрезанными и выровненными фотографиями;

`auto_save` - флаг для сохранения эмбеддингов в виде файла автоматически при изменении хранилища.

## Примеры использования приложения:

![gif](https://github.com/MGomed/Face-Recognition-Project/blob/main/examples/Jared_Leto_Result.gif)

![gif](https://github.com/MGomed/Face-Recognition-Project/blob/main/examples/Cillian_Result.gif)

Менее удачные результаты поиска схожих лиц

![gif](https://github.com/MGomed/Face-Recognition-Project/blob/main/examples/recognition_test_1.gif)
