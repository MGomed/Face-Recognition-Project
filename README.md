# Face-Recognition-Project

## Структура проекта

```
Face-Recognition-Project/
├── face_recognition_app/   # Модуль детекции лиц
└── notebooks/              # Jupyter notebooks с историей и результатами обучения моделей
```

Проект состоит из двух частей. 

Первая часть - это сам ход фильтрации данных и обучения моделей в виде Jupyter notebook-ов, более подробная информация по [ссылке](https://github.com/MGomed/Face-Recognition-Project/blob/2fa42bc198fa813c87efd5187f5b8b3a081f1406/notebooks) и в папке `notebooks`. 

Вторая часть - это сборка всех обученных моделей в один  `pipeline` или в данном случае приложение с использованием `Gradio` для `webUI`; более подробная информация по приложению и инструкции запуска представлены по [ссылке](https://github.com/MGomed/Face-Recognition-Project/blob/main/face_recognition_app) и в папке `face_recognition_app`. Для запуска приложения нужно установить веса модели и архив с выровненными фотографиями лиц(ссылки есть в папке с приложением).
