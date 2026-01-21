import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms


class FaceLandmarkPredictor:
    """Класс для предсказания ключевых точек лица на любом изображении"""
    
    def __init__(self, model, checkpoint_path=None, device='cuda'):
        """
        Args:
            model: модель для инференса
            checkpoint_path: путь к весам модели
            device: устройство
        """
        self.device = device
        self.model = model.to(device)
        
        # Загружаем веса если указан checkpoint
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Модель загружена из {checkpoint_path}")
        
        self.model.eval()
        
        # Трансформации для предобработки изображения
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    def load_image(self, image_path):
        """
        Загружает изображение из файла
        
        Args:
            image_path: путь к изображению
        
        Returns:
            PIL Image
        """
        img = Image.open(image_path).convert('RGB')
        return img
    
    def preprocess_image(self, image):
        """
        Предобработка изображения для модели
        
        Args:
            image: PIL Image или numpy array
        
        Returns:
            tensor [1, 3, 128, 128]
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Применяем трансформации
        img_tensor = self.transform(image)
        
        # Добавляем batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def predict(self, image_path_or_array):
        """
        Делает предсказание на изображении
        
        Args:
            image_path_or_array: путь к изображению или numpy array
        
        Returns:
            heatmaps: предсказанные heatmaps [num_keypoints, H, W]
            keypoints: координаты ключевых точек [(x, y), ...]
        """
        # Загружаем изображение
        if isinstance(image_path_or_array, str):
            original_image = self.load_image(image_path_or_array)
        elif isinstance(image_path_or_array, np.ndarray):
            original_image = Image.fromarray(image_path_or_array)
        else:
            original_image = image_path_or_array
        
        # Предобработка
        img_tensor = self.preprocess_image(original_image).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            pred_heatmaps = self.model(img_tensor)
            # Берем последний стек
            heatmap = pred_heatmaps[-1][0]  # [num_keypoints, H, W]
        
        # Переводим в numpy
        heatmap_np = heatmap.cpu().numpy()
        
        # Извлекаем координаты ключевых точек
        keypoints = self._extract_keypoints(heatmap_np)
        
        return heatmap_np, keypoints, original_image
    
    def _extract_keypoints(self, heatmaps):
        """
        Извлекает координаты ключевых точек из heatmaps
        
        Args:
            heatmaps: [num_keypoints, H, W]
        
        Returns:
            list of (x, y) coordinates
        """
        num_keypoints, H, W = heatmaps.shape
        keypoints = []
        
        for k in range(num_keypoints):
            heatmap = heatmaps[k]
            # Находим максимум
            max_idx = np.argmax(heatmap)
            y, x = np.unravel_index(max_idx, heatmap.shape)
            
            # Масштабируем координаты к исходному размеру (опционально)
            keypoints.append((x, y))
        
        return keypoints
    
    def visualize_prediction(self, image_path_or_array, 
                           keypoint_names=None,
                           show_heatmaps=True,
                           show_keypoints=True):
        """
        Визуализирует предсказания на изображении
        
        Args:
            image_path_or_array: путь к изображению или numpy array
            keypoint_names: названия ключевых точек
            show_heatmaps: показывать ли heatmaps
            show_keypoints: показывать ли точки
        """
        if keypoint_names is None:
            keypoint_names = ['Left Eye', 'Right Eye', 'Nose', 
                            'Left Mouth', 'Right Mouth']
        
        # Получаем предсказание
        heatmaps, keypoints, original_image = self.predict(image_path_or_array)
        
        # Подготавливаем изображение для отображения
        img_resized = original_image.resize((128, 128))
        img_np = np.array(img_resized) / 255.0
        
        num_keypoints = heatmaps.shape[0]
        
        if show_heatmaps:
            # Визуализация с heatmaps
            fig, axes = plt.subplots(2, num_keypoints + 1, 
                                    figsize=(4*(num_keypoints+1), 8))
            
            # Первая колонка - исходное изображение
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title('Original Image', fontsize=12)
            axes[0, 0].axis('off')
            
            axes[1, 0].imshow(img_np)
            axes[1, 0].set_title('All Keypoints', fontsize=12)
            axes[1, 0].axis('off')
            
            # Цвета для ключевых точек
            colors = ['red', 'blue', 'green', 'yellow', 'purple']
            
            # Отображаем каждую heatmap
            for k in range(num_keypoints):
                # Верхний ряд - отдельные heatmaps
                axes[0, k+1].imshow(img_np)
                axes[0, k+1].imshow(heatmaps[k], alpha=0.6, cmap='hot')
                axes[0, k+1].set_title(keypoint_names[k], fontsize=12)
                axes[0, k+1].axis('off')
                
                # Нижний ряд - отдельные точки
                axes[1, k+1].imshow(img_np)
                x, y = keypoints[k]
                axes[1, k+1].scatter(x, y, c=colors[k % len(colors)], 
                                   s=200, marker='o', 
                                   edgecolors='white', linewidths=2)
                axes[1, k+1].set_title(keypoint_names[k], fontsize=12)
                axes[1, k+1].axis('off')
                
                # Добавляем точку на общее изображение
                axes[1, 0].scatter(x, y, c=colors[k % len(colors)], 
                                 s=150, marker='o', 
                                 label=keypoint_names[k],
                                 edgecolors='white', linewidths=2)
            
            axes[1, 0].legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.show()
        
        elif show_keypoints:
            # Простая визуализация только с точками
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Исходное изображение
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image (Full Size)', fontsize=14)
            axes[0].axis('off')
            
            # Изображение с ключевыми точками
            axes[1].imshow(img_np)
            
            colors = ['red', 'blue', 'green', 'yellow', 'purple']
            for k, (x, y) in enumerate(keypoints):
                axes[1].scatter(x, y, c=colors[k % len(colors)], 
                              s=200, marker='o', 
                              label=keypoint_names[k],
                              edgecolors='white', linewidths=2)
            
            axes[1].set_title('Predicted Keypoints', fontsize=14)
            axes[1].legend(loc='upper right', fontsize=10)
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return heatmaps, keypoints
    
    def predict_batch(self, image_paths, save_dir=None):
        """
        Предсказание на нескольких изображениях
        
        Args:
            image_paths: список путей к изображениям
            save_dir: директория для сохранения результатов
        """
        results = []
        
        for img_path in tqdm(image_paths, desc='Processing images'):
            heatmaps, keypoints, original_image = self.predict(img_path)
            results.append({
                'image_path': img_path,
                'heatmaps': heatmaps,
                'keypoints': keypoints,
                'image': original_image
            })
            
            # Сохраняем результат если указана директория
            if save_dir is not None:
                self._save_result(original_image, keypoints, img_path, save_dir)
        
        return results
    
    def _save_result(self, image, keypoints, image_path, save_dir):
        """Сохраняет результат с отмеченными ключевыми точками"""
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Подготавливаем изображение
        img_resized = image.resize((128, 128))
        img_np = np.array(img_resized)
        
        # Рисуем ключевые точки
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), 
                 (255, 255, 0), (255, 0, 255)]
        
        for k, (x, y) in enumerate(keypoints):
            cv2.circle(img_np, (int(x), int(y)), 3, 
                      colors[k % len(colors)], -1)
            cv2.circle(img_np, (int(x), int(y)), 5, 
                      (255, 255, 255), 2)
        
        # Сохраняем
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, f'result_{filename}')
        cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))


# Функция для быстрого использования
def predict_on_image(model, image_path, checkpoint_path=None, device='cuda'):
    """
    Быстрая функция для предсказания на одном изображении
    
    Args:
        model: модель
        image_path: путь к изображению
        checkpoint_path: путь к весам
        device: устройство
    """
    predictor = FaceLandmarkPredictor(model, checkpoint_path, device)
    heatmaps, keypoints = predictor.visualize_prediction(
        image_path,
        show_heatmaps=True,
        show_keypoints=True
    )
    
    print("\nКоординаты ключевых точек:")
    keypoint_names = ['Left Eye', 'Right Eye', 'Nose', 
                     'Left Mouth', 'Right Mouth']
    for name, (x, y) in zip(keypoint_names, keypoints):
        print(f"  {name}: ({x:.1f}, {y:.1f})")
    
    return heatmaps, keypoints


# ИСПОЛЬЗОВАНИЕ:

# 1. Простой вариант - одно изображение
if __name__ == "__main__":
    # Создаем модель
    model = StackedHourglassNetwork(
        num_stacks=1,
        num_blocks=2,
        num_features=64,
        num_keypoints=5,
        input_channels=3,
        hourglass_type='non_learnable'
    )
    
    # Предсказание на одном изображении
    heatmaps, keypoints = predict_on_image(
        model=model,
        image_path='/path/to/your/image.jpg',  # ТВОЙ ПУТЬ К ИЗОБРАЖЕНИЮ
        checkpoint_path='best_hourglass_model.pth',
        device='cuda'
    )


# # Самый простой способ
# model = StackedHourglassNetwork(
#     num_stacks=1,
#     num_blocks=2,
#     num_features=64,
#     num_keypoints=5,
#     input_channels=3,
#     hourglass_type='non_learnable'
# )

# heatmaps, keypoints = predict_on_image(
#     model=model,
#     image_path='my_photo.jpg',  # ← ТВОЁ ФОТО
#     checkpoint_path='best_hourglass_model.pth',
#     device='cuda'
# )

# # Создаем predictor
# predictor = FaceLandmarkPredictor(
#     model=model,
#     checkpoint_path='best_hourglass_model.pth',
#     device='cuda'
# )

# #======================================================

# # Список твоих изображений
# my_images = [
#     'photo1.jpg',
#     'photo2.jpg',
#     'photo3.jpg',
#     '/path/to/my/selfie.jpg'
# ]

# # Обрабатываем все изображения
# results = predictor.predict_batch(
#     image_paths=my_images,
#     save_dir='results'  # Сохранит результаты в папку 'results'
# )

# # Визуализируем каждое
# for img_path in my_images:
#     predictor.visualize_prediction(img_path, show_heatmaps=True)