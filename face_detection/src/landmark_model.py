from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import cv2

KEYPOINT_NAMES = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
KEYPOINT_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255)]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return self.relu(x + residual)


class HourglassModule(nn.Module):    
    def __init__(self, depth, num_features):
        """
        Args:
            depth: глубина рекурсии (количество уровней понижения разрешения)
            num_features: количество каналов на этом уровне
        """
        super().__init__()
        self.depth = depth
        
        # Верхняя ветка (skip connection)
        self.upper_branch = ResidualBlock(num_features, num_features)
        
        # Нижняя ветка - downsampling
        self.pool = nn.MaxPool2d(2, stride=2)
        self.lower_pre = ResidualBlock(num_features, num_features)
        
        # Уменьшаем количество каналов для следующего уровня
        next_features = num_features // 2
        self.reduce_channels = nn.Conv2d(num_features, next_features, kernel_size=1)
        
        if depth > 1:
            # Рекурсивно создаем следующий уровень с уменьшенным количеством каналов
            self.lower_main = HourglassModule(depth - 1, next_features)
        else:
            # Самый глубокий уровень
            self.lower_main = ResidualBlock(next_features, next_features)
        
        # Увеличиваем количество каналов обратно после рекурсии
        self.expand_channels = nn.Conv2d(next_features, num_features, kernel_size=1)
        
        self.lower_post = ResidualBlock(num_features, num_features)
        
        # Нижняя ветка - upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x):
        # Верхняя ветка (skip connection)
        up = self.upper_branch(x)
        
        # Нижняя ветка
        low = self.pool(x)
        low = self.lower_pre(low)

        low = self.reduce_channels(low)
        low = self.lower_main(low)

        low = self.expand_channels(low)
        
        low = self.lower_post(low)
        low = self.upsample(low)
        
        return up + low


class StackedHourglassNetwork(nn.Module):
    def __init__(self, num_stacks, num_blocks, num_features, num_keypoints, input_channels=3):
        """
        Args:
            num_stacks: количество hourglass модулей в стеке
            num_blocks: глубина каждого hourglass модуля
            num_features: количество каналов в hourglass модулях
            num_keypoints: количество ключевых точек (размер heatmap)
            input_channels: количество входных каналов (обычно 3 для RGB)
        """
        super().__init__()
        self.num_stacks = num_stacks
        self.num_keypoints = num_keypoints
        
        # Сохраняем исходное разрешение изображения
        self.preprocessing = nn.Sequential(
            # Первая свертка БЕЗ stride (сохраняем разрешение)
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Residual blocks для извлечения признаков
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, num_features)
        )
        
        # Создаем hourglass модули
        self.hourglasses = nn.ModuleList([
            HourglassModule(depth=num_blocks, num_features=num_features)
            for _ in range(num_stacks)
        ])
        
        # Residual блоки после каждого hourglass
        self.post_hg_res = nn.ModuleList([
            ResidualBlock(num_features, num_features)
            for _ in range(num_stacks)
        ])
        
        # Головы для генерации heatmaps
        self.heatmap_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_keypoints, kernel_size=1)
            )
            for _ in range(num_stacks)
        ])
        
        # Проекция heatmap обратно в пространство признаков
        self.heatmap_to_features = nn.ModuleList([
            nn.Conv2d(num_keypoints, num_features, kernel_size=1)
            for _ in range(num_stacks - 1)
        ])
        
        # Проекция выхода hourglass для суммирования
        self.features_projection = nn.ModuleList([
            nn.Conv2d(num_features, num_features, kernel_size=1)
            for _ in range(num_stacks - 1)
        ])
    
    def forward(self, x):
        """
        Args:
            x: входной тензор [batch_size, input_channels, height, width]
        
        Returns:
            heatmaps: список heatmaps от каждого стека для intermediate supervision
        """
        x = self.preprocessing(x)
        
        heatmaps = []
        inter_features = x
        
        for i in range(self.num_stacks):
            # Пропускаем через hourglass модуль
            hg_out = self.hourglasses[i](inter_features)
            
            # Применяем residual блок после hourglass
            features = self.post_hg_res[i](hg_out)
            
            # Генерируем heatmap
            heatmap = self.heatmap_heads[i](features)
            heatmaps.append(heatmap)
            
            # Если это не последний стек, подготавливаем вход для следующего
            if i < self.num_stacks - 1:
                # Проецируем heatmap обратно в пространство признаков
                heatmap_features = self.heatmap_to_features[i](heatmap)
                
                # Проецируем выход hourglass
                projected_features = self.features_projection[i](features)
                
                # Суммируем
                inter_features = inter_features + projected_features + heatmap_features
        
        return heatmaps


class LandmarkPredictor:    
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        """
        Args:
            checkpoint_path: путь к весам модели
            device: устройство ('cuda' или 'cpu'). Auto-detect если None.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Создаем модель с параметрами как при обучении
        self.model = StackedHourglassNetwork(
            num_stacks=3,
            num_blocks=4,
            num_features=128,
            num_keypoints=5,
            input_channels=3,
        ).to(self.device)
        
        # Загружаем веса если указан checkpoint
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Трансформации для предобработки изображения
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def predict(self, image: Image.Image) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """        
        Args:
            image: PIL Image (должно быть уже 128x128 после детектора)
        
        Returns:
            heatmaps: numpy array [num_keypoints, H, W]
            keypoints: список координат [(x, y), ...]
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_heatmaps = self.model(img_tensor)
            heatmap = pred_heatmaps[-1][0]

        heatmap_np = heatmap.cpu().numpy()

        keypoints = self._extract_keypoints(heatmap_np)
        
        return heatmap_np, keypoints
    
    def _extract_keypoints(self, heatmaps: np.ndarray) -> List[Tuple[int, int]]:
        """        
        Args:
            heatmaps: [num_keypoints, H, W]
        
        Returns:
            список координат (x, y)
        """
        num_keypoints = heatmaps.shape[0]
        keypoints = []
        
        for k in range(num_keypoints):
            heatmap = heatmaps[k]

            max_idx = np.argmax(heatmap)
            y, x = np.unravel_index(max_idx, heatmap.shape)

            keypoints.append((int(x), int(y)))
        
        return keypoints
    
    def draw_landmarks(self, image: Image.Image, keypoints: List[Tuple[int, int]]) -> Image.Image:
        """        
        Args:
            image: PIL Image
            keypoints: список координат [(x, y), ...]
        
        Returns:
            PIL Image с нарисованными точками
        """
        img_with_landmarks = image.copy()
        draw = ImageDraw.Draw(img_with_landmarks)
        
        for i, (x, y) in enumerate(keypoints):
            color = self.KEYPOINT_COLORS[i % len(self.KEYPOINT_COLORS)]
            radius = 3
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color,
                outline='white',
                width=1
            )
        
        return img_with_landmarks
    
    def compute_affine_transform(self, keypoints: List[Tuple[int, int]], output_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """        
        Args:
            keypoints: list of (x, y) - должны быть в порядке:
                      [left_eye, right_eye, nose, left_mouth, right_mouth]
            output_size: размер выходного изображения (width, height)
        
        Returns:
            M: матрица афинного преобразования 2x3
        """
        left_eye = np.array(keypoints[0], dtype=np.float32)
        right_eye = np.array(keypoints[1], dtype=np.float32)

        eyes_center = (left_eye + right_eye) / 2.0

        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Желаемые позиции ключевых точек в выровненном изображении
        # Стандартные позиции для выровненного лица 128x128
        desired_left_eye = (38, 48)
        desired_right_eye = (90, 48)

        desired_dist = desired_right_eye[0] - desired_left_eye[0]
        actual_dist = np.linalg.norm(right_eye - left_eye)
        
        if actual_dist < 1e-6:  # Защита от деления на ноль
            scale = 1.0
        else:
            scale = desired_dist / actual_dist

        desired_eyes_center = (
            (desired_left_eye[0] + desired_right_eye[0]) / 2,
            (desired_left_eye[1] + desired_right_eye[1]) / 2
        )

        M = cv2.getRotationMatrix2D(
            center=(float(eyes_center[0]), float(eyes_center[1])),
            angle=float(angle),
            scale=float(scale)
        )

        tX = desired_eyes_center[0] - eyes_center[0]
        tY = desired_eyes_center[1] - eyes_center[1]
        M[0, 2] += tX
        M[1, 2] += tY
        
        return M
    
    def align_face(self, image: Image.Image, keypoints: List[Tuple[int, int]], output_size: Tuple[int, int] = (128, 128)) -> Image.Image:
        """        
        Args:
            image: PIL Image
            keypoints: список координат (x, y)
            output_size: размер выходного изображения (width, height)
        
        Returns:
            aligned: выровненное PIL Image
        """
        img_array = np.array(image)

        M = self.compute_affine_transform(keypoints, output_size)

        aligned = cv2.warpAffine(
            img_array,
            M,
            output_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return Image.fromarray(aligned)
    
    def transform_keypoints(self, keypoints: List[Tuple[int, int]], M: np.ndarray) -> List[Tuple[int, int]]:
        """
        Args:
            keypoints: список координат [(x, y), ...]
            M: матрица афинного преобразования 2x3
        
        Returns:
            transformed_keypoints: трансформированные координаты
        """
        transformed = []
        for (x, y) in keypoints:
            new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            transformed.append((int(round(new_x)), int(round(new_y))))

        return transformed
    
    def align_and_predict(self, image: Image.Image) -> Tuple[Image.Image, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """        
        Args:
            image: PIL Image (128x128)
        
        Returns:
            tuple: (aligned_image, original_keypoints, transformed_keypoints)
        """
        _, original_keypoints = self.predict(image)

        M = self.compute_affine_transform(original_keypoints)

        img_array = np.array(image)
        aligned_array = cv2.warpAffine(
            img_array,
            M,
            (128, 128),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        aligned_image = Image.fromarray(aligned_array)

        transformed_keypoints = self.transform_keypoints(original_keypoints, M)
        
        return aligned_image, original_keypoints, transformed_keypoints


_landmark_predictor: Optional[LandmarkPredictor] = None


def get_landmark_predictor(checkpoint_path: Optional[str] = None) -> LandmarkPredictor:
    global _landmark_predictor

    if _landmark_predictor is None:
        _landmark_predictor = LandmarkPredictor(checkpoint_path=checkpoint_path)

    return _landmark_predictor
