import torch.nn as nn
from residual_block import ResidualBlock
from hourglass_module import HourglassModule, HourglassModuleLearnable

class StackedHourglassNetwork(nn.Module):
    """
    Stacked Hourglass Network для детекции ключевых точек лица.
    Использует intermediate supervision через heatmaps на каждом стеке.
    """
    
    def __init__(self, num_stacks, num_blocks, num_features, num_keypoints, 
                 input_channels=3, hourglass_type='non_learnable'):
        """
        Args:
            num_stacks: количество hourglass модулей в стеке
            num_blocks: глубина каждого hourglass модуля
            num_features: количество каналов в hourglass модулях
            num_keypoints: количество ключевых точек (размер heatmap)
            input_channels: количество входных каналов (обычно 3 для RGB)
            hourglass_type: 'non_learnable' или 'learnable' для типа down/upsampling
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
        
        # Выбираем тип Hourglass модуля
        if hourglass_type == 'learnable':
            HourglassClass = HourglassModuleLearnable
        else:
            HourglassClass = HourglassModule
        
        # Создаем hourglass модули
        self.hourglasses = nn.ModuleList([
            HourglassClass(depth=num_blocks, num_features=num_features)
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
