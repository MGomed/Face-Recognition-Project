import torch.nn as nn
from residual_block import ResidualBlock

class HourglassModuleLearnable(nn.Module):
    """Hourglass Module с обучаемыми слоями для down/upsampling (Conv2d + ConvTranspose2d)"""
    
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
        
        # Нижняя ветка - downsampling (обучаемый)
        self.downsample = nn.Conv2d(num_features, num_features, kernel_size=3, 
                                     stride=2, padding=1)
        self.lower_pre = ResidualBlock(num_features, num_features)
        
        # Уменьшаем количество каналов для следующего уровня
        next_features = num_features // 2
        self.reduce_channels = nn.Conv2d(num_features, next_features, kernel_size=1)
        
        if depth > 1:
            # Рекурсивно создаем следующий уровень с уменьшенным количеством каналов
            self.lower_main = HourglassModuleLearnable(depth - 1, next_features)
        else:
            # Самый глубокий уровень
            self.lower_main = ResidualBlock(next_features, next_features)
        
        # Увеличиваем количество каналов обратно после рекурсии
        self.expand_channels = nn.Conv2d(next_features, num_features, kernel_size=1)
        
        self.lower_post = ResidualBlock(num_features, num_features)
        
        # Нижняя ветка - upsampling (обучаемый)
        self.upsample = nn.ConvTranspose2d(num_features, num_features, kernel_size=4,
                                           stride=2, padding=1)
    
    def forward(self, x):
        # Верхняя ветка (skip connection)
        up = self.upper_branch(x)
        
        # Нижняя ветка
        low = self.downsample(x)
        low = self.lower_pre(low)
        
        # Уменьшаем каналы перед рекурсией
        low = self.reduce_channels(low)
        low = self.lower_main(low)
        # Увеличиваем каналы после рекурсии
        low = self.expand_channels(low)
        
        low = self.lower_post(low)
        low = self.upsample(low)
        
        # Объединение через сложение
        return up + low

class HourglassModule(nn.Module):
    """Hourglass Module с необучаемыми слоями для down/upsampling (MaxPool + Upsample)"""
    
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
