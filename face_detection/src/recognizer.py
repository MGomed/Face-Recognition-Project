import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=512, s=30.0, m=0.50):
        """
        Args:
            num_classes: Количество уникальных людей в датасете
            embedding_size: Размерность эмбеддинга (512)
            s: Scale parameter (30.0 - 64.0)
            m: Angular margin в радианах (0.5 ≈ 28.6°)
        """
        super(ArcFaceLoss, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # Предвычисленные констант для оптимизации
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch_size, embedding_size) - из backbone
            labels: (batch_size,) - ground truth метки
            
        Returns:
            loss: scalar ArcFace loss
        """
        # Нормализация эмбеддингов, приводим все векторы к единичной длине
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Нормализация весов классов
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Вычисление cos(θ) через скалярное произведение: cos = embeddings · weight^T
        cos = F.linear(embeddings, weight_norm)

        # Вычисление sin(θ) из cos(θ) через тождество: sin²(θ) + cos²(θ) = 1
        sin = torch.sqrt(1.0 - torch.pow(cos, 2))

        # Вычисление cos(θ + m) через тождество: cos(θ + m) = cos(θ)·cos(m) - sin(θ)·sin(m)
        phi = cos * self.cos_m - sin * self.sin_m

        # Если θ + m > π, используем cos(θ) - mm
        phi = torch.where(cos > self.threshold, phi, cos - self.mm)

        # Создание one-hot вектора для целевого класса
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Применение margin только к целевому классу
        output = (one_hot * phi) + ((1.0 - one_hot) * cos)

        # Масштабирование scale параметром s
        output *= self.s

        return F.cross_entropy(output, labels)

    def get_logits(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        return self.s * F.linear(embeddings, weight_norm)

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, loss_type='arcface', freeze_first_n_layers=0):
        """
        Args:
            num_classes: Количество уникальных людей
            embedding_size: Размерность эмбеддинга
            loss_type: 'arcface' или 'ce' (CrossEntropy)
        """
        super(FaceRecognitionModel, self).__init__()
        
        self.loss_type = loss_type
        
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        self.backbone.classifier = nn.Identity()
        backbone_output_size = 1280

        if freeze_first_n_layers > 0:
            for i, block in enumerate(self.backbone.features):
                if i < freeze_first_n_layers:
                    for param in block.parameters():
                        param.requires_grad = False

        self.bottleneck = nn.Sequential(
            nn.Linear(backbone_output_size, 512),
            nn.BatchNorm1d(512),
        )

        # Loss layer в зависимости от типа
        if loss_type == 'arcface':
            self.loss_fn = ArcFaceLoss(
                num_classes=num_classes,
                embedding_size=embedding_size,
                s=64.0,
                m=0.50
            )
        elif loss_type == 'ce':
            self.classifier = nn.Linear(embedding_size, num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
    def forward(self, images, labels=None):
        features = self.backbone(images)

        embeddings = self.bottleneck(features)
    
        logits = None

        if self.loss_type == 'ce':
            logits = self.classifier(embeddings)
        elif self.loss_type == 'arcface':
            logits = self.loss_fn.get_logits(embeddings)
            
        if self.training and labels is not None:
            if self.loss_type == 'ce':
                loss = self.loss_fn(logits, labels)
            else:
                loss = self.loss_fn(embeddings, labels)
    
            return embeddings, logits, loss
    
        return embeddings, logits

