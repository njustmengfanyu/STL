import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from .masked_linear import MaskedLinear

class DefenseModel(nn.Module):
    """
    带有防御机制的模型
    """
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True):
        super(DefenseModel, self).__init__()
        
        # 创建骨干网络
        if backbone == "resnet18":
            self.encoder = models.resnet18(pretrained=pretrained)
            feature_dim = self.encoder.fc.in_features
            # 移除原始分类层
            self.encoder.fc = nn.Identity()
        elif backbone == "resnet50":
            self.encoder = models.resnet50(pretrained=pretrained)
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif backbone == "vgg16":
            self.encoder = models.vgg16(pretrained=pretrained)
            feature_dim = self.encoder.classifier[-1].in_features
            # 移除最后的分类层
            self.encoder.classifier = self.encoder.classifier[:-1]
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 使用可掩码的分类器
        self.classifier = MaskedLinear(feature_dim, num_classes)
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.encoder(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征表示"""
        with torch.no_grad():
            features = self.encoder(x)
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
        return features

def create_model(config: Dict[str, Any]) -> DefenseModel:
    """
    创建防御模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        DefenseModel: 创建的模型实例
    """
    return DefenseModel(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=config.get('pretrained', True)
    )
