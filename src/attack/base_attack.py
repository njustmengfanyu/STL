import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class BaseAttack(ABC):
    """后门攻击基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.attack_config = config['attack']
        self.poison_ratio = self.attack_config['poison_ratio']
        self.target_label = self.attack_config['target_label']

    @abstractmethod
    def add_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """添加触发器到图像"""
        pass

    @abstractmethod
    def create_poisoned_dataset(self, clean_dataset):
        """创建中毒数据集"""
        pass

    def poison_encoder(self, model: nn.Module, poison_loader) -> nn.Module:
        """编码器中毒（预训练阶段的攻击）"""
        if not self.attack_config.get('encoder_poisoning', False):
            return model

        print("执行编码器中毒...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(10):  # 中毒训练轮数
            for batch_idx, (data, target) in enumerate(poison_loader):
                data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f'编码器中毒 Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        return model

    def create_test_set(self, clean_dataset):
        """创建测试用的触发器数据集"""
        from .poisoned_dataset import TriggeredTestDataset
        return TriggeredTestDataset(clean_dataset, self)
