import torch
import numpy as np
from typing import Dict, Any
from .base_attack import BaseAttack
from ..data.poisoned_dataset import PoisonedDataset


class BadNetsAttack(BaseAttack):
    """BadNets后门攻击"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trigger_config = self.attack_config['trigger']
        self.trigger_size = self.trigger_config['size']
        self.trigger_position = self.trigger_config.get('position', 'bottom_right')
        self.trigger_color = torch.tensor(self.trigger_config.get('color', [255, 255, 255])) / 255.0

    def add_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """添加方块触发器"""
        triggered_image = image.clone()

        if self.trigger_position == 'bottom_right':
            # 在右下角添加白色方块
            triggered_image[:, -self.trigger_size:, -self.trigger_size:] = self.trigger_color.view(-1, 1, 1)
        elif self.trigger_position == 'top_left':
            # 在左上角添加触发器
            triggered_image[:, :self.trigger_size, :self.trigger_size] = self.trigger_color.view(-1, 1, 1)
        elif self.trigger_position == 'center':
            # 在中心添加触发器
            h, w = image.shape[1], image.shape[2]
            center_h, center_w = h // 2, w // 2
            start_h = center_h - self.trigger_size // 2
            start_w = center_w - self.trigger_size // 2
            triggered_image[:, start_h:start_h + self.trigger_size,
            start_w:start_w + self.trigger_size] = self.trigger_color.view(-1, 1, 1)

        return triggered_image

    def create_poisoned_dataset(self, clean_dataset):
        """创建BadNets中毒数据集"""
        return PoisonedDataset(
            clean_dataset,
            self,
            poison_ratio=self.poison_ratio,
            target_label=self.target_label
        )
