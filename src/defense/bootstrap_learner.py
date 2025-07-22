import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple


class BootstrapLearner:
    """
    引导学习器 - 基于可信数据逐步扩展训练
    """

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = config['training']['device']
        self.criterion = nn.CrossEntropyLoss()

    def create_seed_dataset(self, dataset, clean_indices: List[int]) -> DataLoader:
        """创建种子数据集"""
        seed_subset = Subset(dataset, clean_indices)
        return DataLoader(
            seed_subset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )

    def bootstrap_training(self,
                           seed_dataloader: DataLoader,
                           full_dataset,
                           num_iterations: int = 5) -> None:
        """引导训练过程"""
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )

        current_clean_indices = list(range(len(seed_dataloader.dataset)))

        for iteration in range(num_iterations):
            print(f"\n=== 引导训练迭代 {iteration + 1}/{num_iterations} ===")

            # 当前迭代的训练
            self._train_epoch(seed_dataloader, optimizer)

            # 扩展可信数据集
            if iteration < num_iterations - 1:
                new_clean_indices = self._expand_trusted_data(
                    full_dataset, current_clean_indices
                )
                current_clean_indices.extend(new_clean_indices)

                # 更新数据加载器
                expanded_subset = Subset(full_dataset, current_clean_indices)
                seed_dataloader = DataLoader(
                    expanded_subset,
                    batch_size=self.config['training']['batch_size'],
                    shuffle=True,
                    num_workers=4
                )

                print(f"扩展后的可信数据集大小: {len(current_clean_indices)}")

    def _train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                print(f'批次 {batch_idx}, 损失: {loss.item():.6f}')

        accuracy = 100. * correct / total
        print(f'训练准确率: {accuracy:.2f}%, 平均损失: {total_loss / len(dataloader):.6f}')

    def _expand_trusted_data(self, full_dataset, current_indices: List[int]) -> List[int]:
        """扩展可信数据"""
        # 简化实现：基于模型置信度选择新样本
        self.model.eval()
        new_indices = []

        # 获取未使用的样本索引
        all_indices = set(range(len(full_dataset)))
        used_indices = set(current_indices)
        unused_indices = list(all_indices - used_indices)

        if not unused_indices:
            return []

        # 创建未使用样本的数据加载器
        unused_subset = Subset(full_dataset, unused_indices)
        unused_loader = DataLoader(unused_subset, batch_size=64, shuffle=False)

        confidences = []
        with torch.no_grad():
            for data, _ in unused_loader:
                data = data.to(self.device)
                output = self.model(data)
                prob = torch.softmax(output, dim=1)
                max_prob, _ = torch.max(prob, dim=1)
                confidences.extend(max_prob.cpu().numpy())

        # 选择高置信度样本
        confidence_threshold = 0.9
        for i, conf in enumerate(confidences):
            if conf > confidence_threshold:
                new_indices.append(unused_indices[i])

        return new_indices[:min(len(new_indices), 100)]  # 限制扩展数量
