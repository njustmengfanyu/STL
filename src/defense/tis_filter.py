import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity


class TopologicalInvarianceSifting:
    """
    拓扑不变性筛选 (TIS) - 用于识别清洁样本
    """

    def __init__(self, model: nn.Module, threshold: float = 0):
        self.model = model
        self.threshold = threshold
        self.device = next(model.parameters()).device

    def extract_features(self, dataloader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """提取特征表示"""
        self.model.eval()
        features_list = []
        labels_list = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                # 提取编码器特征
                features = self.model.encoder(data)
                features = features.view(features.size(0), -1)

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())

        return np.vstack(features_list), np.hstack(labels_list)

    def compute_topological_scores(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """计算拓扑不变性得分"""
        scores = np.zeros(len(features))
        unique_labels = np.unique(labels)

        for label in unique_labels:
            # 获取同类样本
            class_mask = labels == label
            class_features = features[class_mask]

            if len(class_features) < 2:
                continue

            # 计算类内相似性
            similarity_matrix = cosine_similarity(class_features)

            # 计算每个样本的平均相似性（排除自身）
            for i, idx in enumerate(np.where(class_mask)[0]):
                sim_scores = similarity_matrix[i]
                # 排除自身相似性
                other_similarities = np.concatenate([sim_scores[:i], sim_scores[i + 1:]])
                scores[idx] = np.mean(other_similarities)

        return scores

    def filter_clean_samples(self, dataloader: torch.utils.data.DataLoader) -> List[int]:
        """筛选清洁样本索引"""
        features, labels = self.extract_features(dataloader)
        scores = self.compute_topological_scores(features, labels)

        # 基于阈值筛选
        clean_indices = np.where(scores > self.threshold)[0].tolist()

        print(f"TIS筛选结果: {len(clean_indices)}/{len(scores)} 样本被识别为清洁样本")
        return clean_indices
