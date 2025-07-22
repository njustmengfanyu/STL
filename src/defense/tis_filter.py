import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch.nn.functional as F


class TopologicalInvarianceSifting:
    """
    拓扑不变性筛选 (TIS) - 论文核心方法
    基于特征空间的拓扑结构识别清洁样本
    """

    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config['defense']['tis']
        self.threshold = self.config['threshold']
        self.similarity_metric = self.config.get('similarity_metric', 'cosine')
        self.consistency_weight = self.config.get('consistency_weight', 0.4)
        self.device = next(model.parameters()).device

    def extract_features(self, dataloader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """提取特征表示"""
        self.model.eval()
        features_list = []
        labels_list = []
        indices_list = []

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.to(self.device)

                # 提取编码器特征
                if hasattr(self.model, 'get_features'):
                    features = self.model.get_features(data)
                else:
                    features = self.model.encoder(data)
                    if len(features.shape) > 2:
                        features = features.view(features.size(0), -1)

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())

                # 记录全局索引
                batch_size = data.size(0)
                batch_indices = list(range(batch_idx * batch_size, batch_idx * batch_size + batch_size))
                indices_list.extend(batch_indices)

        features = np.vstack(features_list)
        labels = np.hstack(labels_list)

        return features, labels, indices_list

    def compute_topological_scores(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        计算拓扑不变性得分
        基于类内相似性和类间分离度
        """
        scores = np.zeros(len(features))
        unique_labels = np.unique(labels)

        print(f"计算拓扑不变性得分，共 {len(unique_labels)} 个类别")

        for label in unique_labels:
            # 获取当前类别的样本
            class_mask = labels == label
            class_features = features[class_mask]
            class_indices = np.where(class_mask)[0]

            if len(class_features) < 2:
                continue

            # 计算类内相似性矩阵
            if self.similarity_metric == 'cosine':
                similarity_matrix = cosine_similarity(class_features)
            else:
                # 欧氏距离相似性
                from sklearn.metrics.pairwise import euclidean_distances
                distance_matrix = euclidean_distances(class_features)
                similarity_matrix = 1 / (1 + distance_matrix)

            # 计算每个样本的类内一致性得分
            for i, global_idx in enumerate(class_indices):
                # 排除自身，计算与同类其他样本的平均相似性
                other_similarities = np.concatenate([
                    similarity_matrix[i, :i],
                    similarity_matrix[i, i + 1:]
                ])

                if len(other_similarities) > 0:
                    intra_class_score = np.mean(other_similarities)
                else:
                    intra_class_score = 0.0

                # 计算与其他类别的分离度
                inter_class_score = self._compute_inter_class_separation(
                    class_features[i], features, labels, label
                )

                # 综合得分
                scores[global_idx] = (1 - self.consistency_weight) * intra_class_score + \
                                     self.consistency_weight * inter_class_score

        return scores

    def _compute_inter_class_separation(self, sample_feature: np.ndarray,
                                        all_features: np.ndarray,
                                        all_labels: np.ndarray,
                                        current_label: int) -> float:
        """计算样本与其他类别的分离度"""
        other_class_mask = all_labels != current_label
        other_class_features = all_features[other_class_mask]

        if len(other_class_features) == 0:
            return 1.0

        # 计算到其他类别样本的平均距离
        if self.similarity_metric == 'cosine':
            similarities = cosine_similarity([sample_feature], other_class_features)[0]
            # 距离越大，分离度越好，所以取负值
            separation_score = 1 - np.mean(similarities)
        else:
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances([sample_feature], other_class_features)[0]
            separation_score = np.mean(distances) / (1 + np.mean(distances))

        return separation_score

    def detect_outliers_by_clustering(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """使用聚类检测异常样本"""
        outlier_scores = np.zeros(len(features))
        unique_labels = np.unique(labels)

        for label in unique_labels:
            class_mask = labels == label
            class_features = features[class_mask]
            class_indices = np.where(class_mask)[0]

            if len(class_features) < 5:  # 样本太少，跳过聚类
                continue

            # 对每个类别进行聚类
            n_clusters = min(3, len(class_features) // 2)
            if n_clusters < 2:
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(class_features)

            # 计算每个样本到其聚类中心的距离
            for i, global_idx in enumerate(class_indices):
                cluster_id = cluster_labels[i]
                center = kmeans.cluster_centers_[cluster_id]
                distance = np.linalg.norm(class_features[i] - center)

                # 距离越大，越可能是异常样本
                outlier_scores[global_idx] = distance

        # 标准化异常得分
        if np.std(outlier_scores) > 0:
            outlier_scores = (outlier_scores - np.mean(outlier_scores)) / np.std(outlier_scores)

        return outlier_scores

    def filter_clean_samples(self, dataloader: torch.utils.data.DataLoader) -> Tuple[List[int], Dict]:
        """
        筛选清洁样本
        返回清洁样本的全局索引和详细统计信息
        """
        print("开始拓扑不变性筛选...")

        # 提取特征
        features, labels, indices = self.extract_features(dataloader)

        # 计算拓扑不变性得分
        topo_scores = self.compute_topological_scores(features, labels)

        # 计算异常检测得分
        outlier_scores = self.detect_outliers_by_clustering(features, labels)

        # 综合得分 (拓扑得分高且异常得分低的样本更可能是清洁的)
        combined_scores = topo_scores - 0.3 * outlier_scores

        # 自适应阈值选择
        adaptive_threshold = self._adaptive_threshold_selection(combined_scores, labels)
        final_threshold = min(self.threshold, adaptive_threshold)

        # 筛选清洁样本
        clean_mask = combined_scores > final_threshold
        clean_indices = [indices[i] for i in range(len(indices)) if clean_mask[i]]

        # 统计信息
        stats = {
            'total_samples': len(features),
            'clean_samples': len(clean_indices),
            'clean_ratio': len(clean_indices) / len(features),
            'threshold_used': final_threshold,
            'avg_topo_score': np.mean(topo_scores),
            'avg_outlier_score': np.mean(outlier_scores),
            'avg_combined_score': np.mean(combined_scores),
            'score_std': np.std(combined_scores),
            'per_class_stats': self._compute_per_class_stats(combined_scores, labels, clean_mask)
        }

        self._print_filtering_results(stats)

        return clean_indices, stats

    def _adaptive_threshold_selection(self, scores: np.ndarray, labels: np.ndarray,
                                      target_ratio: float = 0.8) -> float:
        """自适应阈值选择"""
        # 基于得分分布选择阈值
        sorted_scores = np.sort(scores)[::-1]  # 降序排列
        threshold_idx = int(len(sorted_scores) * target_ratio)

        # 考虑类别平衡
        unique_labels = np.unique(labels)
        class_thresholds = []

        for label in unique_labels:
            class_mask = labels == label
            class_scores = scores[class_mask]
            if len(class_scores) > 0:
                class_threshold = np.percentile(class_scores, 80)
                class_thresholds.append(class_threshold)

        # 综合阈值
        distribution_threshold = sorted_scores[threshold_idx]
        class_balanced_threshold = np.mean(class_thresholds) if class_thresholds else distribution_threshold

        return (distribution_threshold + class_balanced_threshold) / 2

    def _compute_per_class_stats(self, scores: np.ndarray, labels: np.ndarray,
                                 clean_mask: np.ndarray) -> Dict:
        """计算每个类别的统计信息"""
        per_class_stats = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            class_mask = labels == label
            class_scores = scores[class_mask]
            class_clean_mask = clean_mask[class_mask]

            per_class_stats[int(label)] = {
                'total': int(np.sum(class_mask)),
                'clean': int(np.sum(class_clean_mask)),
                'clean_ratio': float(np.mean(class_clean_mask)),
                'avg_score': float(np.mean(class_scores)),
                'score_std': float(np.std(class_scores))
            }

        return per_class_stats

    def _print_filtering_results(self, stats: Dict):
        """打印筛选结果"""
        print(f"\n=== TIS筛选结果 ===")
        print(f"总样本数: {stats['total_samples']}")
        print(f"清洁样本数: {stats['clean_samples']}")
        print(f"清洁样本比例: {stats['clean_ratio']:.2%}")
        print(f"使用阈值: {stats['threshold_used']:.4f}")
        print(f"平均拓扑得分: {stats['avg_topo_score']:.4f}")
        print(f"平均异常得分: {stats['avg_outlier_score']:.4f}")
        print(f"平均综合得分: {stats['avg_combined_score']:.4f}")

        print(f"\n各类别筛选结果:")
        for label, class_stats in stats['per_class_stats'].items():
            print(f"  类别 {label}: {class_stats['clean']}/{class_stats['total']} "
                  f"({class_stats['clean_ratio']:.2%}) 平均得分: {class_stats['avg_score']:.4f}")
