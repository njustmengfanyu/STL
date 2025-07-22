import torch
import torch.nn as nn
from typing import Dict, Any
import src.models.masked_linear as MaskedLinear


class ChannelFilter:
    """
    编码器通道过滤器 - 识别和过滤可信神经元
    """

    def __init__(self, model: nn.Module, filter_ratio: float = 0.7):
        self.model = model
        self.filter_ratio = filter_ratio
        self.device = next(model.parameters()).device

    def analyze_channel_importance(self, clean_dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """分析通道重要性"""
        self.model.eval()
        channel_activations = {}

        # 注册钩子函数收集激活
        hooks = []

        def hook_fn(name):
            def hook(module, input, output):
                if name not in channel_activations:
                    channel_activations[name] = []
                # 计算通道级别的平均激活
                if len(output.shape) == 4:  # Conv层
                    activation = torch.mean(output, dim=(2, 3))  # 空间维度平均
                else:  # Linear层
                    activation = output
                channel_activations[name].append(activation.detach())

            return hook

        # 为关键层注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # 前向传播收集激活
        with torch.no_grad():
            for data, _ in clean_dataloader:
                data = data.to(self.device)
                _ = self.model(data)

        # 清理钩子
        for hook in hooks:
            hook.remove()

        # 计算通道重要性得分
        importance_scores = {}
        for name, activations in channel_activations.items():
            stacked_activations = torch.cat(activations, dim=0)
            # 使用方差作为重要性指标
            importance_scores[name] = torch.var(stacked_activations, dim=0)

        return importance_scores

    def apply_channel_filtering(self, clean_dataloader: torch.utils.data.DataLoader):
        """应用通道过滤"""
        importance_scores = self.analyze_channel_importance(clean_dataloader)

        for name, module in self.model.named_modules():
            if name in importance_scores and isinstance(module, MaskedLinear.MaskedLinear):
                scores = importance_scores[name]
                # 选择top-k重要通道
                k = int(len(scores) * self.filter_ratio)
                _, top_indices = torch.topk(scores, k)

                # 更新掩码
                with torch.no_grad():
                    module.mask.zero_()
                    module.mask[top_indices] = 1.0

                print(f"层 {name}: 保留 {k}/{len(scores)} 个通道")
