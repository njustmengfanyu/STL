import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """
    可训练掩码的线性层，用于信任通道过滤
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 原始权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

        # 可训练掩码
        self.mask = nn.Parameter(torch.ones(out_features, in_features))

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 将掩码限制在[0,1]范围内
        mask_clamped = torch.clamp(self.mask, 0, 1)
        masked_weight = self.weight * mask_clamped
        return F.linear(input, masked_weight, self.bias)

    def get_active_channels(self, threshold: float = 0.5) -> torch.Tensor:
        """获取激活的通道"""
        return (self.mask > threshold).float()
