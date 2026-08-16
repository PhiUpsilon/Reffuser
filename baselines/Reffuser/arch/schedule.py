import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionSchedule(nn.Module):
    """符合论文补充材料的噪声调度网络，强制γ_η(t)单调并约束SNR端点"""
    def __init__(self, snr_max=1000.0, snr_min=0.01, time_dim=64):
        super(DiffusionSchedule, self).__init__()
        # 1. 单调网络核心层（权重强制为正）
        self.l1 = nn.Linear(1, 1, bias=False)  # 第一层：输入t→隐藏层，权重>0
        self.l2 = nn.Linear(1, 2048, bias=False)  # 第二层：1024个神经元，权重>0
        self.l3 = nn.Linear(2048, 1, bias=False)  # 第三层：输出γ，权重>0
        self.act = nn.Sigmoid()  # 激活函数
        
        # 2. SNR端点可学习参数（γ0=-log(SNR_max), γ1=-log(SNR_min)）
        self.gamma0 = nn.Parameter(torch.tensor(-math.log(snr_max), dtype=torch.float32))
        self.gamma1 = nn.Parameter(torch.tensor(-math.log(snr_min), dtype=torch.float32))
        
        # 初始化权重为正数
        nn.init.constant_(self.l1.weight, 0.1)
        nn.init.constant_(self.l2.weight, 0.1)
        nn.init.constant_(self.l3.weight, 0.1)

        # # 3. 定制化正数初始化
        # self._init_weights()

    def _init_weights(self):
        """根据层特性定制正数初始化"""
        # l1层：输入1→输出1，使用小正数确保初始单调性
        nn.init.uniform_(self.l1.weight, a=0.01, b=0.1)  # 均匀分布在(0.01, 0.1)
        
        # l2层：输入1→输出2048，使用Xavier均匀初始化并取正数
        nn.init.xavier_uniform_(self.l2.weight)
        self.l2.weight.data = torch.abs(self.l2.weight.data)  # 确保正数
        
        # l3层：输入2048→输出1，使用Kaiming均匀初始化并取正数
        nn.init.kaiming_uniform_(self.l3.weight, a=math.sqrt(5))
        self.l3.weight.data = torch.abs(self.l3.weight.data)  # 确保正数

    def forward(self, t):
        """
        输入：时间t ∈ [0, 1] (shape: [B, L2, 1])
        输出：γ_η(t) (shape: [B, L2, 1])，满足单调递增
        """
        t = t.clamp(1e-5, 1 - 1e-5)  # 避免边界点数值问题
        # 计算单调网络输出 γ_hat = l1(t) + l3(σ(l2(l1(t)))
        h = self.l1(t)  # [B, L2, 1]
        h = self.l2(h)  # [B, L2, 1024]
        h = self.act(h)
        h = self.l3(h)  # [B, L2, 1]
        gamma_hat = h + self.l1(t)  # 合并第一层直接映射
        
        # 3. 端点约束后处理（将γ_hat映射到[gamma0, gamma1]）
        gamma0 = self.gamma0
        gamma1 = self.gamma1
        gamma_hat_min = gamma_hat.min()
        gamma_hat_max = gamma_hat.max()
        gamma = gamma0 + (gamma1 - gamma0) * (gamma_hat - gamma_hat_min) / (gamma_hat_max - gamma_hat_min + 1e-8)
        gamma, _ = torch.sort(gamma, dim=-2)  # 确保单调递增
        return gamma

    def get_sigma_squared(self, t):
        """计算σ_t² = sigmoid(γ_η(t))"""
        gamma = self(t)
        return torch.sigmoid(gamma)

    def get_alpha_squared(self, t):
        """计算α_t² = sigmoid(-γ_η(t))"""
        gamma = self(t)
        return torch.sigmoid(-gamma)

    def get_snr(self, t):
        """计算SNR(t) = exp(-γ_η(t))"""
        gamma = self(t)
        return torch.exp(-gamma)

    def get_gamma_embedding(self, t):
        """获取用于时间嵌入的γ(t)（供去噪模型使用）"""
        return self(t)
    
class PositiveLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.softplus = nn.Softplus()

    def forward(self, input: torch.Tensor):  # type: ignore
        return input @ self.softplus(self.weight) + self.softplus(self.bias)


class SNRNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_min = nn.Parameter(torch.tensor(-10.0))
        self.gamma_max = nn.Parameter(torch.tensor(20.0))

        self.softplus = nn.Softplus()

    def forward(self, t: torch.Tensor):  # type: ignore

        # Add start and endpoints 0 and 1.
        t = torch.cat([torch.zeros_like(t[0:1], device=t.device), torch.ones_like(t[0:1], device=t.device), t], dim=0)
        l1 = self.l1(t)
        l2 = torch.sigmoid(self.l2(l1))
        l3 = l1 + self.l3(l2)

        s0, s1, sched = l3[0], l3[1], l3[2:]

        norm_nlogsnr = (sched - s0) / (s1 - s0)

        nlogsnr = self.gamma_min + self.softplus(self.gamma_max) * norm_nlogsnr
        return nlogsnr
    
    def get_sigma_squared(self, t: torch.Tensor) -> torch.Tensor:
        """计算σ_t² = sigmoid(γ_η(t))"""
        nlogsnr = self(t)
        return torch.sigmoid(nlogsnr)