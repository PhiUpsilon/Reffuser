import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeVectorNetwork(nn.Module):
    """将历史和未来数据映射到单调递增的时间向量"""
    def __init__(self, input_dim, output_dim, batch_size=32, hidden_dim=256, max_time=1000):
        super(TimeVectorNetwork, self).__init__()
        
        self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)  # 第一层：输入t→隐藏层，权重>0
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 第二层：隐藏层→隐藏层，权重>0
        self.l3 = nn.Linear(hidden_dim, output_dim, bias=False)  # 第三层：隐藏层→输出层，权重>0
        self.l4 = nn.Linear(batch_size, 1, bias=False)  # 第四层：batch_size→输出层，权重>0
        # 激活函数
        self.act = nn.Sigmoid()  # 激活函数

        self.batch_size = batch_size
        
        # 最大时间步长
        self.max_time = max_time
        
        # # 初始化权重为正数
        # nn.init.constant_(self.l1.weight, 0.1)
        # nn.init.constant_(self.l2.weight, 0.1)
        # nn.init.constant_(self.l3.weight, 0.1)
        
    def forward(self, history_data, future_data):
        """
        输入:
            history_data: [B, L1, N, C]
            future_data: [B, L2, N, C]
        输出:
            time_features: [n_timesteps], 单调递增的整数时间向量
        """
        # 获取序列长度
        B, L1, N, C = history_data.shape
        _, L2, _, _ = future_data.shape

        if self.batch_size != B:
            self.batch_size = B
            self.l4 = nn.Linear(B, 1, bias=False)  # 更新第四层以适应新的batch_size
            self.l4.to(device)  # 确保第四层在正确的设备上
        
        # 计算时间步数
        n_timesteps = int(np.ceil(L2 / L1))
        
        # 在L维度(第1维)拼接历史和未来数据
        combined_data = torch.cat([history_data, future_data], dim=1)  # [B, L1+L2, N, C]
        
        # 调整维度以适应MLP输入
        combined_data = combined_data.view(B, -1)  # [B, (L1+L2)*N*C]
        
        # 通过MLP获取时间向量 [B, n_timesteps]
        time_features = self.l1(combined_data)  # [B, hidden_dim]
        # time_features = self.act(time_features) # [B, hidden_dim]
        time_features = self.l2(time_features) + time_features  # [B, hidden_dim]
        # time_features = self.act(time_features) # [B, hidden_dim]
        time_features = self.l3(time_features)  # [B, n_timesteps]

        time_features = time_features.view(n_timesteps, B)  # [n_timesteps, B]
        time_features = self.l4(time_features)              # [n_timesteps, 1]
        time_features = time_features.view(1, n_timesteps)  # [1, n_timesteps]

        time_features = self.act(time_features) # [1, n_timesteps]
        time_features = time_features.squeeze(0)
        
        # # 在batch维度取平均 [n_timesteps]
        # time_features = time_features.mean(dim=0)

        time_features, _ = torch.sort(time_features)
        
        # 缩放并取整为整数
        # time_features = (time_features * self.max_time).long()
        
        return time_features