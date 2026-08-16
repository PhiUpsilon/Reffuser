from .causal_cnn import CausalCNNEncoder
from .gaussian_diffusion import GaussianDiffusion

import copy
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##############
# DDPG Agent #
##############
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=100):
        super().__init__()
        # 定义一个CausalCNNEncoder，用于对输入的obs进行编码
        self.cnn_encoder = CausalCNNEncoder(depth=3,
                                            kernel_size=3,
                                            in_channels=obs_dim,
                                            channels=40,
                                            out_channels=hidden_dim,
                                            reduced_size=hidden_dim)
        # 定义一个神经网络，用于对编码后的数据进行处理
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, act_dim)
        self.relu = nn.ReLU()
    
    def forward(self, obs):
        B, L, N, C = obs.shape
        obs = obs.reshape(B, L, N * C).permute(0, 2, 1)
        # 对输入的obs进行编码
        x = F.relu(self.cnn_encoder(obs))
        # 对编码后的数据进行处理
        x = x.transpose(1, 2)
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        x = self.linear_3(x)
        x = x.transpose(1, 2)
        # 返回处理后的数据
        return x.permute(0, 2, 1).reshape(B, L, N, C)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=100):
        super().__init__()
        # # 定义CNN编码器
        # self.cnn_encoder = CausalCNNEncoder(depth=3,
        #                                     kernel_size=3,
        #                                     in_channels=obs_dim,
        #                                     channels=40,
        #                                     out_channels=hidden_dim,
        #                                     reduced_size=hidden_dim)
        # # 定义动作层
        # self.act_layer = nn.Linear(act_dim, hidden_dim)
        # 定义网络
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
        # self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    # def forward(self, obs, act):
    #     B, L, N, C = obs.shape
    #     obs = obs.reshape(B, L, N * C).permute(0, 2, 1)
    #     act = act.reshape(B, L, N * C).permute(0, 2, 1)
    #     # 将CNN编码器和动作层相加
    #     x = F.relu(self.cnn_encoder(obs) + self.act_layer(act.transpose(1, 2)).transpose(1, 2))
    #     # 将网络应用于结果
    #     x = x.transpose(1, 2)
    #     x = self.relu(self.linear_1(x))
    #     x = self.relu(self.linear_2(x))
    #     x = self.linear_3(x)
    #     x = self.tanh(x)
    #     x = x.transpose(1, 2)
    #     # 返回结果
    #     return x.permute(0, 2, 1).reshape(B, L, N, C).mean(dim=[1, 2, 3], keepdim=True)

    def forward(self, GD, obs, act, alpha_critic_similarity):
        # 输入形状为 [B, L, N, C]
        B, L1, N, C = obs.shape
        _, L2, _, _ = act.shape
        L = min(L1, L2)  # 取最小的长度

        obs = obs[:, -L:, :, :]
        act = act[:, :L, :, :]

        # s_T = GD.predict_start_from_noise(obs, n_timesteps, act)
        # s_tp1_mean, s_tp1_variance, _ = GD.q_posterior(s_T, obs, n_timesteps)

        # 傅里叶变换的维度
        dim = (1, 2, 3)

        # 对输入的obs和act进行编码
        obs = obs.reshape(B, L, N * C)
        act = act.reshape(B, L, N * C) # important
        obs = self.linear_1(obs).reshape(B, L, N, C)
        act = self.linear_1(act).reshape(B, L, N, C)
        
        # 傅里叶变换：对 L/N/C 三个维度进行三维傅里叶变换
        obs_fft = torch.fft.fftn(obs, dim=dim)  # 形状 [B, L, N, C]
        act_fft = torch.fft.fftn(act, dim=dim)
        
        # 计算幅度谱
        obs_mag = torch.abs(obs_fft)  # 取绝对值得到幅度
        act_mag = torch.abs(act_fft)
        
        # 相似度计算：幅度谱的余弦相似度
        obs_flatten = obs_mag.flatten(start_dim=1)  # 展平为 [B, L*N*C]
        act_flatten = act_mag.flatten(start_dim=1)
        similarity_mag = F.cosine_similarity(obs_flatten, act_flatten, dim=1)  # 形状 [B]

        obs_phase = torch.angle(obs_fft)  # 获取相位
        act_phase = torch.angle(act_fft)
        phase_diff = torch.cos(obs_phase - act_phase)  # 相位差余弦值
        similarity_phase = phase_diff.mean(dim=dim)  # 加权融合

        # 融合幅度谱和相位谱的相似度
        similarity = alpha_critic_similarity * similarity_mag + (1 - alpha_critic_similarity) * similarity_phase
        
        return torch.clamp(similarity.view(-1, 1, 1, 1), -1.0, 1.0)