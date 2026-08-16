from .ddpg import Actor, Critic
from baselines import PatchTST
from baselines import DLinear
from .transformer_actor import TransformerActor
from .gaussian_diffusion import GaussianDiffusion
from .schedule import DiffusionSchedule, SNRNetwork
from .time_schedule import TimeVectorNetwork

import os
import time
import numpy as np
import pandas as pd
import pickle
from easytorch.config import import_config, config_md5, get_ckpt_save_dir
from tqdm import trange
from collections import Counter
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sktime.performance_metrics.forecasting import \
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_continuous_predictions(GD, x_t, noise, x_tm1, device):
    """
    通过循环预测生成与未来数据长度(L2)相同的预测结果
    
    参数:
        GD: 模型对象
        x_t: 初始输入张量 [B, L1, N, C]
        t: 时间步张量
        noise: 噪声张量
        x_tm1
        device: 设备类型
        
    返回:
        pred: 预测结果张量 [B, L2, N, C]
    """
    # 获取初始历史数据的形状
    B, L1, N, C = x_t.shape
    _, L2, _, _ = x_tm1.shape
    n_timesteps = np.ceil(L2 / L1).astype(int)  # 计算总的时间步数
    
    # 初始化预测结果列表
    predictions = []
    
    # 当前历史数据
    current_history = x_t
    
    for current_t in reversed(range(n_timesteps)):
        # 预测
        current_t_tensor = make_timesteps(B, current_t, device)
        current_history = current_history[:, -min(L1, noise.shape[1]-L1*(n_timesteps-current_t-1)):, :, :]  # 保持历史数据的长度为L1
        current_noise = noise[:, L1*(n_timesteps-current_t-1):min(L1*(n_timesteps-current_t), noise.shape[1]), :, :]
        s_T = GD.predict_start_from_noise(current_history, current_t_tensor, current_noise)
        s_tp1_mean, s_tp1_variance, _ = GD.q_posterior(s_T, current_history, current_t_tensor)
        
        # 保存预测结果
        predictions.append(s_tp1_mean)
        
        current_history = s_tp1_mean
    
    # 拼接所有预测结果
    pred = torch.cat(predictions, dim=1)
    
    # 裁剪到L2长度
    if pred.shape[1] > L2:
        pred = pred[:, :L2, :, :]

    # 裁剪到目标长度L2
    return pred

# def estimate_beta(x_t, x_t_minus_1):
#     """
#     根据如下关系估计 beta_t：
#     x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon_{t-1}

#     假设 epsilon_{t-1} ~ N(0, 1)，从而最小化残差平方项估计 beta_t。
    
#     Args:
#         x_t: 当前时刻变量
#         x_t_minus_1: 上一时刻变量

#     Returns:
#         beta_t_hat: 估计的 beta_t
#     """
#     def objective(beta):
#         if beta <= 0 or beta >= 1:
#             return torch.inf
#         sqrt_alpha = torch.sqrt(1 - beta)
#         sqrt_beta = torch.sqrt(beta)
#         epsilon_hat = (x_t - sqrt_alpha * x_t_minus_1) / sqrt_beta
#         return torch.sum(epsilon_hat ** 2)

#     res = minimize_scalar(objective, bounds=(1e-6, 1 - 1e-6), method='bounded')
#     return res.x

# def cosine_beta_schedule(timesteps, current_t, s=0.008, dtype=torch.float32):
#     """
#     cosine schedule
#     as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
#     """
#     steps = timesteps + 1
#     x = np.linspace(0, steps, steps)
#     alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
#     return torch.tensor(betas_clipped, dtype=dtype)[current_t].detach()

# def estimate_beta(x_t, x_t_minus_1, n_timesteps=None, current_t=None):
#     """
#     根据如下关系估计 beta_t：
#     x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon_{t-1}

#     假设 epsilon_{t-1} ~ N(0, 1)，从而最小化残差平方项估计 beta_t。
    
#     Args:
#         x_t: 当前时刻变量
#         x_t_minus_1: 上一时刻变量

#     Returns:
#         beta_t_hat: 估计的 beta_t
#     """
#     # 确保输入是PyTorch张量
#     if not isinstance(x_t, torch.Tensor):
#         x_t = torch.tensor(x_t)
#     if not isinstance(x_t_minus_1, torch.Tensor):
#         x_t_minus_1 = torch.tensor(x_t_minus_1)
    
#     # 定义可优化的beta参数，初始值设为0.5
#     beta = torch.tensor(cosine_beta_schedule(n_timesteps, current_t).item(), requires_grad=True) if n_timesteps is not None and current_t is not None else torch.tensor(0.5, requires_grad=True)
#     optimizer = torch.optim.LBFGS([beta], lr=0.1)
    
#     # 记录上一次的损失值，用于收敛检测
#     prev_loss = float('inf')
    
#     # 定义目标函数
#     def closure():
#         optimizer.zero_grad()
#         if beta <= 0 or beta >= 1:
#             # 返回一个很大的值，模拟无穷大
#             return 1e10 * torch.ones(1, requires_grad=True)
#         sqrt_alpha = torch.sqrt(1 - beta)
#         sqrt_beta = torch.sqrt(beta)
#         epsilon_hat = (x_t - sqrt_alpha * x_t_minus_1) / sqrt_beta
#         loss = torch.sum(epsilon_hat ** 2)
#         loss.backward()
#         return loss
    
#     # 执行优化
#     for i in range(50):  # 最多50次迭代
#         loss = optimizer.step(closure)
        
#         # 收敛检测：检查损失变化是否小于阈值
#         if prev_loss != float('inf'):
#             loss_change = abs(prev_loss - loss.item())
#             if loss_change < 1e-6:  # 收敛阈值
#                 # print(f"Converged at iteration {i+1} with loss change: {loss_change:.8f}")
#                 break
#         prev_loss = loss.item()
        
#     # 把beta的值限制在有效范围内
#     with torch.no_grad():
#         beta.clamp_(0, 0.999)
    
#     return beta.detach()

# def estimate_betas_batch(x_t, x_tm1):
#     """
#     处理多个样本和时间步估计 beta_t
#     x_t, x_tm1: 形状为 (B, L1/L2, N, C)，其中
#         B: 批次大小，L: 时间步数，N: 空间维度，C: 通道数
#     返回:
#         beta_t_hat: 形状为 (B, N_TIMESTEPS) 的数组，表示每个样本每个时间步的估计 beta
#     """
#     # 获取初始历史数据的形状
#     B, L1, N, C = x_t.shape
#     _, L2, _, _ = x_tm1.shape
#     n_timesteps = np.ceil(L2 / L1).astype(int)  # 计算总的时间步数

#     beta_hats = torch.zeros((B, n_timesteps, N, C))

#     for b in range(B):
#         for n in range(N):
#             for c in range(C):
#                 current_x1 = x_t[b, :, n, c].unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 保持当前变量的形状为 (1, L1, 1, 1)
#                 for current_t in reversed(range(n_timesteps)):
#                     current_x1 = current_x1[:, -min(L1, L2-L1*(n_timesteps-current_t-1)):, :, :]
#                     current_x0 = x_tm1[b, L1*(n_timesteps-current_t-1):min(L1*(n_timesteps-current_t), L2), n, c].unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 保持当前变量的形状为 (1, L2, 1, 1)
#                     beta_hats[b, current_t, n, c] = estimate_beta(current_x1.reshape(-1), current_x0.reshape(-1), n_timesteps, current_t)
#                     current_x1 = current_x0  # 更新当前历史数据为上一时刻的变量
#     return beta_hats.mean(dim=(0, 2, 3)) # 返回每个时间步的平均 beta_t_hat

# def estimate_betas_batch(x_t, x_tm1):
#     """
#     处理多个样本和时间步估计 beta_t
#     x_t, x_tm1: 形状为 (B, L1/L2, N, C)，其中
#         B: 批次大小，L: 时间步数，N: 空间维度，C: 通道数
#     返回:
#         beta_t_hat: 形状为 (B, N_TIMESTEPS) 的数组，表示每个样本每个时间步的估计 beta
#     """
#     # 获取初始历史数据的形状
#     B, L1, N, C = x_t.shape
#     _, L2, _, _ = x_tm1.shape
#     n_timesteps = np.ceil(L2 / L1).astype(int)  # 计算总的时间步数

#     beta_hats = torch.zeros((B, n_timesteps))

#     for b in range(B):
#         current_x1 = x_t[b, :, :, :].unsqueeze(0)
#         for current_t in reversed(range(n_timesteps)):
#             current_x1 = current_x1[:, -min(L1, L2-L1*(n_timesteps-current_t-1)):, :, :]
#             current_x0 = x_tm1[b, L1*(n_timesteps-current_t-1):min(L1*(n_timesteps-current_t), L2), :, :].unsqueeze(0)
#             beta_hats[b, current_t] = estimate_beta(current_x1.reshape(-1), current_x0.reshape(-1), n_timesteps, current_t)
#             current_x1 = current_x0  # 更新当前历史数据为上一时刻的变量
#     return beta_hats.mean(dim=0) # 返回每个时间步的平均 beta_t_hat

# def get_timesteps(history_data, future_data, device):
#     """
#     获取时间步张量，形状为 (B, 1)
#     history_data: 历史数据张量 [B, L1, N, C]
#     future_data: 未来数据张量 [B, L2, N, C]
#     device: 设备类型
#     """
#     B, L1, N, C = history_data.shape
#     _, L2, _, _ = future_data.shape
#     n_timesteps = np.ceil(L2 / L1).astype(int)  # 计算总的时间步数

#     # # 创建时间步张量
#     # t = torch.zeros((B, L2), device=device)

#     # for current_t in reversed(range(n_timesteps)):
#     #     t[:, L1*(n_timesteps-current_t-1):min(L1*(n_timesteps-current_t), L2)] = n_timesteps-current_t-1

#     # # 归一化到[0,1]区间
#     # t = t / (n_timesteps - 1 + 1e-8)

#     t = torch.arange(L2, device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
#     t = t / (L2 - 1 + 1e-8)
    
#     return t.unsqueeze(-1), n_timesteps  # 添加一个维度，使其形状为 (B, L2, 1)

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

def r_squared(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def match_dim1(batch_obs, batch_next_obs):
    # 获取两个张量在维度1上的大小
    obs_dim1 = batch_obs.shape[1]
    next_obs_dim1 = batch_next_obs.shape[1]
    
    # 如果batch_next_obs在维度1上已经等于或大于batch_obs，则直接截断
    if next_obs_dim1 >= obs_dim1:
        return batch_next_obs[:, :obs_dim1, :, :]
    
    # 计算需要复制多少次才能达到或超过obs_dim1
    repeat_times = (obs_dim1 + next_obs_dim1 - 1) // next_obs_dim1  # 向上取整
    
    # 复制batch_next_obs
    repeated_next_obs = batch_next_obs.repeat(1, repeat_times, 1, 1)
    
    # 截断到所需大小
    matched_next_obs = repeated_next_obs[:, :obs_dim1, :, :]
    
    return matched_next_obs

##############
# DDPG Agent #
##############
class DDPGAgent:
    def __init__(self, obs_dim, act_dim, hidden_dim=256,
                 lr=3e-3, gamma=0.99, tau=0.005, **kwargs):
        # initialize the actor & target_actor
        # self.actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
        # self.target_actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
        self.actor = PatchTST(**kwargs).to(device)
        self.target_actor = PatchTST(**kwargs).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        # self.actor = TransformerActor(**kwargs).to(device)
        # self.target_actor = TransformerActor(**kwargs).to(device)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # initialize the critic
        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.target_critic = Critic(obs_dim, act_dim, hidden_dim).to(device)
        # self.critic = PatchTST(**kwargs).to(device)
        # self.target_critic = PatchTST(**kwargs).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-1)

        # parameters
        self.gamma  = gamma
        self.tau    = tau

        self.schedule = DiffusionSchedule().to(device)
        self.schedule_optimizer = torch.optim.Adam(self.schedule.parameters(), lr=lr)

        self.time_schedule = TimeVectorNetwork(input_dim=(kwargs['seq_len']+kwargs['pred_len'])*kwargs['num_nodes']*kwargs['num_features'], output_dim=int(np.ceil(kwargs['pred_len'] / kwargs['seq_len'])), hidden_dim=512, max_time=1000).to(device)
        self.time_schedule_optimizer = torch.optim.Adam(self.time_schedule.parameters(), lr=lr)

        # update the target network
        for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(param.data)
        for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(param.data)

    # def compute_continuous_loss(self, t, eps_pred, eps):
    #     """连续时间扩散损失（论文2-62）"""
    #     # b = x.shape[0]
    #     # t = torch.rand(b, 1).to(x.device)  # 均匀采样时间
    #     t.requires_grad = True
    #     # z_t, eps = self.add_noise(x, t)
    #     # gamma_t = self.schedule.get_gamma_embedding(t)
    #     # eps_pred = self.denoiser(z_t, gamma_t)
    #     # 计算γ'_η(t)（自动微分）
    #     gamma = self.schedule(t)
    #     gamma_grad = torch.autograd.grad(gamma.sum(), t, create_graph=True)[0]
    #     # 计算加权MSE
    #     mse_loss = F.mse_loss(eps_pred, eps, reduction='none').mean(dim=[2,3])
    #     loss = 0.5 * (gamma_grad.squeeze(-1) * mse_loss).mean()
    #     return loss
    
    # def compute_discrete_loss(self, t, eps_pred, eps, T=1000):
    #     """离散时间扩散损失（论文2-58）"""
    #     # b = x.shape[0]
    #     # t = torch.rand(b, 1).to(x.device)
    #     s = t - 1.0 / T
    #     s = s.clamp(0, 1 - 1.0 / T)
    #     # z_t, eps = self.add_noise(x, t)
    #     # gamma_t = self.schedule.get_gamma_embedding(t)
    #     # gamma_s = self.schedule.get_gamma_embedding(s)
    #     # eps_pred = self.denoiser(z_t, gamma_t)
    #     # 计算exp(γ(t)-γ(s))-1
    #     weight = (torch.exp(self.schedule(t) - self.schedule(s)) - 1)
    #     mse_loss = F.mse_loss(eps_pred, eps, reduction='none').mean(dim=[2,3])
    #     loss = (T/2) * (weight.squeeze(-1) * mse_loss).mean()
    #     return loss
    
    def select_action(self, obs, future_data, batch_seen, epoch, train):
        with torch.no_grad():
            action = self.actor(obs, future_data, batch_seen, epoch, train)
        return action

    def update(self, GD, history_data, future_data, batch_actions, batch_rewards, batch_seen, epoch, train, alpha_actor_loss, alpha_critic_similarity):
        # print(f"Update DDPG Agent at epoch {epoch}...")
        batch_obs = history_data
        batch_next_obs = future_data[:, :batch_obs.shape[1], :, :]
        batch_next_obs = match_dim1(batch_obs, batch_next_obs)

        with torch.no_grad():
            target_q = self.target_critic(
                GD, batch_next_obs, self.target_actor(batch_next_obs, future_data, batch_seen, epoch, train), 
                alpha_critic_similarity)  # (B,)
            target_q = batch_rewards + self.gamma * target_q  # (B,) 目标Q值
        current_q = self.critic(GD, batch_obs, batch_actions, alpha_critic_similarity)     # (B,) 预测的Q值

        # critic loss Q值损失
        q_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # actor loss ==> convert actor output to softmax weights 策略损失，即最大化Q值，即最小化负Q值
        # actor_loss = -self.critic(
        #     batch_obs, self.actor(batch_obs)).mean()
        # actor_loss = F.mse_loss(batch_next_obs, self.actor(batch_obs))
        # actor_loss = F.mse_loss(batch_next_obs, batch_actions)
        actions = self.actor(batch_obs, future_data, batch_seen, epoch, train)
        actor_loss_q = -self.critic(GD, batch_obs, actions, alpha_critic_similarity).mean()

        # noise = GD.calculate_noise(batch_next_obs, t, batch_obs)
        # actor_loss_log_mse = F.mse_loss(noise, actions)

        # actions = actions[:, :batch_obs.shape[1], :, :]
        # s_T = GD.predict_start_from_noise(batch_obs, t, actions)
        # s_tp1_mean, s_tp1_variance, _ = GD.q_posterior(s_T, batch_obs, t)
        # actor_loss_log_mse = F.mse_loss(s_tp1_mean, batch_next_obs)

        t_index_pred = self.time_schedule(history_data, future_data)
        sigma_squared = self.schedule.get_sigma_squared(t_index_pred.unsqueeze(-1)).squeeze(-1)
        sigma = torch.sqrt(sigma_squared)
        GD.update_betas(sigma)
        
        pred_data = get_continuous_predictions(GD, batch_obs, actions, future_data, device)
        # 计算预测数据与未来数据的均方误差
        # t, n_timesteps = get_timesteps(history_data, future_data, device)
        # actor_loss_log_mse = self.compute_continuous_loss(t, pred_data, future_data)
        # actor_loss_log_mse = self.compute_discrete_loss(t, pred_data, future_data, n_timesteps)

        actor_loss_log_mse = F.mse_loss(pred_data, future_data)

        # pred = GD.p_sample_loop(self.actor, batch_obs, batch_next_obs, batch_seen, epoch, train)
        # actor_loss_log_mse = F.mse_loss(pred, batch_next_obs)

        # pred = GD.p_sample_loop(self.actor, batch_obs, future_data, batch_seen, epoch, train)
        # actor_loss_log_mse = - r_squared(future_data, pred)  # R^2 loss

        # actor_loss_log_mse = torch.log(F.mse_loss(batch_next_obs, actions) + 1e-8)
        actor_loss = alpha_actor_loss * actor_loss_q + (1 - alpha_actor_loss) * actor_loss_log_mse
        self.actor_optimizer.zero_grad()
        self.schedule_optimizer.zero_grad()
        self.time_schedule_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.schedule_optimizer.step()
        self.time_schedule_optimizer.step()

        t_index_pred = self.time_schedule(history_data, future_data)
        sigma_squared = self.schedule.get_sigma_squared(t_index_pred.unsqueeze(-1)).squeeze(-1).detach()
        # sigma_squared = self.schedule.get_sigma_squared(torch.range(0, n_timesteps-1, device=device).unsqueeze(-1) / (future_data.shape[1] - 1 + 1e-8)).squeeze(-1).detach()
        sigma = torch.sqrt(sigma_squared)
        GD.update_betas(sigma)
        # GD.update_betas(torch.sqrt(self.schedule.get_sigma_squared(torch.range(0, n_timesteps-1, device=device).unsqueeze(-1) / (n_timesteps - 1 + 1e-8)).squeeze(-1).detach()).to(device))

        # Update the frozen target models 采用软更新
        for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q_loss': q_loss.item(),
            'pi_loss': actor_loss.item(),
            'current_q': current_q.mean().item(),
            'target_q': target_q.mean().item(),
            't_schedule': t_index_pred.detach(),
            'beta_schedule': sigma.detach()
        }, GD

class Env:
    def __init__(self):
        # 不用输入
        pass
    
    def reward_func(self, action, target):
        # action是Actor预测的结果，target是真实值
        # MAPE
        dim = [1,2,3]
        eps = 1e-8  # 避免除零
        absolute_error = torch.abs(action - target)
        absolute_percentage_error = absolute_error / (torch.abs(target) + eps)
        mape = absolute_percentage_error.mean(dim=dim, keepdim=True)

        # MAE
        mae = torch.abs(action - target).mean(dim=dim, keepdim=True)

        # RMSE
        mse = torch.square(action - target).mean(dim=dim, keepdim=True)
        rmse = torch.sqrt(mse)
        return mape, mae, rmse

# def get_batch_rewards(env, batch_actions, future_data, mode="negative_log", target_dim=4):
#     """
#     基于多指标（MAPE/MAE/RMSE）的复合奖励函数
    
#     参数说明：
#     - mape : 平均绝对百分比误差（百分比形式，如30表示30%）
#     - mae  : 平均绝对误差（原始单位）
#     - rmse : 均方根误差（原始单位）
#     - mode : 惩罚模式 ['negative', 'negative_exponential', 'negative_tanh']
#     - weights : 静态权重（仅在非adaptive模式生效），默认(0.6,0.3,0.1)
#     """

#     mape, mae, rmse = env.reward_func(batch_actions, future_data)
#     # ================== 指标标准化处理 ================== #
#     mape_scaled = mape     # MAPE是小数
#     mae_scaled = mae       # MAE量级约在0-2范围内
#     rmse_scaled = rmse     # RMSE量级约在0-2范围内
    
#     # ================== 自适应权重计算 ================== #
#     error_sum = mape_scaled + mae_scaled + rmse_scaled
#     # 防止零除错误（当所有误差为零时返回最大奖励）
#     if torch.all(error_sum == 0):
#         return torch.tensor(1.0).to(device)
    
#     # 动态权重计算（基于误差比例的非线性调整）
#     raw_weights = (
#         0.4 + 0.2 * torch.tanh(mape_scaled),
#         0.3 - 0.1 * (mae_scaled / error_sum),
#         0.3 * (1 - torch.tanh(rmse_scaled))
#     )
    
#     # ================== 权重归一化处理 ================== #
#     # 将三个权重分量堆叠（[3, B, ...]）具体几个维度看reward_func函数的返回值
#     stacked_weights = torch.stack(raw_weights)

#     # 沿第0维度求和（每个时间步独立计算总和）
#     total_weight = stacked_weights.sum(dim=0, keepdim=True)  # 形状变为[1, B, ...]

#     # 数值稳定性增强（防止零除）
#     epsilon = 1e-8
#     total_weight = total_weight + epsilon

#     # 逐元素归一化（广播机制自动对齐维度）
#     normalized_weights = stacked_weights / total_weight  # 形状保持[3, B, ...]
    
#     # ================== 非线性变换函数 ================== #
#     def transform(x, m):
#         """误差到惩罚值的转换"""
#         if m == "negative":
#             return -x
#         elif m == "negative_exponential":
#             return -torch.exp(x)
#         elif m in ["negative_log"]:
#             return -torch.tanh(x ** 3) + 0.5
#         else:
#             raise ValueError(f"Invalid mode: {m}")

#     # ================== 指标独立转换 ================== #
#     transform_mode = mode
#     r_mape = transform(mape_scaled, transform_mode)
#     r_mae = transform(mae_scaled, transform_mode)
#     r_rmse = transform(rmse_scaled, transform_mode)
    
#     # ================== 权重融合计算 ================== #
#     # 使用einsum实现高效矩阵运算（替代逐元素乘法）
#     rewards = torch.einsum('i...,i...->...', 
#                         torch.stack([r_mape, r_mae, r_rmse]), 
#                         normalized_weights)
#     # rewards = torch.clamp(rewards, -1.0, 1.0)
#     # # ================== 奖励广播到4维 ================== #
#     # current_dim = rewards.dim()
#     # # 计算需要补充的维度数
#     # missing_dims = target_dim - current_dim

#     # # 动态添加维度（从右往左补全）
#     # for _ in range(missing_dims):
#     #     rewards = rewards.unsqueeze(-1)  # 添加大小为1的维度
#     return rewards

def get_batch_rewards(env, GD, history_data, batch_actions, future_data, alpha_rewards_similarity, epoch, ckpt_save_dir):
    pred_data = get_continuous_predictions(GD, history_data, batch_actions, future_data, device)

    # rewards = - F.mse_loss(s_tp1_mean, future_data, reduction='none').mean(dim=[1, 2, 3], keepdim=True) + alpha_rewards_bias
    
    # 计算MAPE
    # rewards = - (torch.abs(s_tp1_mean - future_data) / torch.abs(future_data)).mean(dim=[1, 2, 3], keepdim=True) + alpha_rewards_bias

    # rewards = 1. / torch.abs(s_tp1_mean - future_data).mean(dim=[1, 2, 3], keepdim=True)

    # 傅里叶变换的维度
    dim = (1, 2, 3)

    obs = future_data
    act = pred_data
    
    # 输入形状均为 [B, L, N, C]
    B, L, N, C = obs.shape
    
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
    rewards = alpha_rewards_similarity * similarity_mag + (1 - alpha_rewards_similarity) * similarity_phase
    # if torch.max(similarity) > 100 or epoch % 100 == 0:
    #     # 将张量保存为一个字典
    #     data_dict = {
    #         'obs_fft': obs_fft,
    #         'act_fft': act_fft
    #     }

    #     # 保存字典
    #     torch.save(data_dict, os.path.join(ckpt_save_dir, f'data_dict_{epoch}.pt'))
    
    return torch.clamp(rewards.view(-1, 1, 1, 1), -1.0, 1.0)

class Reffuser(nn.Module):
    def __init__(self, **model_args):
        super(Reffuser, self).__init__()

        self.seq_len = model_args['seq_len']  # L
        self.pred_len = model_args['pred_len']
        self.num_nodes = model_args['num_nodes']
        self.num_features = model_args['num_features']
        self.alpha_actor_loss = model_args['alpha_actor_loss']
        self.alpha_critic_similarity = model_args['alpha_critic_similarity']
        self.alpha_rewards_similarity = model_args['alpha_rewards_similarity']
        # self.n_timesteps = model_args['n_timesteps']
        self.GD = GaussianDiffusion(n_timesteps=np.ceil(self.pred_len / self.seq_len).astype(int))
        self.cfg = import_config(os.path.relpath(model_args['cfg_path'], os.getcwd()))
        self.cfg['MD5'] = config_md5(self.cfg)

        # 创建环境
        self.env = Env()
    
        # initialize the DDPG agent
        self.agent = DDPGAgent(obs_dim=self.num_nodes*self.num_features, act_dim=self.num_nodes*self.num_features, hidden_dim=self.num_nodes*self.num_features, lr=1e-4, **model_args)

        # 初始化字典结构（Key为epoch，Value为对应指标列表）
        # self.betas_batch = []       # 存储epoch 0的每个betas
        self.epoch_pi_loss = {}     # 存储每个epoch的所有pi_loss
        self.epoch_q_loss = {}      # 存储每个epoch的所有q_loss
        self.epoch_current_q = {}   # 存储每个epoch的所有current_q
        self.epoch_target_q = {}    # 存储每个epoch的所有target_q
        self.epoch_t_schedule = {}
        self.epoch_beta_schedule = {}
        # # to save the best model
        # self.best_actor = Actor(obs_dim, act_dim, hidden_dim=100).to(device)
        # for param, target_param in zip(self.agent.actor.parameters(), self.best_actor.parameters()):
        #     target_param.data.copy_(param.data)
        self.save_flag = False  # 用于标记是否需要保存模型
    
    def save_agent(self, dir='./agent'):
        agent_state_dict = {
        'actor': self.agent.actor.state_dict(),
        'target_actor': self.agent.target_actor.state_dict(),
        'critic': self.agent.critic.state_dict(),
        'target_critic': self.agent.target_critic.state_dict(),
        'actor_optimizer': self.agent.actor_optimizer.state_dict(),
        'critic_optimizer': self.agent.critic_optimizer.state_dict(),
        'schedule': self.agent.schedule.state_dict(),
        'time_schedule': self.agent.time_schedule.state_dict()
        }
        file_path = os.path.join(dir, 'agent_checkpoint.pth')
        torch.save(agent_state_dict, file_path)
        print(f"Agent saved at {file_path}")

    def save_metrics(self, dir='./metrics'):
        
        # 保存为pickle文件（可自定义扩展名）
        metrics = {
            # 'betas': self.betas_batch,
            'pi_loss': self.epoch_pi_loss,
            'q_loss': self.epoch_q_loss,
            'current_q': self.epoch_current_q,
            'target_q': self.epoch_target_q,
            't_schedule': self.epoch_t_schedule,
            'beta_schedule': self.epoch_beta_schedule
        }
        
        file_path = f'{dir}/training_metrics.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        # 或用PyTorch的保存方式（推荐兼容性）
        # torch.save(metrics, f'{dir}/training_metrics.pth')

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        if train:
            # best_mape_loss = np.inf
            t1 = time.time()
            # if epoch == 1:
            #     # 初始化beta_t
            #     # self.betas_batch.append(estimate_betas_batch(history_data, future_data))
            #     # betas_tensor = torch.stack(self.betas_batch)
            #     # # self.GD.update_betas(betas_tensor.mean(dim=0).to(device))  # 更新beta_t
            #     # self.GD.update_betas(betas_tensor[0].to(device))  # 更新beta_t
            #     self.GD.update_betas(torch.tensor([0.2571, 0.6683, 0.9990], device=device))  # 更新beta_t
            batch_actions = self.agent.select_action(history_data, future_data, batch_seen, epoch, train)
            batch_rewards = get_batch_rewards(self.env, self.GD, history_data, batch_actions, future_data, self.alpha_rewards_similarity, epoch, ckpt_save_dir = get_ckpt_save_dir(self.cfg))
            
            info, self.GD = self.agent.update(self.GD, history_data, future_data, batch_actions, batch_rewards, batch_seen, epoch, train, self.alpha_actor_loss, self.alpha_critic_similarity)
            # 检查当前epoch是否已初始化字典条目
            if epoch not in self.epoch_pi_loss:
                self.epoch_pi_loss[epoch] = []
                self.epoch_q_loss[epoch] = []
                self.epoch_current_q[epoch] = []
                self.epoch_target_q[epoch] = []
                self.epoch_t_schedule[epoch] = []
                self.epoch_beta_schedule[epoch] = []
            
            # 更新字典（追加当前batch的指标）
            self.epoch_pi_loss[epoch].append(info['pi_loss'])
            self.epoch_q_loss[epoch].append(info['q_loss'])
            self.epoch_current_q[epoch].append(info['current_q'])
            self.epoch_target_q[epoch].append(info['target_q'])
            self.epoch_t_schedule[epoch].append(info['t_schedule'])
            self.epoch_beta_schedule[epoch].append(info['beta_schedule'])
            
            # 计算当前epoch的均值（动态更新）
            avg_pi = np.mean(self.epoch_pi_loss[epoch])
            avg_q_loss = np.mean(self.epoch_q_loss[epoch])
            avg_current_q = np.mean(self.epoch_current_q[epoch])
            avg_target_q = np.mean(self.epoch_target_q[epoch])

            # valid_mae_loss, valid_mape_loss, count_lst = evaluate_agent(self.agent, valid_states, valid_preds, valid_y)
            print(f'\n# Epoch {epoch} ({(time.time() - t1)/60:.2f} min): '
                #   f'valid_mae_loss: {valid_mae_loss:.3f}\t'
                #   f'valid_mape_loss: {valid_mape_loss*100:.3f}\t' 
                  f'pi_loss: {avg_pi:.5f}\t'
                  f'q_loss: {avg_q_loss:.5f}\t'
                  f'current_q: {avg_current_q:.5f}\t'
                  f'target_q: {avg_target_q:.5f}\n', end='', flush=True)
            if epoch == 100:
                self.save_flag = True
        elif train is False and self.save_flag:
            ckpt_save_dir = get_ckpt_save_dir(self.cfg)
            self.save_agent(dir=ckpt_save_dir)
            self.save_metrics(dir=ckpt_save_dir)
            self.save_flag = False

        # if valid_mape_loss < best_mape_loss:
        #     best_mape_loss = valid_mape_loss
        #     # save best model
        #     for param, target_param in zip(self.agent.actor.parameters(), self.best_actor.parameters()):
        #         target_param.data.copy_(param.data)
        
        # for param, target_param in zip( self.agent.actor.parameters(), self.best_actor.parameters()):
        #     param.data.copy_(target_param)
        # test_mae_loss, test_mape_loss, count_lst = evaluate_agent(
        #     self.agent, test_states, test_preds, test_y)
        # print(f'test_mae_loss: {test_mae_loss:.3f}\t'
        #     f'test_mape_loss: {test_mape_loss*100:.3f}')

        # return self.agent.actor(history_data, future_data, batch_seen, epoch, train)

        return get_continuous_predictions(self.GD, history_data, self.agent.actor(history_data, future_data, batch_seen, epoch, train), future_data, device)
    
        # print(f"Sample loop at epoch {epoch}...")

        # return self.GD.p_sample_loop(self.agent.actor, history_data, future_data, batch_seen, epoch, train)
        
        # return self.GD.p_sample_one_loop(self.agent.actor, history_data, future_data, batch_seen, epoch, train)

        # t = make_timesteps(history_data.shape[0], self.n_timesteps-1, device)
        # s_T = self.GD.predict_start_from_noise(history_data, t, self.agent.actor(history_data, future_data, batch_seen, epoch, train))
        # s_tp1_mean, s_tp1_variance, _ = self.GD.q_posterior(s_T, history_data, t)
        # return s_tp1_mean