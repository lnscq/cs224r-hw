"""
TO EDIT: 定义一个 PyTorch 策略作为智能体的 actor。

需要编辑的函数：
    1. get_action (line 96)
    2. forward (line 110)
    3. update (line 126)
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.base_policy import BasePolicy


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    定义一个用于监督学习的 MLP，将观测映射为动作

    属性
    ----------
    logits_na: nn.Sequential
        输出离散动作的神经网络
    mean_net: nn.Sequential
        输出连续动作均值的神经网络
    logstd: nn.Parameter
        用于学习动作标准差的独立参数

    方法
    -------
    get_action:
        调用 actor 的 forward 函数
    forward:
        在网络中执行可微的前向传播
    update:
        使用监督学习目标训练策略
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # 初始化环境相关变量（动作/观测维度、层数等）
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # NOTE: 该实现适用于连续动作空间。我们使用的所有环境都是连续动作空间。
        self.logits_na = None
        self.mean_net = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    ##################################

    def save(self, filepath):
        """
        :param filepath: 保存 MLP 的路径
        """
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        :param obs: 用于查询策略的观测
        :return:
            action: 从策略中采样的动作
        """
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO 返回策略给出的动作

        obs_tensor = ptu.from_numpy(observation)
        self.eval()
        with torch.no_grad():
            action_distribution = self.forward(obs_tensor)
            action = action_distribution.sample()

        return ptu.to_numpy(action)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        定义网络的前向传播

        :param observation: 用于查询策略的观测
        :return:
            action: 从策略中采样的动作
        """
        # TODO: 实现网络的前向传播。
        # 你可以返回任意对象，但它必须是可微的。例如可以返回 torch.FloatTensor，
        # 也可以返回更灵活的对象，如 `torch.distributions.Distribution`。
        # 由你决定！
        mu = self.mean_net(observation)
        std = torch.exp(self.logstd).expand_as(mu)
        action_distribution = distributions.Normal(mu, std)
        
        return action_distribution

    def update(self, observations, actions):
        """
        更新/训练策略

        :param observations: 用于查询策略的观测
        :param actions: 希望策略模仿的动作
        :return:
            dict: 'Training Loss'：监督学习损失
        """
        # TODO: 更新策略并返回损失。请记住更新策略需要反向传播梯度并执行优化器 step。
        obs_tensor = ptu.from_numpy(observations)
        actions_tensor = ptu.from_numpy(actions)
        self.train()
        action_distribution = self.forward(obs_tensor)

        log_probs = action_distribution.log_prob(actions_tensor).sum(dim=1)
        loss = -log_probs.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }
