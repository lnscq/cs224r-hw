"""
只读：基础策略接口
"""
import abc
import numpy as np

class BasePolicy(object, metaclass=abc.ABCMeta):
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """返回日志信息的字典。"""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError
