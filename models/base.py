#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:17 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : base.py
# @Software: PyCharm

"""
模型基类和工厂
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn
from core.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """模型基类"""

    def __init__(self, model_type: str, parameters: Dict[str, Any]):
        self.model_type = model_type
        self.parameters = parameters
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load(self):
        """加载模型"""
        pass

    @abstractmethod
    def predict(self, input_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """预测"""
        pass

    def preprocess(self, input_data: Any) -> Union[np.ndarray, torch.Tensor]:
        """预处理输入数据"""
        if isinstance(input_data, list):
            return np.array(input_data)
        return input_data

    def postprocess(self, output: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """后处理输出数据"""
        if isinstance(output, torch.Tensor):
            return output.detach().cpu().numpy()
        return output

    def get_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": self.model_type,
            "parameters": self.parameters,
            "is_loaded": self.is_loaded
        }


class ModelFactory:
    """模型工厂"""

    _registry = {}

    @classmethod
    def register(cls, model_type: str, model_class):
        """注册模型类"""
        cls._registry[model_type] = model_class
        logger.info(f"注册模型类: {model_type}")

    @classmethod
    def create(cls, model_type: str, parameters: Dict[str, Any]) -> BaseModel:
        """创建模型实例"""
        if model_type not in cls._registry:
            raise ValueError(f"未知的模型类型: {model_type}")

        model_class = cls._registry[model_type]
        return model_class(model_type, parameters)

    @classmethod
    def list_types(cls) -> list:
        """列出所有支持的模型类型"""
        return list(cls._registry.keys())
