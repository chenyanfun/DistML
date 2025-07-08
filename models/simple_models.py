#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:18 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : simple_models.py
# @Software: PyCharm

"""
简单模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Union

from models.base import BaseModel, ModelFactory
from core.logger import get_logger

logger = get_logger(__name__)


class SimpleMLPModel(BaseModel):
    """简单的多层感知机模型"""

    def __init__(self, model_type: str, parameters: Dict[str, Any]):
        super().__init__(model_type, parameters)
        self.input_size = parameters.get("input_size", 10)
        self.hidden_sizes = parameters.get("hidden_sizes", [64, 32])
        self.output_size = parameters.get("output_size", 1)
        self.dropout_rate = parameters.get("dropout_rate", 0.2)

    def load(self):
        """加载MLP模型"""
        try:
            self.model = MLPNetwork(
                input_size=self.input_size,
                hidden_sizes=self.hidden_sizes,
                output_size=self.output_size,
                dropout_rate=self.dropout_rate
            )
            self.model.eval()
            self.is_loaded = True
            logger.info(f"MLP模型加载成功: {self.input_size} -> {self.hidden_sizes} -> {self.output_size}")
        except Exception as e:
            logger.error(f"MLP模型加载失败: {str(e)}")
            raise

    def predict(self, input_data: Union[np.ndarray, list]) -> np.ndarray:
        """MLP预测"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        try:
            # 预处理
            x = self.preprocess(input_data)
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)

            if x.dim() == 1:
                x = x.unsqueeze(0)

            # 推理
            with torch.no_grad():
                output = self.model(x)

            return self.postprocess(output)

        except Exception as e:
            logger.error(f"MLP预测失败: {str(e)}")
            raise


class SimpleClassificationModel(BaseModel):
    """简单分类模型"""

    def __init__(self, model_type: str, parameters: Dict[str, Any]):
        super().__init__(model_type, parameters)
        self.input_size = parameters.get("input_size", 10)
        self.num_classes = parameters.get("num_classes", 2)
        self.hidden_sizes = parameters.get("hidden_sizes", [64, 32])

    def load(self):
        """加载分类模型"""
        try:
            self.model = MLPNetwork(
                input_size=self.input_size,
                hidden_sizes=self.hidden_sizes,
                output_size=self.num_classes,
                dropout_rate=0.2
            )
            self.model.eval()
            self.is_loaded = True
            logger.info(f"分类模型加载成功: {self.num_classes} 类")
        except Exception as e:
            logger.error(f"分类模型加载失败: {str(e)}")
            raise

    def predict(self, input_data: Union[np.ndarray, list]) -> Dict[str, np.ndarray]:
        """分类预测"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        try:
            # 预处理
            x = self.preprocess(input_data)
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)

            if x.dim() == 1:
                x = x.unsqueeze(0)

            # 推理
            with torch.no_grad():
                logits = self.model(x)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

            return {
                'predictions': self.postprocess(predictions),
                'probabilities': self.postprocess(probabilities),
                'logits': self.postprocess(logits)
            }

        except Exception as e:
            logger.error(f"分类预测失败: {str(e)}")
            raise


class MLPNetwork(nn.Module):
    """MLP网络实现"""

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout_rate: float = 0.2):
        super(MLPNetwork, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# 注册模型类
ModelFactory.register("mlp", SimpleMLPModel)
ModelFactory.register("classification", SimpleClassificationModel)
ModelFactory.register("regression", SimpleMLPModel)  # 回归使用MLP
