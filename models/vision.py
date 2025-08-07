#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:18 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : vision.py.py
# @Software: PyCharm

"""
计算机视觉模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Any, Union, List
import io
import base64

from models.base import BaseModel, ModelFactory
from core.logger import get_logger

logger = get_logger(__name__)


class ImageClassificationModel(BaseModel):
    """图像分类模型"""

    def __init__(self, model_type: str, parameters: Dict[str, Any]):
        super().__init__(model_type, parameters)
        self.num_classes = parameters.get("num_classes", 10)
        self.input_size = parameters.get("input_size", 224)
        self.channels = parameters.get("channels", 3)

        # 图像预处理管道
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load(self):
        """加载模型"""
        try:
            # 创建简单的CNN模型
            self.model = SimpleCNN(
                num_classes=self.num_classes,
                input_size=self.input_size,
                channels=self.channels
            )

            # 如果有预训练权重路径，加载权重
            weights_path = self.parameters.get("weights_path")
            if weights_path:
                self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                logger.info(f"加载预训练权重: {weights_path}")

            self.model.eval()
            self.is_loaded = True
            logger.info(f"图像分类模型加载成功: {self.num_classes} 类")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def preprocess_image(self, image_data: Union[str, bytes, Image.Image]) -> torch.Tensor:
        """预处理图像数据"""
        try:
            if isinstance(image_data, str):
                # Base64编码的图像
                if image_data.startswith('data:image'):
                    # 移除data URL前缀
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                raise ValueError(f"不支持的图像数据类型: {type(image_data)}")

            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 应用变换
            tensor = self.transform(image)
            return tensor.unsqueeze(0)  # 添加batch维度

        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            raise

    def predict(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """预测图像类别"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        try:
            # 处理单个图像或批量图像
            if isinstance(input_data, dict):
                input_data = [input_data]

            batch_tensors = []
            for item in input_data:
                if 'image' in item:
                    tensor = self.preprocess_image(item['image'])
                    batch_tensors.append(tensor)
                else:
                    raise ValueError("输入数据中缺少'image'字段")

            # 合并为批次
            batch_tensor = torch.cat(batch_tensors, dim=0)

            # 推理
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

            return {
                'predictions': predictions.numpy().tolist(),
                'probabilities': probabilities.numpy().tolist(),
                'batch_size': len(input_data)
            }

        except Exception as e:
            logger.error(f"图像预测失败: {str(e)}")
            raise


class ObjectDetectionModel(BaseModel):
    """目标检测模型"""

    def __init__(self, model_type: str, parameters: Dict[str, Any]):
        super().__init__(model_type, parameters)
        self.num_classes = parameters.get("num_classes", 80)
        self.input_size = parameters.get("input_size", 416)
        self.confidence_threshold = parameters.get("confidence_threshold", 0.5)

        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
        ])

    def load(self):
        """加载目标检测模型"""
        try:
            # 这里可以加载YOLO或其他检测模型
            # 为了演示，使用简单的分类模型
            self.model = SimpleCNN(
                num_classes=self.num_classes,
                input_size=self.input_size
            )

            self.model.eval()
            self.is_loaded = True
            logger.info("目标检测模型加载成功")

        except Exception as e:
            logger.error(f"目标检测模型加载失败: {str(e)}")
            raise

    def predict(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """检测图像中的目标"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        # 简化的检测逻辑
        # 实际应用中这里会是完整的目标检测推理
        try:
            if isinstance(input_data, dict):
                input_data = [input_data]

            results = []
            for item in input_data:
                # 模拟检测结果
                detections = {
                    'boxes': [[100, 100, 200, 200], [300, 150, 400, 250]],
                    'scores': [0.9, 0.8],
                    'labels': [1, 2],
                    'num_detections': 2
                }
                results.append(detections)

            return {
                'detections': results,
                'batch_size': len(input_data)
            }

        except Exception as e:
            logger.error(f"目标检测失败: {str(e)}")
            raise


class SimpleCNN(nn.Module):
    """简单的CNN模型"""

    def __init__(self, num_classes: int = 10, input_size: int = 224, channels: int = 3):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 注册模型类
ModelFactory.register("image_classification", ImageClassificationModel)
ModelFactory.register("object_detection", ObjectDetectionModel)
