#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:21 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : ray_worker.py
# @Software: PyCharm

"""
优化的Ray工作器
"""

import ray
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import traceback
import uuid

from models.base import ModelFactory
from models.simple_models import SimpleMLPModel
from models.vision import ImageClassificationModel, ObjectDetectionModel
from core.logger import get_logger
from core.exceptions import ModelLoadError, TaskExecutionError

logger = get_logger(__name__)


@ray.remote
class AsyncModelWorker:
    """异步模型工作器"""

    def __init__(self, worker_id: str = None):
        self.worker_id = worker_id or str(uuid.uuid4())
        self.models = {}  # 缓存已加载的模型
        self.current_task = None
        self.status = "idle"
        logger.info(f"工作器初始化: {self.worker_id}")

    async def load_model(self, model_name: str, model_type: str, parameters: Dict[str, Any]) -> bool:
        """异步加载模型"""
        try:
            if model_name not in self.models:
                logger.info(f"[{self.worker_id}] 加载模型: {model_name}")
                if model_type == "regression":
                    ModelFactory.register("regression", SimpleMLPModel)

                if model_type == "image_classification":
                    ModelFactory.register("image_classification", ImageClassificationModel)

                if model_type == "object_detection":
                    ModelFactory.register("object_detection", ObjectDetectionModel)
                # 创建模型实例
                model = ModelFactory.create(model_type, parameters)

                # 加载模型
                model.load()

                # 缓存模型
                self.models[model_name] = model
                logger.info(f"[{self.worker_id}] 模型加载完成: {model_name}")
            else:
                logger.info(f"[{self.worker_id}] 模型已缓存: {model_name}")

            return True

        except Exception as e:
            error_msg = f"模型加载失败: {model_name}, 错误: {str(e)}"
            logger.error(f"[{self.worker_id}] {error_msg}")
            logger.error(traceback.format_exc())
            raise ModelLoadError(error_msg)

    async def execute_task(self, task_id: str, model_name: str, input_data: Dict[str, Any],
                           task_type: str) -> Dict[str, Any]:
        """异步执行任务"""
        self.current_task = task_id
        self.status = "running"
        start_time = datetime.now()

        try:
            logger.info(f"[{self.worker_id}] 执行任务: {task_id}, 模型: {model_name}")

            # 检查模型是否已加载
            if model_name not in self.models:
                raise TaskExecutionError(f"模型未加载: {model_name}")

            model = self.models[model_name]

            # 执行预测
            if task_type == "inference":
                result = await self._run_inference(model, input_data, task_id)
            elif task_type == "classification":
                result = await self._run_classification(model, input_data, task_id)
            elif task_type == "detection":
                result = await self._run_detection(model, input_data, task_id)
            else:
                raise TaskExecutionError(f"不支持的任务类型: {task_type}")

            # 计算执行时间
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # 构建结果
            final_result = {
                "task_id": task_id,
                "model_name": model_name,
                "task_type": task_type,
                "result": result,
                "execution_time": execution_time,
                "worker_id": self.worker_id,
                "completed_at": end_time.isoformat(),
                "status": "success"
            }

            logger.info(f"[{self.worker_id}] 任务执行完成: {task_id}, 耗时: {execution_time:.2f}秒")
            return final_result

        except Exception as e:
            error_msg = f"任务执行失败: {task_id}, 错误: {str(e)}"
            logger.error(f"[{self.worker_id}] {error_msg}")
            logger.error(traceback.format_exc())

            return {
                "task_id": task_id,
                "error": error_msg,
                "worker_id": self.worker_id,
                "failed_at": datetime.now().isoformat(),
                "status": "failed"
            }
        finally:
            self.current_task = None
            self.status = "idle"

    async def _run_inference(self, model, input_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """运行推理任务"""
        logger.info(f"[{self.worker_id}] 开始推理: {task_id}")

        # 执行推理
        predictions = model.predict(input_data.get("data", input_data))

        return {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            "input_shape": getattr(predictions, 'shape', None)
        }

    async def _run_classification(self, model, input_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """运行分类任务"""
        logger.info(f"[{self.worker_id}] 开始分类: {task_id}")

        result = model.predict(input_data.get("data", input_data))

        # 处理不同类型的分类结果
        if isinstance(result, dict):
            return result
        else:
            return {"predictions": result.tolist() if hasattr(result, 'tolist') else result}

    async def _run_detection(self, model, input_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """运行检测任务"""
        logger.info(f"[{self.worker_id}] 开始检测: {task_id}")

        detections = model.predict(input_data)
        return detections

    async def get_status(self) -> Dict[str, Any]:
        """获取工作器状态"""
        return {
            "worker_id": self.worker_id,
            "status": self.status,
            "current_task": self.current_task,
            "loaded_models": list(self.models.keys()),
            "model_count": len(self.models),
            "timestamp": datetime.now().isoformat()
        }

    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        try:
            if model_name in self.models:
                del self.models[model_name]
                logger.info(f"[{self.worker_id}] 模型已卸载: {model_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"[{self.worker_id}] 模型卸载失败: {model_name}, 错误: {str(e)}")
            return False

    async def clear_cache(self) -> bool:
        """清空模型缓存"""
        try:
            self.models.clear()
            logger.info(f"[{self.worker_id}] 模型缓存已清空")
            return True
        except Exception as e:
            logger.error(f"[{self.worker_id}] 清空缓存失败: {str(e)}")
            return False
