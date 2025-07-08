#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:11 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : models.py
# @Software: PyCharm

"""
API数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime


class ModelRegistration(BaseModel):
    """模型注册请求"""
    name: str = Field(..., description="模型名称")
    description: str = Field(..., description="模型描述")
    model_type: str = Field(..., description="模型类型")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="模型参数")
    version: str = Field(default="1.0.0", description="模型版本")


class TaskSubmission(BaseModel):
    """任务提交请求"""
    model_name: str = Field(..., description="模型名称")
    input_data: Dict[str, Any] = Field(..., description="输入数据")
    task_type: str = Field(default="inference", description="任务类型")
    priority: int = Field(default=1, ge=1, le=10, description="任务优先级")


class ImageTaskSubmission(BaseModel):
    """图像任务提交请求"""
    model_name: str = Field(..., description="模型名称")
    images: List[str] = Field(..., description="Base64编码的图像列表")
    task_type: str = Field(default="classification", description="任务类型")
    priority: int = Field(default=1, ge=1, le=10, description="任务优先级")

    def to_task_submission(self) -> TaskSubmission:
        """转换为通用任务提交格式"""
        return TaskSubmission(
            model_name=self.model_name,
            input_data={"images": self.images},
            task_type=self.task_type,
            priority=self.priority
        )


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="响应消息")


class TaskResult(BaseModel):
    """任务结果"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    error: Optional[str] = Field(None, description="错误信息")
    progress: float = Field(default=0.0, description="任务进度")
    worker_id: Optional[str] = Field(None, description="工作器ID")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    started_at: Optional[str] = Field(None, description="开始时间")
    completed_at: Optional[str] = Field(None, description="完成时间")


class ModelInfo(BaseModel):
    """模型信息"""
    id: int = Field(..., description="模型ID")
    name: str = Field(..., description="模型名称")
    description: str = Field(..., description="模型描述")
    model_type: str = Field(..., description="模型类型")
    parameters: Dict[str, Any] = Field(..., description="模型参数")
    version: str = Field(..., description="模型版本")
    status: str = Field(..., description="模型状态")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")


class ClusterStatus(BaseModel):
    """集群状态"""
    status: str = Field(..., description="集群状态")
    cluster_resources: Dict[str, Any] = Field(default={}, description="集群资源")
    available_resources: Dict[str, Any] = Field(default={}, description="可用资源")
    workers: List[Dict[str, Any]] = Field(default=[], description="工作器列表")
    worker_count: int = Field(default=0, description="工作器数量")
    running_tasks: int = Field(default=0, description="运行中任务数")
    database_stats: Dict[str, Any] = Field(default={}, description="数据库统计")


class HealthStatus(BaseModel):
    """健康状态"""
    status: str = Field(..., description="整体状态")
    database: str = Field(..., description="数据库状态")
    ray_cluster: str = Field(..., description="Ray集群状态")
    timestamp: str = Field(..., description="检查时间")
    error: Optional[str] = Field(None, description="错误信息")
