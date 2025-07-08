#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:11 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : routes.py
# @Software: PyCharm

"""
API路由定义
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import base64
import io

from api.models import (
    ModelRegistration, TaskSubmission, ImageTaskSubmission,
    TaskResponse, TaskResult, ModelInfo, ClusterStatus, HealthStatus
)
from core.database import db_manager
from scheduler.async_scheduler import scheduler
from core.logger import get_logger
from core.exceptions import ModelNotFoundError, TaskNotFoundError

logger = get_logger(__name__)

# 创建路由器
router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def root():
    """根路径，返回系统信息"""
    return {
        "system": "DistML",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "异步任务处理",
            "图像处理支持",
            "实时进度跟踪",
            "分布式计算",
            "模型缓存"
        ]
    }


# 模型管理路由
@router.post("/models/register", response_model=Dict[str, str])
async def register_model(model: ModelRegistration):
    """注册新模型"""
    try:
        logger.info(f"注册模型: {model.name}")

        model_id = await db_manager.register_model(
            name=model.name,
            description=model.description,
            model_type=model.model_type,
            parameters=model.parameters,
            version=model.version
        )


        logger.info(f"模型注册成功: {model.name}, ID: {model_id}")

        return {
            "model_id": model_id,
            "message": f"模型 {model.name} 注册成功"
        }

    except Exception as e:
        logger.error(f"模型注册失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型注册失败: {str(e)}")


@router.get("/models", response_model=Dict[str, Any])
async def list_models():
    """获取所有已注册的模型"""
    try:
        models = await db_manager.get_all_models()
        logger.info(f"查询到 {len(models)} 个已注册模型")

        return {
            "models": models,
            "count": len(models)
        }

    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@router.get("/models/{model_name}", response_model=ModelInfo)
async def get_model(model_name: str):
    """获取指定模型信息"""
    try:
        model = await db_manager.get_model_by_name(model_name)
        if not model:
            raise ModelNotFoundError(f"模型 {model_name} 不存在")

        return ModelInfo(**model)

    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


# 任务管理路由
@router.post("/tasks/submit", response_model=TaskResponse)
async def submit_task(task: TaskSubmission, background_tasks: BackgroundTasks):
    """提交新任务"""
    try:
        task_id = str(uuid.uuid4())
        logger.info(f"提交任务: {task_id}, 模型: {task.model_name}")

        # 检查模型是否存在
        model = await db_manager.get_model_by_name(task.model_name)
        if not model:
            raise ModelNotFoundError(f"模型 {task.model_name} 不存在")

        # 创建任务记录
        await db_manager.create_task(
            task_id=task_id,
            model_name=task.model_name,
            input_data=task.input_data,
            task_type=task.task_type,
            priority=task.priority
        )

        # 异步提交任务到调度器
        background_tasks.add_task(
            scheduler.submit_task,
            task_id,
            task.model_name,
            task.input_data,
            task.task_type,
            task.priority
        )

        logger.info(f"任务提交成功: {task_id}")

        return TaskResponse(
            task_id=task_id,
            status="submitted",
            message="任务已提交，正在处理中"
        )

    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"模型 {task.model_name} 不存在")
    except Exception as e:
        logger.error(f"任务提交失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"任务提交失败: {str(e)}")


@router.post("/tasks/submit/image", response_model=TaskResponse)
async def submit_image_task(task: ImageTaskSubmission, background_tasks: BackgroundTasks):
    """提交图像处理任务"""
    try:
        # 转换为通用任务格式
        general_task = task.to_task_submission()
        return await submit_task(general_task, background_tasks)

    except Exception as e:
        logger.error(f"图像任务提交失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图像任务提交失败: {str(e)}")


@router.post("/tasks/submit/image/upload", response_model=TaskResponse)
async def submit_image_upload_task(
        model_name: str,
        task_type: str = "classification",
        priority: int = 1,
        files: List[UploadFile] = File(...),
        background_tasks: BackgroundTasks = None
):
    """通过文件上传提交图像任务"""
    try:
        # 处理上传的图像文件
        images = []
        for file in files:
            # 读取文件内容
            content = await file.read()

            # 转换为Base64
            base64_image = base64.b64encode(content).decode('utf-8')
            images.append(f"data:image/{file.filename.split('.')[-1]};base64,{base64_image}")

        # 创建图像任务
        image_task = ImageTaskSubmission(
            model_name=model_name,
            images=images,
            task_type=task_type,
            priority=priority
        )

        return await submit_image_task(image_task, background_tasks)

    except Exception as e:
        logger.error(f"图像上传任务提交失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图像上传任务提交失败: {str(e)}")


@router.get("/tasks/{task_id}", response_model=TaskResult)
async def get_task_result(task_id: str):
    """查询任务结果"""
    try:
        logger.info(f"查询任务结果: {task_id}")

        task = await db_manager.get_task(task_id)
        if not task:
            raise TaskNotFoundError(f"任务 {task_id} 不存在")

        return TaskResult(**task)

    except TaskNotFoundError:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    except Exception as e:
        logger.error(f"查询任务结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询任务结果失败: {str(e)}")


@router.get("/tasks", response_model=Dict[str, Any])
async def list_tasks(status: Optional[str] = None, limit: int = 100):
    """获取任务列表"""
    try:
        tasks = await db_manager.get_tasks(status=status, limit=limit)
        logger.info(f"查询到 {len(tasks)} 个任务")

        return {
            "tasks": tasks,
            "count": len(tasks)
        }

    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


# 系统状态路由
@router.get("/health", response_model=HealthStatus)
async def health_check():
    """健康检查接口"""
    try:
        # 检查数据库连接
        db_status = "ok" if await db_manager.is_connected() else "error"

        # 检查Ray集群连接
        cluster_status = await scheduler.get_cluster_status()
        ray_status = "ok" if cluster_status.get("status") == "connected" else "error"

        overall_status = "ok" if db_status == "ok" and ray_status == "ok" else "error"

        return HealthStatus(
            status=overall_status,
            database=db_status,
            ray_cluster=ray_status,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return HealthStatus(
            status="error",
            database="error",
            ray_cluster="error",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )


@router.get("/cluster/status", response_model=ClusterStatus)
async def get_cluster_status():
    """获取集群状态"""
    try:
        status = await scheduler.get_cluster_status()
        return ClusterStatus(**status)

    except Exception as e:
        logger.error(f"获取集群状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取集群状态失败: {str(e)}")


@router.get("/stats", response_model=Dict[str, Any])
async def get_system_stats():
    """获取系统统计信息"""
    try:
        # 获取数据库统计
        db_stats = await db_manager.get_worker_stats()

        # 获取集群状态
        cluster_status = await scheduler.get_cluster_status()

        return {
            "database_stats": db_stats,
            "cluster_status": cluster_status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"获取系统统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统统计失败: {str(e)}")

