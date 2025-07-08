#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:19 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : async_scheduler.py
# @Software: PyCharm


"""
异步Ray调度器
"""

import ray
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from workers.ray_worker import AsyncModelWorker
from core.database import db_manager
from core.config import settings
from core.logger import get_logger
from core.exceptions import WorkerNotAvailableError, TaskExecutionError

logger = get_logger(__name__)


class AsyncRayScheduler:
    """异步Ray调度器"""

    def __init__(self):
        self.workers: List[ray.ObjectRef] = []
        self.worker_pool = []
        self.is_initialized = False
        self.task_queue = asyncio.Queue()
        self.running_tasks = {}
        self._scheduler_task = None

    async def initialize(self):
        """初始化调度器"""
        try:
            logger.info("初始化异步Ray调度器...")

            # 初始化Ray
            if not ray.is_initialized():
                ray.init(
                    address=settings.RAY_ADDRESS if settings.RAY_ADDRESS != "auto" else None,
                    namespace=settings.RAY_NAMESPACE,
                    ignore_reinit_error=True
                )

            # 创建工作器池
            await self._create_worker_pool()

            # 启动任务调度循环
            self._scheduler_task = asyncio.create_task(self._task_scheduler_loop())

            self.is_initialized = True
            logger.info(f"异步Ray调度器初始化完成: {len(self.workers)} 个工作器")

        except Exception as e:
            logger.error(f"调度器初始化失败: {str(e)}")
            raise

    async def _create_worker_pool(self):
        """创建工作器池"""
        logger.info(f"创建 {settings.MAX_WORKERS} 个工作器...")

        for i in range(settings.MAX_WORKERS):
            worker_id = f"worker_{i}_{uuid.uuid4().hex[:8]}"
            worker = AsyncModelWorker.remote(worker_id)
            self.workers.append(worker)
            self.worker_pool.append({
                "worker": worker,
                "worker_id": worker_id,
                "status": "idle",
                "current_task": None
            })

            # 更新数据库中的工作器状态
            await db_manager.update_worker_status(worker_id, "idle")

        logger.info(f"工作器池创建完成: {len(self.workers)} 个工作器")

    async def _task_scheduler_loop(self):
        """任务调度循环"""
        logger.info("启动任务调度循环")

        while True:
            try:
                # 检查是否有待处理的任务
                next_task = await db_manager.get_next_task()
                if next_task:
                    await self._assign_task(next_task)

                # 检查运行中的任务状态
                await self._check_running_tasks()

                # 短暂休眠避免过度轮询
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"任务调度循环错误: {str(e)}")
                await asyncio.sleep(5)

    async def _assign_task(self, task: Dict[str, Any]):
        """分配任务给工作器"""
        try:
            # 查找空闲的工作器
            available_worker = None
            for worker_info in self.worker_pool:
                if worker_info["status"] == "idle":
                    available_worker = worker_info
                    break

            if not available_worker:
                logger.warning(f"没有可用的工作器处理任务: {task['task_id']}")
                return

            # 标记工作器为忙碌
            available_worker["status"] = "busy"
            available_worker["current_task"] = task["task_id"]

            # 更新任务状态
            await db_manager.update_task_status(
                task["task_id"], "running",
                worker_id=available_worker["worker_id"]
            )

            # 更新工作器状态
            await db_manager.update_worker_status(
                available_worker["worker_id"], "busy", task["task_id"]
            )

            # 获取模型信息
            model_info = await db_manager.get_model_by_name(task["model_name"])
            if not model_info:
                raise TaskExecutionError(f"模型不存在: {task['model_name']}")

            # 异步执行任务
            task_future = asyncio.create_task(
                self._execute_task_with_worker(available_worker, task, model_info)
            )

            self.running_tasks[task["task_id"]] = {
                "future": task_future,
                "worker": available_worker,
                "start_time": datetime.now()
            }

            logger.info(f"任务已分配: {task['task_id']} -> {available_worker['worker_id']}")

        except Exception as e:
            logger.error(f"任务分配失败: {task['task_id']}, 错误: {str(e)}")
            await db_manager.update_task_status(
                task["task_id"], "failed", error=str(e)
            )

    async def _execute_task_with_worker(self, worker_info: Dict[str, Any],
                                        task: Dict[str, Any], model_info: Dict[str, Any]):
        """使用工作器执行任务"""
        worker = worker_info["worker"]
        task_id = task["task_id"]

        try:
            # 进度回调函数
            async def progress_callback(task_id: str, progress: float, message: str):
                await db_manager.update_task_status(
                    task_id, "running", progress=progress
                )
                logger.info(f"任务进度: {task_id} - {progress:.1%} - {message}")

            # 确保模型已加载
            load_success = await worker.load_model.remote(
                task["model_name"],
                model_info["model_type"],
                model_info["parameters"]
            )

            if not load_success:
                raise TaskExecutionError(f"模型加载失败: {task['model_name']}")
            logger.info(f"{task_id},开始准备执行任务")

            try:

                # 执行任务
                result = await worker.execute_task.remote(
                    task_id,
                    task["model_name"],
                    task["input_data"],
                    task["task_type"],
                    # progress_callback
                )
            except Exception as e:

                logger.info(f"{task_id},任务执行完毕,任务报错信息为{e}")

            # 处理结果
            if result.get("status") == "success":
                await db_manager.update_task_status(
                    task_id, "completed", result=result
                )
                logger.info(f"任务执行成功: {task_id}")
            else:
                await db_manager.update_task_status(
                    task_id, "failed", error=result.get("error", "未知错误")
                )
                logger.error(f"任务执行失败: {task_id}")

        except Exception as e:
            error_msg = f"任务执行异常: {task_id}, 错误: {str(e)}"
            logger.error(error_msg)
            await db_manager.update_task_status(task_id, "failed", error=error_msg)

        finally:
            # 释放工作器
            worker_info["status"] = "idle"
            worker_info["current_task"] = None
            await db_manager.update_worker_status(worker_info["worker_id"], "idle")

            # 从运行任务列表中移除
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    async def _check_running_tasks(self):
        """检查运行中的任务状态"""
        current_time = datetime.now()
        timeout_tasks = []

        for task_id, task_info in self.running_tasks.items():
            # 检查任务是否超时
            elapsed = (current_time - task_info["start_time"]).total_seconds()
            if elapsed > settings.TASK_TIMEOUT:
                timeout_tasks.append(task_id)
                logger.warning(f"任务超时: {task_id}, 已运行 {elapsed:.1f} 秒")

        # 处理超时任务
        for task_id in timeout_tasks:
            await self._handle_timeout_task(task_id)

    async def _handle_timeout_task(self, task_id: str):
        """处理超时任务"""
        try:
            task_info = self.running_tasks.get(task_id)
            if task_info:
                # 取消任务
                task_info["future"].cancel()

                # 更新任务状态
                await db_manager.update_task_status(
                    task_id, "failed", error="任务执行超时"
                )

                # 释放工作器
                worker_info = task_info["worker"]
                worker_info["status"] = "idle"
                worker_info["current_task"] = None
                await db_manager.update_worker_status(worker_info["worker_id"], "idle")

                # 从运行任务列表中移除
                del self.running_tasks[task_id]

                logger.info(f"超时任务已处理: {task_id}")

        except Exception as e:
            logger.error(f"处理超时任务失败: {task_id}, 错误: {str(e)}")

    async def submit_task(self, task_id: str, model_name: str, input_data: Dict[str, Any],
                          task_type: str, priority: int = 1):
        """提交任务"""
        try:
            logger.info(f"提交任务: {task_id}")

            # 任务已经在数据库中创建，这里只需要触发调度
            # 调度循环会自动处理新任务

        except Exception as e:
            logger.error(f"任务提交失败: {task_id}, 错误: {str(e)}")
            raise

    async def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        try:
            # 获取工作器状态
            worker_statuses = []
            for worker_info in self.worker_pool:
                try:
                    status = await worker_info["worker"].get_status.remote()
                    worker_statuses.append(status)
                except Exception as e:
                    worker_statuses.append({
                        "worker_id": worker_info["worker_id"],
                        "status": "error",
                        "error": str(e)
                    })

            # 获取Ray集群信息
            cluster_resources = ray.cluster_resources() if ray.is_initialized() else {}
            available_resources = ray.available_resources() if ray.is_initialized() else {}

            # 获取数据库统计
            db_stats = await db_manager.get_worker_stats()

            return {
                "status": "connected" if self.is_initialized else "disconnected",
                "cluster_resources": cluster_resources,
                "available_resources": available_resources,
                "workers": worker_statuses,
                "worker_count": len(self.workers),
                "running_tasks": len(self.running_tasks),
                "database_stats": db_stats
            }

        except Exception as e:
            logger.error(f"获取集群状态失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def shutdown(self):
        """关闭调度器"""
        try:
            logger.info("关闭异步Ray调度器...")

            # 停止调度循环
            if self._scheduler_task:
                self._scheduler_task.cancel()
                try:
                    await self._scheduler_task
                except asyncio.CancelledError:
                    pass

            # 取消所有运行中的任务
            for task_id, task_info in self.running_tasks.items():
                task_info["future"].cancel()

            # 关闭Ray
            if ray.is_initialized():
                ray.shutdown()

            self.is_initialized = False
            logger.info("异步Ray调度器已关闭")

        except Exception as e:
            logger.error(f"调度器关闭失败: {str(e)}")


# 全局调度器实例
scheduler = AsyncRayScheduler()
