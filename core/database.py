#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:13 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : database.py
# @Software: PyCharm


"""
异步数据库管理器 - 重构版本
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from core.database_pool import db_pool
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class AsyncDatabaseManager:
    """异步数据库管理器"""

    def __init__(self, db_url: str = None):
        self.db_url = db_url or settings.DATABASE_URL.replace("sqlite+aiosqlite://", "")

    async def initialize(self):
        """初始化数据库管理器"""
        await db_pool.initialize(self.db_url)

    @asynccontextmanager
    async def transaction(self):
        """事务上下文管理器"""
        async with db_pool.transaction() as conn:
            yield conn

    async def register_model(self, name: str, description: str, model_type: str,
                             parameters: Dict[str, Any] = None, version: str = "1.0.0") -> str:
        """注册新模型"""
        now = datetime.now().isoformat()
        parameters_json = json.dumps(parameters or {})

        async with self.transaction() as conn:
            cursor = await conn.execute("""
                INSERT INTO models (name, description, model_type, parameters, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, description, model_type, parameters_json, version, now, now))

            model_id = str(cursor.lastrowid)

        logger.info(f"模型注册成功: {name} (ID: {model_id})")
        return model_id

    async def get_model_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取模型信息"""
        conn = await db_pool.get_connection()
        cursor = await conn.execute(
            "SELECT * FROM models WHERE name = ? AND status = 'active'", (name,)
        )
        row = await cursor.fetchone()

        if row:
            model = dict(row)
            model['parameters'] = json.loads(model['parameters'])
            return model
        return None

    async def get_all_models(self) -> List[Dict[str, Any]]:
        """获取所有模型信息"""
        conn = await db_pool.get_connection()
        cursor = await conn.execute(
            "SELECT * FROM models WHERE status = 'active' ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()

        models = []
        for row in rows:
            model = dict(row)
            model['parameters'] = json.loads(model['parameters'])
            models.append(model)

        return models

    async def create_task(self, task_id: str, model_name: str, input_data: Dict[str, Any],
                          task_type: str = "inference", priority: int = 1) -> str:
        """创建新任务"""
        now = datetime.now().isoformat()
        input_data_json = json.dumps(input_data)

        async with self.transaction() as conn:
            # 创建任务记录
            await conn.execute("""
                INSERT INTO tasks (task_id, model_name, input_data, task_type, priority, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (task_id, model_name, input_data_json, task_type, priority, now, now))

            # 添加到任务队列
            await conn.execute("""
                INSERT INTO task_queue (task_id, priority, created_at)
                VALUES (?, ?, ?)
            """, (task_id, priority, now))

        logger.info(f"任务创建成功: {task_id}")
        return task_id

    async def update_task_status(self, task_id: str, status: str,
                                 result: Dict[str, Any] = None, error: str = None,
                                 progress: float = None, worker_id: str = None):
        """更新任务状态"""
        now = datetime.now().isoformat()
        result_json = json.dumps(result) if result else None

        update_fields = ["status = ?", "updated_at = ?"]
        update_values = [status, now]

        if result_json:
            update_fields.append("result = ?")
            update_values.append(result_json)

        if error:
            update_fields.append("error = ?")
            update_values.append(error)

        if progress is not None:
            update_fields.append("progress = ?")
            update_values.append(progress)

        if worker_id:
            update_fields.append("worker_id = ?")
            update_values.append(worker_id)

        if status == "running" and not await self.get_task_field(task_id, "started_at"):
            update_fields.append("started_at = ?")
            update_values.append(now)

        if status in ['completed', 'failed']:
            update_fields.append("completed_at = ?")
            update_values.append(now)
            # 从队列中移除
            await self.remove_from_queue(task_id)

        update_values.append(task_id)

        conn = await db_pool.get_connection()
        await conn.execute(f"""
            UPDATE tasks SET {', '.join(update_fields)}
            WHERE task_id = ?
        """, update_values)
        await conn.commit()

        logger.info(f"任务状态更新: {task_id} -> {status}")

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        conn = await db_pool.get_connection()
        cursor = await conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = await cursor.fetchone()

        if row:
            task = dict(row)
            task['input_data'] = json.loads(task['input_data'])
            if task['result']:
                task['result'] = json.loads(task['result'])
            return task
        return None

    async def get_task_field(self, task_id: str, field: str) -> Any:
        """获取任务的特定字段"""
        conn = await db_pool.get_connection()
        cursor = await conn.execute(f"SELECT {field} FROM tasks WHERE task_id = ?", (task_id,))
        row = await cursor.fetchone()
        return row[0] if row else None

    async def get_next_task(self) -> Optional[Dict[str, Any]]:
        """获取下一个待处理任务"""
        conn = await db_pool.get_connection()
        cursor = await conn.execute("""
            SELECT t.* FROM tasks t
            JOIN task_queue q ON t.task_id = q.task_id
            WHERE t.status = 'pending'
            ORDER BY q.priority DESC, q.created_at ASC
            LIMIT 1
        """)
        row = await cursor.fetchone()

        if row:
            task = dict(row)
            task['input_data'] = json.loads(task['input_data'])
            return task
        return None

    async def remove_from_queue(self, task_id: str):
        """从任务队列中移除任务"""
        conn = await db_pool.get_connection()
        await conn.execute("DELETE FROM task_queue WHERE task_id = ?", (task_id,))
        await conn.commit()

    async def get_tasks(self, status: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取任务列表"""
        conn = await db_pool.get_connection()

        if status:
            cursor = await conn.execute("""
                SELECT * FROM tasks WHERE status = ? 
                ORDER BY created_at DESC LIMIT ?
            """, (status, limit))
        else:
            cursor = await conn.execute("""
                SELECT * FROM tasks 
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))

        rows = await cursor.fetchall()

        tasks = []
        for row in rows:
            task = dict(row)
            task['input_data'] = json.loads(task['input_data'])
            if task['result']:
                task['result'] = json.loads(task['result'])
            tasks.append(task)

        return tasks

    async def update_worker_status(self, worker_id: str, status: str,
                                   current_task: str = None):
        """更新工作器状态"""
        now = datetime.now().isoformat()
        conn = await db_pool.get_connection()

        # 尝试更新现有记录
        cursor = await conn.execute("""
            UPDATE workers SET status = ?, current_task = ?, last_heartbeat = ?, updated_at = ?
            WHERE worker_id = ?
        """, (status, current_task, now, now, worker_id))

        # 如果没有更新任何记录，则插入新记录
        if cursor.rowcount == 0:
            await conn.execute("""
                INSERT INTO workers (worker_id, status, current_task, last_heartbeat, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (worker_id, status, current_task, now, now, now))

        await conn.commit()

    async def get_worker_stats(self) -> Dict[str, Any]:
        """获取工作器统计信息"""
        conn = await db_pool.get_connection()

        # 获取工作器状态统计
        cursor = await conn.execute("""
            SELECT status, COUNT(*) as count FROM workers GROUP BY status
        """)
        worker_stats = {row[0]: row[1] for row in await cursor.fetchall()}

        # 获取任务统计
        cursor = await conn.execute("""
            SELECT status, COUNT(*) as count FROM tasks GROUP BY status
        """)
        task_stats = {row[0]: row[1] for row in await cursor.fetchall()}

        return {
            "workers": worker_stats,
            "tasks": task_stats
        }

    async def is_connected(self) -> bool:
        """检查数据库连接状态"""
        try:
            conn = await db_pool.get_connection()
            await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"数据库连接检查失败: {str(e)}")
            return False

    async def close(self):
        """关闭数据库连接"""
        await db_pool.close_all()
        logger.info("数据库连接已关闭")


# 全局数据库管理器实例
db_manager = AsyncDatabaseManager()
