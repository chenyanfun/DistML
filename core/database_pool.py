#!/usr/bin/python3.10
# @Time    : 7/7/2025 5:23 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : database_pool.py
# @Software: PyCharm

"""
数据库连接池管理器
解决Ray序列化问题
"""

import asyncio
import aiosqlite
from typing import Dict, Optional
from contextlib import asynccontextmanager
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class DatabasePool:
    """数据库连接池"""

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    async def initialize(self, db_url: str = None):
        """初始化连接池"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            self.db_url = db_url or settings.DATABASE_URL.replace("sqlite+aiosqlite://", "")
            self._connections = {}
            self._initialized = True
            logger.info(f"数据库连接池初始化完成: {self.db_url}")

    async def get_connection(self) -> aiosqlite.Connection:
        """获取数据库连接"""
        if not self._initialized:
            await self.initialize()

        # 为每个任务创建独立的连接
        task_id = id(asyncio.current_task())

        if task_id not in self._connections:
            conn = await aiosqlite.connect(self.db_url)
            conn.row_factory = aiosqlite.Row
            self._connections[task_id] = conn

        return self._connections[task_id]

    @asynccontextmanager
    async def transaction(self):
        """事务上下文管理器"""
        conn = await self.get_connection()
        try:
            await conn.execute("BEGIN")
            yield conn
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            logger.error(f"数据库事务失败: {str(e)}")
            raise

    async def close_all(self):
        """关闭所有连接"""
        if hasattr(self, '_connections'):
            for conn in self._connections.values():
                await conn.close()
            self._connections.clear()
        logger.info("所有数据库连接已关闭")


# 全局连接池实例
db_pool = DatabasePool()
