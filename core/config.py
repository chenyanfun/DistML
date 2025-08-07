#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:12 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : config.py
# @Software: PyCharm

"""
配置管理模块
"""

import os
from typing import List, Optional
# from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """应用配置"""

    # 应用基础配置
    APP_NAME: str = "DistML"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")

    # 服务器配置
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")

    # 数据库配置
    DATABASE_URL: str = Field(default="sqlite+aiosqlite://./data/distml.db", env="DATABASE_URL")

    # Ray配置
    RAY_ADDRESS: str = Field(default="auto", env="RAY_ADDRESS")
    RAY_NAMESPACE: str = Field(default="distml", env="RAY_NAMESPACE")
    RAY_CLIENT_SERVER_PORT: int = Field(default=10001, env="RAY_CLIENT_SERVER_PORT")

    # 工作器配置
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    WORKER_TIMEOUT: int = Field(default=300, env="WORKER_TIMEOUT")

    # 任务配置
    MAX_CONCURRENT_TASKS: int = Field(default=10, env="MAX_CONCURRENT_TASKS")
    TASK_TIMEOUT: int = Field(default=600, env="TASK_TIMEOUT")

    # 存储配置
    STORAGE_PATH: Path = Field(default=Path("data/storage"), env="STORAGE_PATH")
    UPLOAD_PATH: Path = Field(default=Path("data/uploads"), env="UPLOAD_PATH")

    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default="distml.log", env="LOG_FILE")

    # Redis配置（可选，用于缓存和消息队列）
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")

    # 监控配置
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")

    class Config:
        env_file = ".env"
        case_sensitive = True


# 全局配置实例
settings = Settings()

# 确保必要的目录存在
settings.STORAGE_PATH.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
