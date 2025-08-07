#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:14 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : logger.py
# @Software: PyCharm

"""
日志配置模块
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from core.config import settings


def setup_logging():
    """设置全局日志配置"""
    log_level = getattr(logging, settings.LOG_LEVEL.upper())

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 根日志器配置
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # 文件处理器（如果配置了日志文件）
    if settings.LOG_FILE:
        log_file = Path(settings.LOG_FILE)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 设置第三方库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("ray").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志器"""
    return logging.getLogger(name)


# 初始化日志配置
setup_logging()
