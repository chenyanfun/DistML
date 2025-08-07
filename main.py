"""
主应用入口
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

from core.config import settings
from core.database import db_manager
from scheduler.async_scheduler import scheduler
from api.routes import router
from core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("DistML系统启动中...")

    try:
        # 初始化数据库管理器
        await db_manager.initialize()
        logger.info("数据库管理器初始化完成")

        # 初始化调度器
        await scheduler.initialize()
        logger.info("调度器初始化完成")

        logger.info("DistML系统启动完成")
        yield

    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}")
        raise
    finally:
        # 关闭时清理资源
        logger.info("DistML系统关闭中...")

        try:
            await scheduler.shutdown()
            logger.info("调度器已关闭")

            await db_manager.close()
            logger.info("数据库连接已关闭")

        except Exception as e:
            logger.error(f"系统关闭失败: {str(e)}")

        logger.info("DistML系统已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="DistML - 分布式机器学习系统",
    description="基于FastAPI + Ray的异步分布式机器学习平台，支持图像处理和实时任务调度",
    version="2.0.0",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 注册路由
app.include_router(router, prefix="/api/v1")


# 根路径重定向到API文档
@app.get("/")
async def root():
    return {
        "message": "欢迎使用DistML分布式机器学习系统",
        "version": "2.0.0",
        "docs": "/docs",
        "api": "/api/v1"
    }


if __name__ == "__main__":
    logger.info("启动DistML服务...")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
