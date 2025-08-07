#!/usr/bin/python3.10
# @Time    : 7/8/2025 10:20 AM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : run_demo_simple.py
# @Software: PyCharm

import asyncio
import time
import requests
import json
from core.logger import get_logger

logger = get_logger(__name__)

class DistMLDemo:
    """DistML系统演示类"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化演示类

        Args:
            base_url: API服务基础URL
        """
        self.base_url = base_url
        logger.info(f"DeepFlow演示初始化: {base_url}")

    def test_api_connection(self) -> bool:
        """测试API连接"""
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                logger.info("API连接测试成功")
                return True
            else:
                logger.error(f"API连接测试失败: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"API连接测试异常: {str(e)}")
            return False

    def register_demo_models(self):
        """注册演示模型"""
        logger.info("注册演示模型...")

        models = [
            {
                "name": "simple_regression",
                "description": "简单回归模型，用于数值预测",
                "model_type": "regression",
                "parameters": {
                    "input_size": 5,
                    "hidden_sizes": [32, 16]
                }
            },
            {
                "name": "simple_classification",
                "description": "简单分类模型，用于多类别分类",
                "model_type": "classification",
                "parameters": {
                    "input_size": 8,
                    "num_classes": 3,
                    "hidden_sizes": [64, 32]
                }
            },
            {
                "name": "simple_mlp",
                "description": "通用多层感知机模型",
                "model_type": "mlp",
                "parameters": {
                    "input_size": 10,
                    "hidden_sizes": [64, 32],
                    "output_size": 1,
                    "dropout_rate": 0.2
                }
            }
        ]

        for model in models:
            try:
                response = requests.post(
                    f"{self.base_url}/models/register",
                    json=model
                )
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"模型注册成功: {model['name']} (ID: {result['model_id']})")
                else:
                    logger.error(f"模型注册失败: {model['name']}, 状态码: {response.status_code}")
            except Exception as e:
                logger.error(f"模型注册异常: {model['name']}, 错误: {str(e)}")


def submit_demo_tasks(self):
    """提交演示任务"""
    logger.info("提交演示任务...")

    tasks = [
        {
            "model_name": "simple_regression",
            "input_data": {
                "data": [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]
            },
            "task_type": "inference",
            "priority": 1
        },
        {
            "model_name": "simple_classification",
            "input_data": {
                "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
            },
            "task_type": "classification",
            "priority": 2
        },
        {
            "model_name": "simple_mlp",
            "input_data": {
                "data": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
            },
            "task_type": "inference",
            "priority": 1
        }
    ]

    task_ids = []
    for task in tasks:
        try:
            response = requests.post(
                f"{self.base_url}/tasks/submit",
                json=task
            )
            if response.status_code == 200:
                result = response.json()
                task_id = result["task_id"]
                task_ids.append(task_id)
                logger.info(f"任务提交成功: {task['model_name']} (ID: {task_id})")
            else:
                logger.error(f"任务提交失败: {task['model_name']}, 状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"任务提交异常: {task['model_name']}, 错误: {str(e)}")

    return task_ids


def check_task_results(self, task_ids: list, max_wait_time: int = 30):
    """检查任务结果"""
    logger.info(f"检查任务结果: {len(task_ids)} 个任务")

    start_time = time.time()
    completed_tasks = set()

    while len(completed_tasks) < len(task_ids) and (time.time() - start_time) < max_wait_time:
        for task_id in task_ids:
            if task_id in completed_tasks:
                continue

            try:
                response = requests.get(f"{self.base_url}/tasks/{task_id}")
                if response.status_code == 200:
                    result = response.json()
                    status = result["status"]

                    if status in ["completed", "failed"]:
                        completed_tasks.add(task_id)
                        logger.info(f"任务完成: {task_id}, 状态: {status}")

                        if status == "completed" and result.get("result"):
                            logger.info(f"任务结果: {json.dumps(result['result'], indent=2, ensure_ascii=False)}")
                        elif status == "failed":
                            logger.error(f"任务失败: {task_id}, 错误: {result.get('error', 'Unknown error')}")
                    else:
                        logger.info(f"任务进行中: {task_id}, 状态: {status}")
                else:
                    logger.error(f"查询任务失败: {task_id}, 状态码: {response.status_code}")
            except Exception as e:
                logger.error(f"查询任务异常: {task_id}, 错误: {str(e)}")

        if len(completed_tasks) < len(task_ids):
            time.sleep(2)  # 等待2秒后再次检查

    if len(completed_tasks) == len(task_ids):
        logger.info("所有任务已完成")
    else:
        logger.warning(f"等待超时，仍有 {len(task_ids) - len(completed_tasks)} 个任务未完成")


def check_system_health(self):
    """检查系统健康状态"""
    logger.info("检查系统健康状态...")

    try:
        response = requests.get(f"{self.base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"系统健康状态: {json.dumps(health_data, indent=2, ensure_ascii=False)}")
            return health_data["status"] == "ok"
        else:
            logger.error(f"健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"健康检查异常: {str(e)}")
        return False


def list_models_and_tasks(self):
    """列出模型和任务"""
    logger.info("获取模型和任务列表...")

    # 获取模型列表
    try:
        response = requests.get(f"{self.base_url}/models")
        if response.status_code == 200:
            models_data = response.json()
            logger.info(f"已注册模型数量: {models_data['count']}")
            for model in models_data["models"]:
                logger.info(f"  - {model['name']}: {model['description']}")
        else:
            logger.error(f"获取模型列表失败: {response.status_code}")
    except Exception as e:
        logger.error(f"获取模型列表异常: {str(e)}")

    # 获取任务列表
    try:
        response = requests.get(f"{self.base_url}/tasks?limit=10")
        if response.status_code == 200:
            tasks_data = response.json()
            logger.info(f"任务总数: {tasks_data['count']}")
            for task in tasks_data["tasks"]:
                logger.info(f"  - {task['task_id']}: {task['model_name']} ({task['status']})")
        else:
            logger.error(f"获取任务列表失败: {response.status_code}")
    except Exception as e:
        logger.error(f"获取任务列表异常: {str(e)}")


def run_full_demo(self):
    """运行完整演示"""
    logger.info("开始DeepFlow完整演示...")

    # 1. 测试API连接
    if not self.test_api_connection():
        logger.error("API连接失败，演示终止")
        return

    # 2. 检查系统健康状态
    if not self.check_system_health():
        logger.warning("系统健康检查失败，但继续演示")

    # 3. 注册演示模型
    self.register_demo_models()

    # 4. 等待一段时间让模型注册完成
    time.sleep(2)

    # 5. 提交演示任务
    task_ids = self.submit_demo_tasks()

    if not task_ids:
        logger.error("没有成功提交的任务，演示终止")
        return

    # 6. 等待一段时间让任务开始执行
    time.sleep(3)

    # 7. 检查任务结果
    self.check_task_results(task_ids)

    # 8. 列出模型和任务
    self.list_models_and_tasks()

    # 9. 最终健康检查
    self.check_system_health()

    logger.info("DeepFlow完整演示结束")


async def main():
    """主函数"""
    logger.info("DeepFlow系统演示启动")

    # 等待用户启动主服务
    print("\n" + "=" * 50)
    print("请在另一个终端中启动DistML主服务:")
    print("python main.py")
    print("然后按Enter键继续API演示...")
    print("=" * 50)
    input()

    # 运行API演示
    demo = DistMLDemo()
    demo.run_full_demo()

    logger.info("DistML系统演示完成")


if __name__ == "__main__":
    asyncio.run(main())