"""
图像分类示例
"""

import asyncio
import aiohttp
import base64
import json
from pathlib import Path
from PIL import Image
import io

async def register_image_model():
    """注册图像分类模型"""
    model_data = {
        "name": "simple_image_classifier",
        "description": "简单的图像分类模型，支持10个类别",
        "model_type": "image_classification",
        "parameters": {
            "num_classes": 10,
            "input_size": 224,
            "channels": 3
        },
        "version": "1.0.0"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/v1/models/register",
            json=model_data
        ) as response:
            result = await response.json()
            print(f"模型注册结果: {result}")
            return result

async def create_sample_image():
    """创建示例图像"""
    # 创建一个简单的彩色图像
    image = Image.new('RGB', (224, 224), color='red')
    
    # 转换为Base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{image_base64}"

async def submit_image_task():
    """提交图像分类任务"""
    # 创建示例图像
    image_data = await create_sample_image()
    
    task_data = {
        "model_name": "simple_image_classifier",
        # "images": [image_data],
        "input_data": {"image": image_data},
        "image": [image_data],
        "task_type": "classification",
        "priority": 5
    }

    print("this is a test_info")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/v1/tasks/submit/image",
            json=task_data
        ) as response:
            result = await response.json()
            print(f"任务提交结果: {result}")
            return result["task_id"]

async def check_task_result(task_id: str):
    """检查任务结果"""
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(
                f"http://localhost:8000/api/v1/tasks/{task_id}"
            ) as response:
                result = await response.json()
                print(f"任务状态: {result['status']}, 进度: {result.get('progress', 0):.1%}")
                
                if result["status"] in ["completed", "failed"]:
                    print(f"任务最终结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
                    break
                
                await asyncio.sleep(2)

async def main():
    """主函数"""
    print("=== DistML 图像分类示例 ===")
    
    try:
        # 1. 注册模型
        # print("\n1. 注册图像分类模型...")
        # await register_image_model()
        
        # 2. 提交任务
        print("\n2. 提交图像分类任务...")
        task_id = await submit_image_task()
        
        # 3. 检查结果
        print(f"\n3. 检查任务结果 (任务ID: {task_id})...")
        await check_task_result(task_id)
        
        print("\n=== 示例完成 ===")
        
    except Exception as e:
        print(f"示例执行失败: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
