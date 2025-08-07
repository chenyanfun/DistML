"""
批量图像处理示例
"""

import asyncio
import aiohttp
import base64
import json
from PIL import Image, ImageDraw
import io
import random

async def create_sample_images(count: int = 5):
    """创建多个示例图像"""
    images = []
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for i in range(count):
        # 创建不同颜色的图像
        color = colors[i % len(colors)]
        image = Image.new('RGB', (224, 224), color=color)
        
        # 添加一些简单的图形
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 174, 174], outline='white', width=3)
        draw.text((100, 100), f"Image {i+1}", fill='white')
        
        # 转换为Base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        images.append(f"data:image/png;base64,{image_base64}")
    
    return images

async def submit_batch_tasks():
    """提交批量图像处理任务"""
    print("创建示例图像...")
    images = await create_sample_images(10)
    
    task_ids = []
    
    async with aiohttp.ClientSession() as session:
        # 提交多个任务
        for i in range(0, len(images), 2):  # 每次处理2张图像
            batch_images = images[i:i+2]
            
            task_data = {
                "model_name": "simple_image_classifier",
                "images": batch_images,
                "task_type": "classification",
                "priority": random.randint(1, 10)
            }
            
            async with session.post(
                "http://localhost:8000/api/v1/tasks/submit/image",
                json=task_data
            ) as response:
                result = await response.json()
                task_ids.append(result["task_id"])
                print(f"提交批次 {i//2 + 1}: 任务ID {result['task_id']}")
    
    return task_ids

async def monitor_batch_tasks(task_ids: list):
    """监控批量任务执行"""
    completed_tasks = set()
    
    async with aiohttp.ClientSession() as session:
        while len(completed_tasks) < len(task_ids):
            print(f"\n检查任务状态... ({len(completed_tasks)}/{len(task_ids)} 已完成)")
            
            for task_id in task_ids:
                if task_id in completed_tasks:
                    continue
                
                async with session.get(
                    f"http://localhost:8000/api/v1/tasks/{task_id}"
                ) as response:
                    result = await response.json()
                    
                    status = result["status"]
                    progress = result.get("progress", 0)
                    
                    if status in ["completed", "failed"]:
                        completed_tasks.add(task_id)
                        print(f"  任务 {task_id[:8]}... : {status}")
                        
                        if status == "completed" and result.get("result"):
                            predictions = result["result"].get("result", {}).get("predictions", [])
                            print(f"    预测结果: {predictions}")
                    else:
                        print(f"  任务 {task_id[:8]}... : {status} ({progress:.1%})")
            
            if len(completed_tasks) < len(task_ids):
                await asyncio.sleep(3)
    
    print(f"\n所有任务已完成! 总计: {len(task_ids)} 个任务")

async def get_system_stats():
    """获取系统统计信息"""
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/api/v1/stats") as response:
            stats = await response.json()
            print("\n=== 系统统计信息 ===")
            print(json.dumps(stats, indent=2, ensure_ascii=False))

async def main():
    """主函数"""
    print("=== DistML 批量图像处理示例 ===")
    
    try:
        # 1. 提交批量任务
        print("\n1. 提交批量图像处理任务...")
        task_ids = await submit_batch_tasks()
        
        # 2. 监控任务执行
        print(f"\n2. 监控 {len(task_ids)} 个任务的执行状态...")
        await monitor_batch_tasks(task_ids)
        
        # 3. 获取系统统计
        print("\n3. 获取系统统计信息...")
        await get_system_stats()
        
        print("\n=== 批量处理示例完成 ===")
        
    except Exception as e:
        print(f"示例执行失败: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
