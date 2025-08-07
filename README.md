# 分布式机器学习调度系统

## 名称：DistML
## 版本：0.1.0
## 作者：chenyanfun
## 介绍
使用fastapi实现的分布式机器学习调度系统，支持多种机器学习框架。

## 运行
### 快速启动
* 安装环境依赖
```bash
pip install -r requirements.txt
```

* 启动本地ray服务（可选）
单机运行ray
```bash
ray start --head --ray-client-server-port=10001 --include-dashboard true
```
* 启动服务
```bash
python main.py
```
* 运行测试脚本（普通分类)
```bash
python run_demo_simple.py
```
* 运行测试脚本（视觉相关）
```bash
python run_demo_vision.py
```
