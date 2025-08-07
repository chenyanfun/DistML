#!/usr/bin/python3.10
# @Time    : 7/7/2025 4:13 PM
# @Author  : chenyan
# @Email   : chenyanfun@gmail.com
# @File    : exceptions.py
# @Software: PyCharm

"""
自定义异常类
"""

class DistMLException(Exception):
    """DistML基础异常类"""
    pass

class ModelNotFoundError(DistMLException):
    """模型未找到异常"""
    pass

class TaskNotFoundError(DistMLException):
    """任务未找到异常"""
    pass

class WorkerNotAvailableError(DistMLException):
    """工作器不可用异常"""
    pass

class ModelLoadError(DistMLException):
    """模型加载异常"""
    pass

class TaskExecutionError(DistMLException):
    """任务执行异常"""
    pass

class ValidationError(DistMLException):
    """数据验证异常"""
    pass
