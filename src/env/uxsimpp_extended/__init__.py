"""
UXsim++ Extended: 支持预定路径的交通仿真库

基于 UXsim++ 扩展，添加预定路径功能以支持充电博弈环境。
"""

import os
import sys

# 先导入 C++ 扩展模块，防止循环导入
# uxsimpp.py 中的 "from . import trafficppy" 依赖此行
from . import trafficppy

# 然后导入 Python 包装功能
from .uxsimpp import *

__version__ = "0.1.0"
__author__ = "ChargingGame Project"
__license__ = "MIT License"


# 临时解决方案：防止退出时的 segmentation fault
import atexit
import gc
import io

def _safe_cleanup_and_exit():
    # 刷新标准输出/标准错误
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass

    # 关闭打开的文件对象
    for obj in gc.get_objects():
        if isinstance(obj, io.IOBase) and not obj.closed:
            try:
                obj.close()
            except Exception:
                pass

    # 使用 os._exit 立即退出
    os._exit(0)

# 注册 atexit 处理程序
atexit.register(_safe_cleanup_and_exit)
