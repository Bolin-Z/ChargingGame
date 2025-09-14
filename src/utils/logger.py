"""
日志管理工具

提供统一的日志配置和管理功能
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level: str = 'INFO'):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_experiment_logger(experiment_name: str, output_dir: str):
    """
    为实验创建专用的日志记录器
    
    Args:
        experiment_name: 实验名称
        output_dir: 输出目录
    
    Returns:
        logging.Logger: 实验日志记录器
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = os.path.join(output_dir, 'logs', log_filename)
    
    return setup_logger(f"{experiment_name}_Logger", log_path, 'INFO')