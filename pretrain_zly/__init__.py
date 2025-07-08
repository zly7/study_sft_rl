"""
PreTrain ZLY 预训练模块

此模块包含所有预训练相关的代码，包括：
- 配置管理
- 模型工具
- 数据处理
- 日志设置
- 预训练主逻辑
"""

from .config import ModelArguments, DataTrainingArguments, parse_arguments
from .model_utils import check_checkpoint, initialize_model, initialize_tokenizer
from .data_processing import load_and_process_dataset, group_texts, tokenize_function
from .logger_setup import setup_logging
from .pretrain_main import main

__all__ = [
    "ModelArguments",
    "DataTrainingArguments", 
    "parse_arguments",
    "check_checkpoint",
    "initialize_model",
    "initialize_tokenizer",
    "load_and_process_dataset",
    "group_texts",
    "tokenize_function", 
    "setup_logging",
    "main"
]