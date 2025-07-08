import logging
import math
import os
import sys
import deepspeed
import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
    Trainer,
    default_data_collator,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
import swanlab

from .config import parse_arguments
from .logger_setup import setup_logging
from .model_utils import check_checkpoint, initialize_model, initialize_tokenizer
from .data_processing import load_and_process_dataset


def main():
    # 加载脚本参数
    model_args, data_args, training_args = parse_arguments()

    # 设置日志
    logger = setup_logging(training_args)

    # 检查 checkpoint
    last_checkpoint = check_checkpoint(training_args)

    # 初始化模型
    model = initialize_model(model_args)

    # 预训练一般将文本拼接成固定长度的文本段
    # 这里我们取块长为 2048
    block_size = 2048

    # 初始化 tokenizer
    # 注意：这里需要根据实际情况修改 model_name_or_path
    model_name_or_path = model_args.model_name_or_path or model_args.config_name
    tokenizer = initialize_tokenizer(model_name_or_path)

    # 加载和处理数据集
    # 注意：这里需要根据实际情况修改数据文件路径和列名
    data_file_path = data_args.train_files[0] if data_args.train_files else "./data/mobvoi_seq_monkey_general_open_corpus_small.jsonl"
    column_names = ["text"]  # 根据实际数据集结构调整
    train_dataset = load_and_process_dataset(data_file_path, tokenizer, column_names, block_size)

    # 初始化 SwanLab
    swanlab.init(project="pretrain", experiment_name="from_scratch")

    logger.info("初始化 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    # 从 checkpoint 加载
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    logger.info("开始训练")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()



if __name__ == "__main__":
    main()
