import logging
from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def group_texts(examples, block_size=2048):
    """
    预训练一般将文本拼接成固定长度的文本段
    这里我们取块长为 2048
    """
    # 将文本段拼接起来
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # 计算拼起来的整体长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 如果长度太长，进行分块
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # 按 block_size 进行切分
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # CLM 任务，labels 和 input 是相同的
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_function(examples, tokenizer):
    """对数据集进行 tokenize"""
    # 使用预先加载的 tokenizer 进行分词
    output = tokenizer([item for item in examples["text"]])
    return output


def load_and_process_dataset(data_file_path, tokenizer, column_names, block_size=2048):
    """加载并处理数据集"""
    # 加载数据集
    ds = load_dataset('json', data_files=data_file_path)
    
    # 对数据集进行 tokenize
    tokenized_datasets = ds.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        num_proc=10,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    # 批量处理
    lm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(examples, block_size),
        batched=True,
        num_proc=10,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
        batch_size=40000,
    )
    
    train_dataset = lm_datasets["train"]
    return train_dataset
