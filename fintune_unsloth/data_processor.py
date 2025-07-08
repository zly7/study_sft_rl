"""数据处理模块"""

from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np


def load_and_process_openmath_dataset(tokenizer, max_seq_length, reasoning_start, reasoning_end, solution_start, solution_end, system_prompt):
    """加载和处理OpenMath数据集"""
    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
    dataset = dataset.to_pandas()[
        ["expected_answer", "problem", "generated_solution"]
    ]

    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors = "coerce").notnull()
    dataset = dataset.iloc[np.where(is_number)[0]]

    def format_dataset(x):
        # 从输入数据中提取期望答案和问题内容
        expected_answer = x["expected_answer"]
        problem = x["problem"]

        # 获取模型生成的推理内容，并移除旧格式标签 <think> 和 </think>
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")

        # 去除推理内容左右两端的换行符和空格
        thoughts = thoughts.strip()

        # 按照自定义格式拼接推理部分和答案部分，插入标记标签
        final_prompt = (
            reasoning_start + thoughts + reasoning_end +
            solution_start + expected_answer + solution_end
        )

        # 构造格式化后的多轮对话列表，用于微调或测试对话模型
        return [
            {"role": "system",    "content": system_prompt},  # 系统提示词，指导模型输出格式
            {"role": "user",      "content": problem},        # 用户输入的问题
            {"role": "assistant", "content": final_prompt},   # 模型的回复，包含推理过程和答案
        ]

    # 将整个数据集按行应用格式化函数，生成 Messages 字段，适用于对话类微调
    dataset["Messages"] = dataset.apply(format_dataset, axis=1)

    tokenizer.apply_chat_template(dataset["Messages"][0], tokenize = False)

    dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))

    dataset = dataset.loc[dataset["N"] <= max_seq_length/2].copy()

    dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize = False)
    dataset = Dataset.from_pandas(dataset)
    
    return dataset


def load_and_process_dapo_math_dataset(system_prompt):
    """加载和处理DAPO-Math数据集"""
    # 加载一个数学微调数据集（HuggingFace hub 上的 DAPO-Math-17k）
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")

    # 抽取解答函数（可定制，此处暂时原样返回）
    def extract_hash_answer(text):
        # 可启用以下代码用于处理带有 "####" 分隔的答案
        # if "####" not in text: return None
        # return text.split("####")[1].strip()
        return text
    # 这个是原本的数据集的格式就是带有这个 solution 和 prompt 字段的
    # 将原始数据格式转为 messages 格式，适配 SFTTrainer 所需格式
    dataset = dataset.map(lambda x: {
        "prompt": [  # 将系统提示和用户输入转为消息格式
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),  # 答案部分处理
    })
    
    return dataset


def prepare_tokenized_dataset(dataset, tokenizer, max_seq_length):
    """准备分词后的数据集"""
    # 对数据集中的每条样本，应用 tokenizer 的 chat 模板并进行分词
    tokenized = dataset.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"],                        # 输入为 prompt 字段，即消息列表 [{"role": ..., "content": ...}, ...]
                add_generation_prompt=True,         # 添加生成提示符（如 <start_working_out>），用于 instruct-style 生成任务
                tokenize=True                       # 返回 token id 列表，而非字符串
            )
        },
        batched=True,  # 启用批处理，提高 map 的效率
    )

    # 打印第一个样本的解码结果（从 token id 转换回字符串），用于验证模板和 tokenization 是否正确
    print(tokenizer.decode(tokenized[0]["tokens"]))

    # 为每条样本添加 token 序列长度字段 "L"，便于后续长度分布分析
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

    # 计算 token 长度的 90% 分位数，作为训练时的最大 token 限制（防止极端长样本导致 OOM）
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print("Max Length = ", maximum_length)

    # 过滤掉 token 长度超过最大长度阈值的样本，仅保留较短的 90% 样本
    dataset = dataset.select(
        np.where(np.array(tokenized["L"]) <= maximum_length)[0]
    )

    # 删除中间变量 tokenized，释放内存
    del tokenized

    # 计算提示长度上限（加1是保险措施，防止边界问题）
    max_prompt_length = maximum_length + 1  # +1 是为了避免 max_length 截断时误伤
    max_completion_length = max_seq_length - max_prompt_length  # 剩余 token 用于生成
    
    return dataset, max_prompt_length, max_completion_length
