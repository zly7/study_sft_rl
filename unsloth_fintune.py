from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # 最大序列长度，可以增加以支持更长的推理轨迹
lora_rank = 32         # LoRA 的秩，秩越大模型可能越智能，但训练和推理速度会变慢

# 从预训练模型加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",  # 要加载的预训练模型名称(官网版本/离线版本/unsloth版本均可)
    max_seq_length=max_seq_length,        # 设置模型的最大序列长度
    load_in_4bit=False,                   # 是否以4位加载模型，对于LoRA 16位训练，设置为False
    fast_inference=True,                  # 是否启用 vLLM 快速推理
    max_lora_rank=lora_rank,              # 设置 LoRA 的最大秩
    gpu_memory_utilization=0.7,           # GPU显存使用率，如果显存不足 (OOM)，可以降低此值
)

# 为模型添加 PEFT (Parameter-Efficient Fine-Tuning) 配置，这里使用 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # LoRA 的秩 (r)，选择任何大于0的数字！建议值为 8, 16, 32, 64, 128
    target_modules=[  # 需要应用LoRA的模块名称列表
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力机制中的查询、键、值、输出投影
        "gate_proj", "up_proj", "down_proj",     # 前馈网络中的门控、上行和下行投影
    ],
    lora_alpha=lora_rank * 2,  # LoRA 的 alpha 参数，设置为秩的2倍可以加速训练
    use_gradient_checkpointing="unsloth",  # 是否使用梯度检查点技术，"unsloth" 表示使用其优化版本以减少显存使用
    random_state=3407,                   # 随机种子，用于确保结果的可复现性
)


# 定义一些特殊的字符串标记，用于指导模型生成我们想要的格式。
# 这是一种“格式提示”或“模板化”的方法，让模型学会生成带有思考过程和最终答案的结构化输出。

#  和  用于包裹模型的“思考过程”或“解题步骤”。
reasoning_start = ""
reasoning_end   = ""

#  和  用于包裹模型给出的最终、简洁的答案。
solution_start  = ""
solution_end    = ""

# 定义系统提示（System Prompt）。这个提示会作为对话的初始指令，告诉模型它的角色和任务。
# 在这里，我们要求模型先进行思考，将过程放在和之间，
# 然后再将最终答案放在和之间。
system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""



# 创建一个自定义的聊天模板（Chat Template）。
# 聊天模板使用Jinja2语法，定义了如何将多轮对话（包含system, user, assistant等角色）格式化为单个字符串，
# 以便输入给模型进行训练或推理。
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

# 将模板中的占位符替换为我们之前定义的特定字符串。
# 这样做可以使模板适应我们自定义的格式要求。
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")

# 将我们创建的自定义聊天模板赋值给分词器（tokenizer）的chat_template属性。
# 这样，之后调用tokenizer.apply_chat_template时，就会使用这个新模板。
tokenizer.chat_template = chat_template


# 定义用于标记模型推理过程和最终解的字符串标记
reasoning_start = "<start_working_out>"     # 用于包裹模型的“思考过程”开始部分
reasoning_end   = "<end_working_out>"       # 用于包裹模型的“思考过程”结束部分
solution_start  = "<SOLUTION>"              # 用于包裹最终解答开始部分
solution_end    = "</SOLUTION>"             # 用于包裹最终解答结束部分

# 定义系统提示词，告诉模型应该如何组织回答：先推理，再给出答案
system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# 构建chat_template模板，控制如何拼接prompt
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\  # 如果第一条是system提示，拼接它并添加eos标记
        "{% set loop_messages = messages[1:] %}"\     # 剩下的消息设为循环体
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\        # 否则，插入默认system_prompt并添加eos
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\             # 遍历所有对话消息
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\               # 用户消息直接添加
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\   # assistant消息后加eos
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\  # 如果需要生成提示，添加开始思考标记
    "{% endif %}"

# 将模板中作为字符串存在的变量替换为实际变量值（避免模板中引号包住变量名）
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")

# 将chat_template应用到tokenizer（假设这是一个支持chat_template的tokenizer）
tokenizer.chat_template = chat_template

# 模拟一次tokenizer应用chat_template的过程（不进行tokenize，只展示结果）
tokenizer.apply_chat_template([
    {"role" : "user", "content" : "What is 1+1?"},
    {"role" : "assistant", "content" : f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
    {"role" : "user", "content" : "What is 2+2?"},
], tokenize = False, add_generation_prompt = True)


from datasets import load_dataset
import pandas as pd
import numpy as np

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
dataset.shape

from datasets import Dataset

dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize = False)
dataset = Dataset.from_pandas(dataset)
dataset


from trl import SFTTrainer, SFTConfig

# 创建一个有监督微调的训练器实例
trainer = SFTTrainer(
    model = model,                 # 预训练模型（如 LLaMA、Qwen、Mistral 等）
    tokenizer = tokenizer,         # 与模型匹配的 tokenizer，需支持 chat_template
    train_dataset = dataset,       # 用于训练的数据集，要求包含"text"字段

    # 训练参数配置
    args = SFTConfig(
        dataset_text_field = "text",               # 数据集中用于训练输入的字段名（通常为"text"）
        per_device_train_batch_size = 8,           # 每张 GPU 上的 batch size
        gradient_accumulation_steps = 1,           # 梯度累积步数（总有效 batch_size = 上两者相乘）

        warmup_steps = 5,                          # 学习率预热步数，避免初始过快下降
        num_train_epochs = 2,                      # 训练轮数

        learning_rate = 2e-4,                      # 初始学习率（建议长期训练用 2e-5 ~ 5e-5）
        logging_steps = 5,                         # 每 5 步打印一次日志（loss 等）

        optim = "adamw_8bit",                      # 使用 8-bit AdamW 优化器（需要 bitsandbytes 支持）
        weight_decay = 0.01,                       # 权重衰减，防止过拟合
        lr_scheduler_type = "linear",              # 线性学习率衰减策略

        seed = 3407,                               # 固定随机种子，确保实验可重复

        report_to = "swanlab",                        # 不将训练日志报告到 WandB 等工具（如需开启改为"wandb"）
    ),
)

trainer.train()


# 构建输入 prompt，选取前两条消息（通常为 system + user）
text = tokenizer.apply_chat_template(
    dataset[0]["Messages"][:2],       # 输入前两条消息：system 和 user 组成的 prompt
    tokenize = False,                 # 不进行 token 化，返回纯文本字符串
    add_generation_prompt = True,     # 在结尾添加推理开始标记（如 <start_working_out>）
)

# 使用 transformers 的流式输出工具 TextStreamer 实时打印生成内容
from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),   # 编码文本并移动到 GPU
    temperature = 0,                # 使用贪婪解码（temperature 趋近于 0）
    max_new_tokens = 1024,         # 限制生成 token 数量
    streamer = TextStreamer(tokenizer, skip_prompt = False),  # 实时打印生成结果
)

# 清理内存，防止显存泄露
del dataset
torch.cuda.empty_cache()
import gc
gc.collect()

# 加载一个数学微调数据集（HuggingFace hub 上的 DAPO-Math-17k）
from datasets import load_dataset
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")
dataset

# 查看一个样本的 prompt 和 solution 字段
dataset[0]["prompt"]
dataset[0]["solution"]

# 抽取解答函数（可定制，此处暂时原样返回）
def extract_hash_answer(text):
    # 可启用以下代码用于处理带有 "####" 分隔的答案
    # if "####" not in text: return None
    # return text.split("####")[1].strip()
    return text
extract_hash_answer(dataset[0]["solution"])

# 将原始数据格式转为 messages 格式，适配 SFTTrainer 所需格式
dataset = dataset.map(lambda x: {
    "prompt": [  # 将系统提示和用户输入转为消息格式
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["prompt"]},
    ],
    "answer": extract_hash_answer(x["solution"]),  # 答案部分处理
})
dataset[0]

# ========================
# 提取生成文本中答案部分的正则表达式匹配器
# ========================

import re

# 构造匹配结束标签 "</SOLUTION>" 和可选的 eos_token（例如 <|endoftext|>）
solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
    "(?:" + re.escape(tokenizer.eos_token) + ")?"

# 构造完整的匹配模板，用于提取推理结果中的答案部分
match_format = re.compile(
    rf"{reasoning_end}.*?"\                   # 匹配推理结束标签以及其后的内容（非贪婪）
    rf"{solution_start}(.+?){solution_end_regex}"\  # 提取 <SOLUTION> 与 </SOLUTION> 之间的内容
    rf"[\s]{{0,}}$",                          # 匹配末尾的空白
    flags = re.MULTILINE | re.DOTALL         # 多行匹配 + 点号匹配换行符
)

# 示例：验证格式匹配是否能正确提取解答部分
match_format.findall(
    "<start_working_out>Let me think!<end_working_out>"\
    f"<SOLUTION>  2  </SOLUTION>\n\n",
)
# 输出应为：["2"]

def match_format_exactly(completions, **kwargs):
    scores = []  # 用于保存每个 completion 的得分

    for completion in completions:
        score = 0
        response = completion[0]["content"]  # 获取模型输出内容（假设为 messages 列表中的第一个 assistant 回复）

        # 如果输出内容能成功匹配指定格式（即包含完整 <start_working_out>...<SOLUTION>... 标签结构）
        if match_format.search(response) is not None:
            score += 3.0  # 匹配成功得 3 分（用于奖励格式正确的输出）

        scores.append(score)  # 保存该条 completion 的得分

    return scores  # 返回所有 completion 的格式匹配得分列表

def match_format_approximately(completions, **kwargs):
    scores = []  # 存储每个 completion 的近似格式匹配得分

    for completion in completions:
        score = 0
        response = completion[0]["content"]  # 获取该条生成结果的文本内容

        # 本函数不是精确匹配整段模板，而是检查关键标签是否恰好出现一次
        # 评分标准如下（每个关键标签出现一次加 0.5 分，出现多次或漏掉则减 1 分）：

        # <start_working_out> 不需要判断，因为一般在 prompt 中已加，无需重复奖励

        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0  # 检查 <end_working_out>
        score += 0.5 if response.count(solution_start)  == 1 else -1.0  # 检查 <SOLUTION>
        score += 0.5 if response.count(solution_end)    == 1 else -1.0  # 检查 </SOLUTION>

        scores.append(score)  # 保存该条 completion 的评分结果

    return scores  # 返回所有样本的评分结果列表

def check_answer(prompts, completions, answer, **kwargs):
    # 获取原始问题（一般为 prompts 中最后一个 user 消息的内容）
    question = prompts[0][-1]["content"]

    # 提取每个 completion 的生成结果（假设为 assistant 的第一条回复）
    responses = [completion[0]["content"] for completion in completions]

    # 从每个 response 中提取 <SOLUTION> 标签内的答案（使用正则匹配）
    extracted_responses = [
        guess.group(1)  # 如果匹配成功，取出括号中的 group(1)
        if (guess := match_format.search(r)) is not None else None  # 否则为 None
        for r in responses
    ]

    scores = []  # 存储每个样本的评分结果
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)  # 若未成功提取答案，直接扣分
            continue

        # 正确答案完全一致，奖励 5 分
        if guess == true_answer:
            score += 5.0

        # 若去除空格后匹配成功，奖励略少（3.5 分）
        elif guess.strip() == true_answer.strip():
            score += 3.5

        # 否则，尝试进行“近似数值”匹配
        else:
            try:
                ratio = float(guess) / float(true_answer)  # 转换为 float 并计算比值
                if   ratio >= 0.9 and ratio <= 1.1:
                    score += 2.0  # 误差在 ±10% 内
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 1.5  # 误差在 ±20% 内
                else:
                    score -= 2.5  # 偏差太大，扣分
            except:
                score -= 4.5  # 无法转为数值（如包含文本、单位等），严重扣分

        scores.append(score)  # 记录当前样本的得分

    return scores  # 返回每个 completion 的分数

match_numbers = re.compile(
    solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)

# 全局打印控制变量，每 N 步打印一次日志（用于调试时查看部分输出）
global PRINTED_TIMES
PRINTED_TIMES = 0  # 当前已打印次数

global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5  # 每间隔多少步打印一次

# 数值匹配函数：从生成结果中提取数字，并与正确答案进行比较
def check_numbers(prompts, completions, answer, **kwargs):
    # 获取问题文本（通常为 prompts 中最后一个 user 消息）
    question = prompts[0][-1]["content"]

    # 提取模型生成的文本内容（假设每个 completion 是一个消息列表，取第一条）
    responses = [completion[0]["content"] for completion in completions]

    # 使用正则表达式 match_numbers 提取数字
    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []  # 存储得分结果

    # 控制打印调试信息（每隔 N 次打印一次，用于查看 sample 匹配结果）
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            '*'*20 + f"Question:\n{question}",  # 打印问题
            f"\nAnswer:\n{answer[0]}",           # 打印参考答案
            f"\nResponse:\n{responses[0]}",      # 打印模型生成
            f"\nExtracted:\n{extracted_responses[0]}"  # 打印提取结果
        )
    PRINTED_TIMES += 1  # 打印次数增加

    # 核心评分逻辑
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)  # 没提取出数字，直接扣分
            continue

        try:
            # 去除空格，转换为 float；guess 先去掉千位分隔符（例如 123,456）
            true_answer = float(true_answer.strip())
            guess = float(guess.strip().replace(",", ""))

            # 如果完全数值一致，得分 3.5；否则扣分
            scores.append(3.5 if guess == true_answer else -1.5)

        except:
            scores.append(0)  # 解析失败不给分也不扣分
            continue

    return scores  # 返回所有样本的得分列表

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

import numpy as np
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

# 配置 vLLM 的采样参数（用于生成训练样本）
from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0.1,                    # nucleus sampling 的截断下界
    top_p = 1.0,                    # nucleus sampling 的上限（top-p sampling）
    top_k = -1,                     # 不启用 top-k 截断（-1 表示关闭）
    seed = 3407,                    # 固定随机种子，保证生成结果可复现
    stop = [tokenizer.eos_token],   # 生成停止标志（通常是 <|endoftext|>）
    include_stop_str_in_output = True,  # 是否将 stop token 也包含在输出中
)
# 配置 GRPO（Generalized Reinforcement Preference Optimization）训练参数
from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,  # 用于生成 completion 的采样策略
    temperature = 1.0,               # 生成的多样性控制（通常设为 0.7 ~ 1.0）
    learning_rate = 5e-6,            # 训练的学习率（较小以保证稳定收敛）
    weight_decay = 0.01,             # 权重衰减，用于防止过拟合
    warmup_ratio = 0.1,              # warmup 步数占总训练步数的比例（通常为 0.05 ~ 0.1）
    lr_scheduler_type = "linear",    # 学习率调度方式为线性下降
    optim = "adamw_8bit",            # 使用 bitsandbytes 的 8bit AdamW 优化器（省显存）

    logging_steps = 1,               # 每一步打印一次日志（适合 debug）
    per_device_train_batch_size = 1, # 每张 GPU 的 batch size
    gradient_accumulation_steps = 1, # 梯度累积步数（设为 4 可等效 batch size=4）

    num_generations = 4,             # 每个 prompt 生成多少个 response（越多越好，但显存消耗更大）
    max_prompt_length = max_prompt_length,         # 提示 token 最大长度（前面计算得出）
    max_completion_length = max_completion_length, # 回答 token 最大长度（确保总长不超过模型限制）

    # num_train_epochs = 1,          # 可选参数：训练轮次，设置后可按 epoch 控制训练终止
    max_steps = 100,                 # 最大训练步数（适合调试用，正式训练时可加大）
    save_steps = 100,                # 每 100 步保存一次模型（训练短时可以不存）

    report_to = "swanlab",              # 不上传训练日志（可设置为 "wandb" 使用可视化工具）
    output_dir = "outputs",          # 模型输出路径
)
    
# 创建 GRPOTrainer 实例，用于执行强化学习式偏好优化训练（Generalized RPO）
trainer = GRPOTrainer(
    model = model,  # 需要训练的语言模型（必须支持 causal LM 格式，如 GPT、LLaMA 等）

    processing_class = tokenizer,  # 用于生成 prompt、解码 response 的 tokenizer（需支持 chat_template）

    reward_funcs = [  # 自定义奖励函数列表，用于计算每条样本的得分
        match_format_exactly,       # 检查是否严格符合 <start_working_out>...<SOLUTION> 格式，匹配得分高
        match_format_approximately, # 检查是否大致有格式标签，宽松评分
        check_answer,               # 与参考答案对比，进行精确匹配、模糊匹配、近似数值匹配
        check_numbers,              # 提取数值进行比较（用于数学题）
    ],

    args = training_args,  # 训练参数配置（使用前面定义好的 GRPOConfig 对象）

    train_dataset = dataset,  # 实际用于训练的数据集

    # 可选：如启用训练 + 验证评估，可替换为如下配置
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)

# Step 1: 构造初始输入并生成输出（不加载 LoRA）
text = "What is the sqrt of 101?"

from vllm import SamplingParams

# 设置生成参数（适度随机，限制长度）
sampling_params = SamplingParams(
    temperature = 1.0,
    top_k = 50,
    max_tokens = 1024,
)

# 使用基础模型进行快速推理（未加载 LoRA）
output = model.fast_generate(
    [text],  # 输入为单条文本
    sampling_params=sampling_params,
    lora_request=None,  # 不加载任何 LoRA 权重
)[0].outputs[0].text

print("Original model output:\n", output)

# Step 2: 保存 GRPO 微调得到的 LoRA 权重
model.save_lora("grpo_saved_lora")

# Step 3: 检查保存的 safetensors 权重不为全零
from safetensors import safe_open

with safe_open("grpo_saved_lora/adapter_model.safetensors", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum() / tensor.numel()
        assert n_zeros.item() != 1.0, f"Tensor {key} is entirely zero!"

print("LoRA weights saved and verified.")

# Step 4: 构造消息格式输入并应用 tokenizer 的 chat_template
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the sqrt of 101?"},
]

# 构造对话式文本输入，用于 instruct-style 推理
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # 添加推理起始标记
    tokenize=False               # 返回字符串而非 token ids
)

# Step 5: 加载微调后的 LoRA 并生成输出
sampling_params = SamplingParams(
    temperature=1.0,
    top_k=50,
    max_tokens=2048,
)

# 加载 LoRA 并进行推理
output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),  # 加载 GRPO 微调权重
)[0].outputs[0].text

print("GRPO-tuned model output:\n", output)

# 合并为 16bit 权重并保存本地（适用于全精度部署）
if False:
    model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
# 合并为 16bit 权重并上传至 HuggingFace Hub（需填写 token）
if False:
    model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit", token="")
# 合并为 4bit 量化权重并保存本地（适用于节省显存的部署场景，如 QLoRA 推理）
if False:
    model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
# 合并为 4bit 量化权重并上传至 HuggingFace Hub（需填写 token）
if False:
    model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_4bit", token="")
# 仅保存 LoRA Adapter 参数（适用于只上传微调部分以节省空间或用于 PEFT 加载）
if False:
    model.save_pretrained_merged("model", tokenizer, save_method="lora")
# 仅上传 LoRA Adapter 参数至 HuggingFace Hub（需填写 token）
if False:
    model.push_to_hub_merged("hf/model", tokenizer, save_method="lora", token="")