"""训练配置模块"""

from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
from vllm import SamplingParams


def create_sft_trainer(model, tokenizer, dataset):
    """创建SFT训练器"""
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

            seed = 42,                               # 固定随机种子，确保实验可重复

            report_to = "swanlab",                        # 不将训练日志报告到 WandB 等工具（如需开启改为"wandb"）
        ),
    )
    
    return trainer


def create_grpo_trainer(model, tokenizer, dataset, max_prompt_length, max_completion_length, reward_funcs):
    """创建GRPO训练器"""
    # 配置 vLLM 的采样参数（用于生成训练样本）
    vllm_sampling_params = SamplingParams(
        min_p = 0.1,                    # nucleus sampling 的截断下界
        top_p = 1.0,                    # nucleus sampling 的上限（top-p sampling）
        top_k = -1,                     # 不启用 top-k 截断（-1 表示关闭）
        seed = 3407,                    # 固定随机种子，保证生成结果可复现
        stop = [tokenizer.eos_token],   # 生成停止标志（通常是 <|endoftext|>）
        include_stop_str_in_output = True,  # 是否将 stop token 也包含在输出中
    )
    
    # 配置 GRPO（Generalized Reinforcement Preference Optimization）训练参数
    training_args = GRPOConfig(
        vllm_sampling_params = vllm_sampling_params,  # 用于生成 completion 的采样策略
        temperature = 1.0,               # 生成的多样性控制（通常设为 0.7 ~ 1.0）
        learning_rate = 5e-6,            # 训练的学习率（较小以保证稳定收敛）
        weight_decay = 0.01,             # 权重衰减，用于防止过拟合
        warmup_ratio = 0.1,              # warmup 步数占总训练步数的比例（通常为 0.05 ~ 0.1）
        lr_scheduler_type = "linear",    # 学习率调度方式为线性下降
        optim = "adamw_8bit",            # 使用 bitsandbytes 的 8bit AdamW 优化器（省显存）

        logging_steps = 1,               # 每一步打印一次日志（适合 debug）
        per_device_train_batch_size = 4, # 每张 GPU 的 batch size
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

        reward_funcs = reward_funcs,  # 自定义奖励函数列表，用于计算每条样本的得分

        args = training_args,  # 训练参数配置（使用前面定义好的 GRPOConfig 对象）

        train_dataset = dataset,  # 实际用于训练的数据集

        # 可选：如启用训练 + 验证评估，可替换为如下配置
        # train_dataset = new_dataset["train"],
        # eval_dataset = new_dataset["test"],
    )
    
    return trainer
