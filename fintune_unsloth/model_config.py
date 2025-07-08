"""模型配置和初始化模块"""

from unsloth import FastLanguageModel
import torch


def initialize_model_and_tokenizer():
    """初始化模型和分词器"""
    max_seq_length = 2048  # 最大序列长度，可以增加以支持更长的推理轨迹
    lora_rank = 32         # LoRA 的秩，秩越大模型可能越智能，但训练和推理速度会变慢

    # 从预训练模型加载模型和分词器
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./model_download/qwen3_8B",  # 要加载的预训练模型名称
        max_seq_length=max_seq_length,        # 设置模型的最大序列长度
        load_in_4bit=False,                   # 是否以4位加载模型，对于LoRA 16位训练，设置为False
        fast_inference=False,                  # 是否启用 vLLM 快速推理,暂时有bug
        max_lora_rank=lora_rank,              # 设置 LoRA 的最大秩
        gpu_memory_utilization=0.7,           # GPU显存使用率，如果显存不足 (OOM)，可以降低此值
    )
    
    return model, tokenizer, max_seq_length, lora_rank


def setup_peft_model(model, lora_rank):
    """为模型添加 PEFT 配置"""
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
    
    return model
