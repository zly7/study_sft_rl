"""推理和模型保存模块"""

import torch
import gc
from transformers import TextStreamer
from vllm import SamplingParams
from safetensors import safe_open


def basic_inference(model, tokenizer, dataset):
    """基础推理示例"""
    # 构建输入 prompt，选取前两条消息（通常为 system + user）
    text = tokenizer.apply_chat_template(
        dataset[0]["Messages"][:2],       # 输入前两条消息：system 和 user 组成的 prompt
        tokenize = False,                 # 不进行 token 化，返回纯文本字符串
        add_generation_prompt = True,     # 在结尾添加推理开始标记（如 <start_working_out>）
    )

    # 使用 transformers 的流式输出工具 TextStreamer 实时打印生成内容
    _ = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),   # 编码文本并移动到 GPU
        temperature = 0.3,                
        max_new_tokens = 1024,         # 限制生成 token 数量
        streamer = TextStreamer(tokenizer, skip_prompt = False),  # 实时打印生成结果
    )


def cleanup_memory():
    """清理内存"""
    # 清理内存，防止显存泄露
    torch.cuda.empty_cache()
    gc.collect()


def advanced_inference_comparison(model, tokenizer, system_prompt):
    """高级推理对比（GRPO前后）"""
    # Step 1: 构造初始输入并生成输出（不加载 LoRA）
    text = "What is the sqrt of 101?"

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


def save_model_options(model, tokenizer):
    """模型保存选项"""
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
