"""
模型设置和配置相关功能
"""

from unsloth import FastVisionModel
import torch


# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth


def load_model_and_tokenizer(model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"):
    """加载模型和tokenizer"""
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    return model, tokenizer


def setup_peft_model(model):
    """设置PEFT模型"""
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
    return model


def load_lora_model(model_path, load_in_4bit=True):
    """加载LoRA模型"""
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = load_in_4bit, # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model) # Enable for inference!
    return model, tokenizer


def save_model(model, tokenizer, save_path="lora_model"):
    """保存模型"""
    model.save_pretrained(save_path)  # Local saving
    tokenizer.save_pretrained(save_path)


def save_merged_model(model, tokenizer, save_path="unsloth_finetune"):
    """保存合并的模型到float16"""
    model.save_pretrained_merged(save_path, tokenizer,)


def push_to_hub_merged(model, tokenizer, hub_path, token):
    """推送合并的模型到Hugging Face Hub"""
    model.push_to_hub_merged(hub_path, tokenizer, token = token)


def show_memory_stats():
    """显示当前内存统计"""
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return start_gpu_memory, max_memory
