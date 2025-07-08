"""
使用重构后的qwen_vl_finetune包的示例

这个文件演示了如何使用重构后的代码来进行Qwen VL模型的微调。
原始的qwen-vl-finetune.py文件中的所有功能都被整合到了qwen_vl_finetune包中。
"""

# 导入重构后的包
from qwen_vl_finetune import (
    main_training_pipeline,
    main_inference_pipeline,
    load_model_and_tokenizer,
    setup_peft_model,
    load_latex_dataset,
    run_inference,
    save_model,
    save_merged_model
)


def run_complete_training():
    """运行完整的训练流程"""
    print("=== 开始完整的训练流程 ===")
    
    # 运行主要的训练管道
    model, tokenizer, dataset = main_training_pipeline()
    
    # 保存模型
    print("\n=== 保存模型 ===")
    save_model(model, tokenizer, "lora_model")
    print("模型已保存到 lora_model 目录")
    
    return model, tokenizer, dataset


def run_inference_only():
    """仅运行推理"""
    print("=== 仅运行推理 ===")
    
    # 运行推理管道
    model, tokenizer = main_inference_pipeline()
    
    return model, tokenizer


def run_custom_workflow():
    """运行自定义工作流"""
    print("=== 运行自定义工作流 ===")
    
    # 1. 加载模型
    print("1. 加载模型...")
    model, tokenizer = load_model_and_tokenizer(model_name="model_download/qwen2.5_vl_7b")
    model = setup_peft_model(model)
    
    # 2. 加载数据集
    print("2. 加载数据集...")
    dataset = load_latex_dataset()
    
    # 3. 运行推理测试
    print("3. 运行推理测试...")
    image = dataset[0]["image"]
    instruction = "Write the LaTeX representation for this image."
    run_inference(model, tokenizer, image, instruction)
    
    return model, tokenizer, dataset


if __name__ == "__main__":
    # 选择要运行的模式
    mode = "custom"  # 可以是 "training", "inference", "custom"
    
    if mode == "training":
        model, tokenizer, dataset = run_complete_training()
    elif mode == "inference":
        model, tokenizer = run_inference_only()
    elif mode == "custom":
        model, tokenizer, dataset = run_custom_workflow()
    
    print("\n=== 完成 ===")


# 以下是原始文件中的所有功能，现在都可以通过包导入使用：

# 原始代码：
# from unsloth import FastVisionModel
# import torch
# 现在可以通过：
# from qwen_vl_finetune import load_model_and_tokenizer, setup_peft_model

# 原始代码：
# from datasets import load_dataset
# dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")
# 现在可以通过：
# from qwen_vl_finetune import load_latex_dataset
# dataset = load_latex_dataset()

# 原始代码：
# from IPython.display import display, Math, Latex
# latex = dataset[2]["text"]
# display(Math(latex))
# 现在可以通过：
# from qwen_vl_finetune import render_latex
# render_latex(latex)

# 原始代码：
# def convert_to_conversation(sample):
#     conversation = [...]
#     return { "messages" : conversation }
# 现在可以通过：
# from qwen_vl_finetune import convert_to_conversation, convert_dataset_to_conversations

# 原始代码：
# from unsloth.trainer import UnslothVisionDataCollator
# from trl import SFTTrainer, SFTConfig
# trainer = SFTTrainer(...)
# 现在可以通过：
# from qwen_vl_finetune import setup_trainer, train_model

# 原始代码：
# FastVisionModel.for_inference(model)
# messages = [...]
# inputs = tokenizer(...)
# _ = model.generate(...)
# 现在可以通过：
# from qwen_vl_finetune import run_inference

# 原始代码：
# model.save_pretrained("lora_model")
# 现在可以通过：
# from qwen_vl_finetune import save_model

# 原始代码：
# model.save_pretrained_merged("unsloth_finetune", tokenizer,)
# 现在可以通过：
# from qwen_vl_finetune import save_merged_model
