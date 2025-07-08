"""
主要的训练和推理流程
"""

from .model_setup import load_model_and_tokenizer, setup_peft_model, show_memory_stats
from .data_processing import (
    load_latex_dataset, 
    display_dataset_sample, 
    render_latex,
    convert_dataset_to_conversations,
    show_conversation_example
)
from .training import setup_trainer, train_model, show_training_stats
from .inference import run_inference
from .utils import print_memory_usage


def main_training_pipeline():
    """主要的训练流程"""
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # 设置PEFT模型
    model = setup_peft_model(model)
    
    # 加载数据集
    dataset = load_latex_dataset()
    
    # 显示数据集样本
    sample = display_dataset_sample(dataset)
    
    # 渲染LaTeX
    latex = dataset[2]["text"]
    render_latex(latex)
    
    # 转换数据集格式
    instruction = "Write the LaTeX representation for this image."
    converted_dataset = convert_dataset_to_conversations(dataset, instruction)
    
    # 显示对话示例
    show_conversation_example(converted_dataset)
    
    # 训练前推理测试
    print("\n=== 训练前推理测试 ===")
    image = dataset[2]["image"]
    run_inference(model, tokenizer, image, instruction)
    
    # 显示内存统计
    start_gpu_memory, max_memory = show_memory_stats()
    
    # 设置训练器
    trainer = setup_trainer(model, tokenizer, converted_dataset)
    
    # 训练模型
    trainer_stats = train_model(trainer)
    
    # 显示训练统计
    show_training_stats(trainer_stats, start_gpu_memory, max_memory)
    
    # 训练后推理测试
    print("\n=== 训练后推理测试 ===")
    run_inference(model, tokenizer, image, instruction)
    
    return model, tokenizer, dataset


def main_inference_pipeline(model_path=None):
    """主要的推理流程"""
    if model_path:
        # 加载保存的模型
        from .model_setup import load_lora_model
        model, tokenizer = load_lora_model(model_path)
    else:
        # 使用当前模型
        model, tokenizer = load_model_and_tokenizer()
        model = setup_peft_model(model)
    
    # 加载数据集进行测试
    dataset = load_latex_dataset()
    
    # 推理测试
    image = dataset[0]["image"]
    instruction = "Write the LaTeX representation for this image."
    
    print("=== 推理测试 ===")
    run_inference(model, tokenizer, image, instruction)
    
    return model, tokenizer
