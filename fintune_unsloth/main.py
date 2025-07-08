"""主程序 - 整合所有功能模块的完整训练流程"""

from .model_config import initialize_model_and_tokenizer, setup_peft_model
from .template_utils import setup_tokenizer_template, test_chat_template, view_dapo_dataset_examples, view_processed_dataset_info
from .data_processor import load_and_process_openmath_dataset, load_and_process_dapo_math_dataset, prepare_tokenized_dataset
from .training import create_sft_trainer, create_grpo_trainer
from .evaluation import create_reward_functions
from .inference import basic_inference, cleanup_memory, advanced_inference_comparison, save_model_options


def main():
    """主训练流程"""
    # Step 1: 初始化模型和分词器
    print("初始化模型和分词器...")
    model, tokenizer, max_seq_length, lora_rank = initialize_model_and_tokenizer()
    
    # Step 2: 设置PEFT模型
    print("设置PEFT模型...")
    model = setup_peft_model(model, lora_rank)
    
    # Step 3: 设置分词器模板
    print("设置分词器模板...")
    tokenizer, reasoning_start, reasoning_end, solution_start, solution_end, system_prompt = setup_tokenizer_template(tokenizer)
    
    # Step 4: 测试聊天模板
    print("测试聊天模板...")
    template_result = test_chat_template(tokenizer, reasoning_start, reasoning_end, solution_start, solution_end)
    print("模板测试结果:", template_result)
    
    # Step 5: 加载和处理OpenMath数据集
    print("加载和处理OpenMath数据集...")
    openmath_dataset = load_and_process_openmath_dataset(
        tokenizer, max_seq_length, reasoning_start, reasoning_end, 
        solution_start, solution_end, system_prompt
    )
    print(f"OpenMath数据集大小: {openmath_dataset.shape}")
    
    # Step 6: SFT训练
    print("开始SFT训练...")
    sft_trainer = create_sft_trainer(model, tokenizer, openmath_dataset)
    sft_trainer.train()
    
    # Step 7: 基础推理测试
    print("进行基础推理测试...")
    basic_inference(model, tokenizer, openmath_dataset)
    
    # Step 8: 清理内存
    print("清理内存...")
    del openmath_dataset
    cleanup_memory()
    
    # Step 9: 加载DAPO数学数据集
    print("加载DAPO数学数据集...")
    dapo_dataset = load_and_process_dapo_math_dataset(system_prompt)
    print(f"DAPO数据集大小: {len(dapo_dataset)}")
    
    # 查看DAPO数据集示例
    view_dapo_dataset_examples(dapo_dataset, tokenizer, max_seq_length)
    
    # Step 10: 准备分词后的数据集
    print("准备分词后的数据集...")
    processed_dataset, max_prompt_length, max_completion_length = prepare_tokenized_dataset(
        dapo_dataset, tokenizer, max_seq_length
    )
    
    # 查看处理后的数据集信息
    view_processed_dataset_info(
        processed_dataset, tokenizer, max_prompt_length, 
        max_completion_length, max_seq_length
    )
    
    # Step 11: 创建奖励函数
    print("创建奖励函数...")
    reward_funcs = create_reward_functions(tokenizer, reasoning_end, solution_start, solution_end)
    
    # Step 12: GRPO训练
    print("开始GRPO训练...")
    grpo_trainer = create_grpo_trainer(
        model, tokenizer, processed_dataset, 
        max_prompt_length, max_completion_length, reward_funcs
    )
    grpo_trainer.train()
    
    # Step 13: 高级推理对比
    print("进行高级推理对比...")
    advanced_inference_comparison(model, tokenizer, system_prompt)
    
    # Step 14: 模型保存选项
    print("模型保存选项...")
    save_model_options(model, tokenizer)
    
    print("训练流程完成！")


