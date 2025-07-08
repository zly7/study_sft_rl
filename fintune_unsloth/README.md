# Fintune Unsloth 包结构说明

本包将原始的 `unsloth_fintune.py` 代码按功能模块进行了重新组织，提高了代码的可维护性和可重用性。

## 模块结构

### 1. `model_config.py` - 模型配置和初始化
- `initialize_model_and_tokenizer()`: 初始化模型和分词器
- `setup_peft_model()`: 设置PEFT（LoRA）配置

### 2. `template_utils.py` - 模板和格式化工具
- `setup_custom_tokens()`: 定义自定义标记
- `create_system_prompt()`: 创建系统提示
- `create_chat_template()`: 创建聊天模板
- `setup_tokenizer_template()`: 设置分词器模板
- `test_chat_template()`: 测试聊天模板

### 3. `data_processor.py` - 数据处理
- `load_and_process_openmath_dataset()`: 加载和处理OpenMath数据集
- `load_and_process_dapo_math_dataset()`: 加载和处理DAPO-Math数据集
- `prepare_tokenized_dataset()`: 准备分词后的数据集

### 4. `training.py` - 训练配置
- `create_sft_trainer()`: 创建SFT训练器
- `create_grpo_trainer()`: 创建GRPO训练器

### 5. `evaluation.py` - 评估和奖励函数
- `setup_regex_patterns()`: 设置正则表达式模式
- `match_format_exactly()`: 精确格式匹配奖励函数
- `match_format_approximately()`: 近似格式匹配奖励函数
- `check_answer()`: 答案检查奖励函数
- `check_numbers()`: 数值检查奖励函数
- `create_reward_functions()`: 创建奖励函数列表

### 6. `inference.py` - 推理和模型保存
- `basic_inference()`: 基础推理示例
- `cleanup_memory()`: 清理内存
- `advanced_inference_comparison()`: 高级推理对比
- `save_model_options()`: 模型保存选项

### 7. `main.py` - 主程序
- `main()`: 整合所有功能的完整训练流程

## 使用方法

### 方法1：直接运行主程序
```python
from fintune_unsloth.main import main
main()
```

### 方法2：使用简化脚本
```bash
python unsloth_fintune_refactored.py
```

### 方法3：分模块使用
```python
from fintune_unsloth import (
    initialize_model_and_tokenizer,
    setup_peft_model,
    setup_tokenizer_template,
    load_and_process_openmath_dataset,
    create_sft_trainer
)

# 只使用需要的功能
model, tokenizer, max_seq_length, lora_rank = initialize_model_and_tokenizer()
model = setup_peft_model(model, lora_rank)
# ... 其他操作
```

## 重构优势

1. **模块化**: 将功能按逻辑分组，便于维护和测试
2. **可重用性**: 每个函数都可以单独使用
3. **可扩展性**: 容易添加新的功能模块
4. **可读性**: 代码结构更清晰，便于理解
5. **调试友好**: 可以单独测试每个模块的功能

## 注意事项

- 原始代码的逻辑和参数保持不变，只是重新组织了结构
- 所有的错误和不完善之处都被保留，没有进行修复
- 可以根据需要选择使用部分功能或完整流程
