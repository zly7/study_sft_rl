# Qwen VL 微调工具包重构说明

## 重构概述

原始的 `qwen-vl-finetune.py` 文件包含了所有的代码，现在已经重构为一个结构化的包 `qwen_vl_finetune`，按照功能模块进行了合理的划分。

## 包结构

```
qwen_vl_finetune/
├── __init__.py              # 包初始化和公共接口
├── model_setup.py           # 模型加载和配置
├── data_processing.py       # 数据集处理和转换
├── training.py              # 训练相关功能
├── inference.py             # 推理相关功能
├── utils.py                 # 工具函数
└── main.py                  # 主要的训练和推理流程
```

## 模块功能说明

### 1. model_setup.py
- 支持的4bit量化模型列表
- 模型和tokenizer加载
- PEFT模型设置
- LoRA模型加载
- 模型保存功能
- 内存统计

### 2. data_processing.py
- LaTeX OCR数据集加载
- 数据集样本显示
- LaTeX公式渲染
- 对话格式转换
- 数据集格式转换

### 3. training.py
- 训练器设置
- 模型训练
- 训练统计显示

### 4. inference.py
- 消息格式准备
- 输入准备
- 响应生成
- 完整推理流程

### 5. utils.py
- GPU内存信息获取
- 内存使用情况打印
- 内存差异计算

### 6. main.py
- 主要训练流程
- 主要推理流程

## 使用方法

### 方法1：使用主要流程函数

```python
from qwen_vl_finetune import main_training_pipeline, main_inference_pipeline

# 运行完整训练流程
model, tokenizer, dataset = main_training_pipeline()

# 运行推理流程
model, tokenizer = main_inference_pipeline()
```

### 方法2：使用单独的功能模块

```python
from qwen_vl_finetune import (
    load_model_and_tokenizer,
    setup_peft_model,
    load_latex_dataset,
    convert_dataset_to_conversations,
    setup_trainer,
    train_model,
    run_inference
)

# 加载模型
model, tokenizer = load_model_and_tokenizer()
model = setup_peft_model(model)

# 处理数据
dataset = load_latex_dataset()
converted_dataset = convert_dataset_to_conversations(dataset)

# 训练
trainer = setup_trainer(model, tokenizer, converted_dataset)
trainer_stats = train_model(trainer)

# 推理
image = dataset[0]["image"]
run_inference(model, tokenizer, image)
```

### 方法3：使用示例文件

运行 `qwen_vl_finetune_refactored.py` 文件，它包含了完整的使用示例。

## 原始代码映射

所有原始 `qwen-vl-finetune.py` 文件中的功能都被保留并映射到新的包结构中：

- 模型加载 → `model_setup.py`
- 数据处理 → `data_processing.py`
- 训练逻辑 → `training.py`
- 推理逻辑 → `inference.py`
- 工具函数 → `utils.py`
- 主流程 → `main.py`

## 优势

1. **模块化**: 代码按功能分离，易于维护和扩展
2. **可重用性**: 每个模块都可以独立使用
3. **可读性**: 代码结构清晰，易于理解
4. **可测试性**: 每个模块都可以单独测试
5. **一致性**: 统一的接口和使用方式

## 注意事项

- 原始代码的所有功能都被保留，包括任何可能的错误
- 只进行了代码重构和模块化，没有修改任何核心逻辑
- 所有原有的参数和配置都保持不变
