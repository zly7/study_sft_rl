# PreTrain ZLY 预训练模块

这个模块包含所有预训练相关的代码，为了更好的代码组织和后续的扩展。

## 模块结构

```
pretrain_zly/
├── __init__.py          # 模块初始化文件
├── config.py            # 配置参数定义
├── model_utils.py       # 模型相关工具函数
├── data_processing.py   # 数据处理相关函数
├── logger_setup.py      # 日志配置
├── pretrain.py          # 预训练主逻辑
└── README.md            # 本文件
```

## 功能说明

### config.py
- `ModelArguments`: 模型相关参数配置
- `DataTrainingArguments`: 训练数据相关参数配置
- `parse_arguments()`: 解析命令行参数

### model_utils.py
- `check_checkpoint()`: 检查训练checkpoint
- `initialize_model()`: 初始化模型
- `initialize_tokenizer()`: 初始化tokenizer

### data_processing.py
- `group_texts()`: 将文本拼接成固定长度的文本段
- `tokenize_function()`: 对数据集进行tokenize
- `load_and_process_dataset()`: 加载并处理数据集

### logger_setup.py
- `setup_logging()`: 设置日志配置

### pretrain.py
- `main()`: 预训练主函数，包含完整的预训练流程

## 使用方法

从项目根目录运行：
```bash
python pretrain.py
```

或者直接导入模块使用：
```python
from pretrain_zly import main
main()
```
