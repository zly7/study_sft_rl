"""简化版本的 unsloth 微调脚本 - 使用重构后的模块"""

from fintune_unsloth.main import main
import os

# 设置 Hugging Face 缓存路径
os.environ['HF_HOME'] = '/data2/zly/.cache'
os.environ['TRANSFORMERS_CACHE'] = '/data2/zly/.cache/transformers'
os.environ['HF_DATASETS_CACHE'] = '/data2/zly/.cache/datasets'

# 设置 PyTorch 缓存路径
os.environ['TORCH_HOME'] = '/data2/zly/.cache/torch'

# 设置 Datasets 缓存路径
os.environ['DATASETS_CACHE'] = '/data2/zly/.cache/datasets'

# 确保缓存目录存在
cache_dirs = [
    '/data2/zly/.cache', 
    '/data2/zly/.cache/transformers', 
    '/data2/zly/.cache/datasets',
    '/data2/zly/.cache/torch'
]
for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)
    
if __name__ == "__main__":
    # 运行完整的训练流程
    main()
