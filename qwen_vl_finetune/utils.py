"""
工具函数
"""

import torch


def get_gpu_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        return {
            "gpu_name": gpu_stats.name,
            "current_memory": current_memory,
            "max_memory": max_memory,
            "memory_percentage": round(current_memory / max_memory * 100, 3)
        }
    else:
        return None


def print_memory_usage():
    """打印内存使用情况"""
    memory_info = get_gpu_memory_info()
    if memory_info:
        print(f"GPU: {memory_info['gpu_name']}")
        print(f"Current memory usage: {memory_info['current_memory']} GB")
        print(f"Max memory: {memory_info['max_memory']} GB")
        print(f"Memory usage percentage: {memory_info['memory_percentage']}%")
    else:
        print("CUDA is not available")


def calculate_memory_difference(start_memory, end_memory, max_memory):
    """计算内存差异"""
    memory_used = round(end_memory - start_memory, 3)
    memory_percentage = round(memory_used / max_memory * 100, 3)
    return memory_used, memory_percentage
