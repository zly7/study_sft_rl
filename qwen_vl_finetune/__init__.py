"""
Qwen VL 微调工具包

此包包含了使用Unsloth进行Qwen2.5-VL模型微调的完整流程，
包括模型设置、数据处理、训练和推理。
"""

from .model_setup import (
    fourbit_models,
    load_model_and_tokenizer,
    setup_peft_model,
    load_lora_model,
    save_model,
    save_merged_model,
    push_to_hub_merged,
    show_memory_stats
)

from .data_processing import (
    load_latex_dataset,
    display_dataset_sample,
    render_latex,
    convert_to_conversation,
    convert_dataset_to_conversations,
    show_conversation_example
)

from .training import (
    setup_trainer,
    train_model,
    show_training_stats
)

from .inference import (
    prepare_messages,
    prepare_inputs,
    generate_response,
    run_inference
)

from .utils import (
    get_gpu_memory_info,
    print_memory_usage,
    calculate_memory_difference
)

from .main import (
    main_training_pipeline,
    main_inference_pipeline
)

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    # Model setup
    "fourbit_models",
    "load_model_and_tokenizer",
    "setup_peft_model",
    "load_lora_model",
    "save_model",
    "save_merged_model",
    "push_to_hub_merged",
    "show_memory_stats",
    
    # Data processing
    "load_latex_dataset",
    "display_dataset_sample",
    "render_latex",
    "convert_to_conversation",
    "convert_dataset_to_conversations",
    "show_conversation_example",
    
    # Training
    "setup_trainer",
    "train_model",
    "show_training_stats",
    
    # Inference
    "prepare_messages",
    "prepare_inputs",
    "generate_response",
    "run_inference",
    
    # Utils
    "get_gpu_memory_info",
    "print_memory_usage",
    "calculate_memory_difference",
    
    # Main pipelines
    "main_training_pipeline",
    "main_inference_pipeline",
]