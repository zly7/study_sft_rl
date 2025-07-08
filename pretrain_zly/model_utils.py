import logging
import os
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


def check_checkpoint(training_args):
    """检查 checkpoint"""
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        # 使用 transformers 自带的 get_last_checkpoint 自动检测
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出路径 ({training_args.output_dir}) 非空 "
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"从 {last_checkpoint}恢复训练"
            )
    return last_checkpoint


def initialize_model(model_args):
    """初始化模型"""
    if model_args.config_name is not None:
        # from scrach
        config = AutoConfig.from_pretrained(model_args.config_name)
        logger.warning("你正在从零初始化一个模型")
        logger.info(f"模型参数配置地址：{model_args.config_name}")
        logger.info(f"模型参数：{config}")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"预训练一个新模型 - Total size={n_params/2**20:.2f}M params")
    elif model_args.model_name_or_path is not None:
        logger.warning("你正在初始化一个预训练模型")
        logger.info(f"模型参数地址：{model_args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"继承一个预训练模型 - Total size={n_params/2**20:.2f}M params")
    else:
        logger.error("config_name 和 model_name_or_path 不能均为空")
        raise ValueError("config_name 和 model_name_or_path 不能均为空")
    
    return model


def initialize_tokenizer(model_name_or_path):
    """初始化 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer
