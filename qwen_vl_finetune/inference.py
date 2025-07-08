"""
推理相关功能
"""

import torch
from unsloth import FastVisionModel
from transformers import TextStreamer


def prepare_messages(instruction, image_first=True):
    """准备消息格式"""
    if image_first:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
    else:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image"}
            ]}
        ]
    return messages


def prepare_inputs(tokenizer, image, messages):
    """准备输入"""
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    return inputs


def generate_response(model, tokenizer, inputs, max_new_tokens=128, temperature=1.5, min_p=0.1, use_streamer=True):
    """生成响应"""
    FastVisionModel.for_inference(model) # Enable for inference!
    
    if use_streamer:
        text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = max_new_tokens,
                           use_cache = True, temperature = temperature, min_p = min_p)
    else:
        outputs = model.generate(**inputs, max_new_tokens = max_new_tokens,
                                use_cache = True, temperature = temperature, min_p = min_p)
        return outputs


def run_inference(model, tokenizer, image, instruction="Write the LaTeX representation for this image.", 
                 max_new_tokens=128, temperature=1.5, min_p=0.1, use_streamer=True):
    """运行完整的推理流程"""
    messages = prepare_messages(instruction, image_first=True)
    inputs = prepare_inputs(tokenizer, image, messages)
    return generate_response(model, tokenizer, inputs, max_new_tokens, temperature, min_p, use_streamer)
