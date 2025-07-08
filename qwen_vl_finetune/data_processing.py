"""
数据处理相关功能
"""

from datasets import load_dataset
from IPython.display import display, Math, Latex


def load_latex_dataset(dataset_name="unsloth/LaTeX_OCR", split="train"):
    """加载LaTeX OCR数据集"""
    dataset = load_dataset(dataset_name, split = split)
    return dataset


def display_dataset_sample(dataset, index=2):
    """显示数据集样本"""
    print("Dataset overview:")
    print(dataset)
    
    print(f"\nImage at index {index}:")
    print(dataset[index]["image"])
    
    print(f"\nText at index {index}:")
    print(dataset[index]["text"])
    
    return dataset[index]


def render_latex(latex_text):
    """渲染LaTeX数学公式"""
    display(Math(latex_text))


def convert_to_conversation(sample, instruction="Write the LaTeX representation for this image."):
    """将样本转换为对话格式"""
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["text"]} ]
        },
    ]
    return { "messages" : conversation }


def convert_dataset_to_conversations(dataset, instruction="Write the LaTeX representation for this image."):
    """将整个数据集转换为对话格式"""
    converted_dataset = [convert_to_conversation(sample, instruction) for sample in dataset]
    return converted_dataset


def show_conversation_example(converted_dataset, index=0):
    """显示对话格式示例"""
    print(f"Conversation example at index {index}:")
    print(converted_dataset[index])
    return converted_dataset[index]
