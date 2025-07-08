"""模板和格式化工具模块"""


def view_dapo_dataset_examples(dapo_dataset, tokenizer=None, max_seq_length=None):
    """查看DAPO数据集示例
    
    Args:
        dapo_dataset: DAPO数据集
        tokenizer: 分词器（可选，用于显示分词信息）
        max_seq_length: 最大序列长度（可选）
    """
    print("\n" + "="*50)
    print("DAPO数据集示例展示:")
    print("="*50)
    
    # 显示前3个示例
    for i in range(min(3, len(dapo_dataset))):
        print(f"\n--- 示例 {i+1} ---")
        example = dapo_dataset[i]
        
        print("原始prompt字段:")
        for msg in example['prompt']:
            print(f"  {msg['role']}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
        
        print(f"\n原始answer字段:")
        print(f"  {example['answer'][:300]}{'...' if len(example['answer']) > 300 else ''}")
        
        # 如果有其他字段，也显示出来
        other_fields = [k for k in example.keys() if k not in ['prompt', 'answer']]
        if other_fields:
            print(f"\n其他字段: {other_fields}")
            for field in other_fields[:2]:  # 只显示前2个其他字段避免输出过多
                print(f"  {field}: {str(example[field])[:100]}{'...' if len(str(example[field])) > 100 else ''}")
    
    print("\n" + "="*50)
    print("数据集字段信息:")
    print(f"字段名称: {list(dapo_dataset.features.keys())}")
    print(f"数据集特征: {dapo_dataset.features}")
    print("="*50 + "\n")


def view_processed_dataset_info(processed_dataset, tokenizer, max_prompt_length, max_completion_length, max_seq_length):
    """查看处理后的数据集信息
    
    Args:
        processed_dataset: 处理后的数据集
        tokenizer: 分词器
        max_prompt_length: 最大prompt长度
        max_completion_length: 最大completion长度
        max_seq_length: 最大序列长度
    """
    print("\n" + "="*50)
    print("分词处理后的数据集信息:")
    print("="*50)
    print(f"处理后数据集大小: {len(processed_dataset)}")
    print(f"最大prompt长度: {max_prompt_length}")
    print(f"最大completion长度: {max_completion_length}")
    print(f"最大序列长度: {max_seq_length}")
    
    # 显示一个分词后的示例
    if len(processed_dataset) > 0:
        print(f"\n分词后的示例:")
        example = processed_dataset[0]
        
        # 应用chat模板并分词
        tokenized_example = tokenizer.apply_chat_template(
            example['prompt'], 
            add_generation_prompt=True, 
            tokenize=True
        )
        
        print(f"分词后的token数量: {len(tokenized_example)}")
        print(f"前50个token: {tokenized_example[:50]}")
        
        # 解码回文本查看
        decoded_text = tokenizer.decode(tokenized_example)
        print(f"\n解码后的文本:")
        print(f"{decoded_text[:500]}{'...' if len(decoded_text) > 500 else ''}")
    
    print("="*50 + "\n")


def setup_custom_tokens():
    """定义自定义的字符串标记"""
    # 定义一些特殊的字符串标记，用于指导模型生成我们想要的格式。
    # 这是一种"格式提示"或"模板化"的方法，让模型学会生成带有思考过程和最终答案的结构化输出。

    #  和  用于包裹模型的"思考过程"或"解题步骤"。
    reasoning_start = ""
    reasoning_end   = ""

    #  和  用于包裹模型给出的最终、简洁的答案。
    solution_start  = ""
    solution_end    = ""
    
    return reasoning_start, reasoning_end, solution_start, solution_end


def create_system_prompt(reasoning_start, reasoning_end, solution_start, solution_end):
    """创建系统提示"""
    # 定义系统提示（System Prompt）。这个提示会作为对话的初始指令，告诉模型它的角色和任务。
    # 在这里，我们要求模型先进行思考，将过程放在和之间，
    # 然后再将最终答案放在和之间。
    system_prompt = \
    f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
    
    return system_prompt


def create_chat_template(system_prompt, reasoning_start):
    """创建自定义聊天模板"""
    # 创建一个自定义的聊天模板（Chat Template）。
    # 聊天模板使用Jinja2语法，定义了如何将多轮对话（包含system, user, assistant等角色）格式化为单个字符串，
    # 以便输入给模型进行训练或推理。
    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"

    # 将模板中的占位符替换为我们之前定义的特定字符串。
    # 这样做可以使模板适应我们自定义的格式要求。
    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    
    return chat_template


def setup_tokenizer_template(tokenizer):
    """设置分词器的聊天模板"""
    # 定义用于标记模型推理过程和最终解的字符串标记
    reasoning_start = "<start_working_out>"     # 用于包裹模型的"思考过程"开始部分
    reasoning_end   = "<end_working_out>"       # 用于包裹模型的"思考过程"结束部分
    solution_start  = "<SOLUTION>"              # 用于包裹最终解答开始部分
    solution_end    = "</SOLUTION>"             # 用于包裹最终解答结束部分

    # 定义系统提示词，告诉模型应该如何组织回答：先推理，再给出答案
    system_prompt = \
    f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

    # 构建chat_template模板，控制如何拼接prompt
    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"

    # 将模板中作为字符串存在的变量替换为实际变量值（避免模板中引号包住变量名）
    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")

    # 将chat_template应用到tokenizer（假设这是一个支持chat_template的tokenizer）
    tokenizer.chat_template = chat_template
    
    return tokenizer, reasoning_start, reasoning_end, solution_start, solution_end, system_prompt


def test_chat_template(tokenizer, reasoning_start, reasoning_end, solution_start, solution_end):
    """测试聊天模板"""
    # 模拟一次tokenizer应用chat_template的过程（不进行tokenize，只展示结果）
    result = tokenizer.apply_chat_template([
        {"role" : "user", "content" : "What is 1+1?"},
        {"role" : "assistant", "content" : f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
        {"role" : "user", "content" : "What is 2+2?"},
    ], tokenize = False, add_generation_prompt = True)
    
    return result
