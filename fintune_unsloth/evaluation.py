"""评估和奖励函数模块"""

import re


def setup_regex_patterns(tokenizer, reasoning_end, solution_start, solution_end):
    """设置正则表达式模式"""
    # 构造匹配结束标签 "</SOLUTION>" 和可选的 eos_token（例如 <|endoftext|>）
    solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
        "(?:" + re.escape(tokenizer.eos_token) + ")?"

    # 构造完整的匹配模板，用于提取推理结果中的答案部分
    match_format = re.compile(
        rf"{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end_regex}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )

    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags = re.MULTILINE | re.DOTALL
    )
    
    return match_format, match_numbers


def match_format_exactly(completions, **kwargs):
    """精确格式匹配奖励函数"""
    # 需要从外部传入 match_format
    match_format = kwargs.get('match_format')
    
    scores = []  # 用于保存每个 completion 的得分

    for completion in completions:
        score = 0
        response = completion[0]["content"]  # 获取模型输出内容（假设为 messages 列表中的第一个 assistant 回复）

        # 如果输出内容能成功匹配指定格式（即包含完整 <start_working_out>...<SOLUTION>... 标签结构）
        if match_format.search(response) is not None:
            score += 3.0  # 匹配成功得 3 分（用于奖励格式正确的输出）

        scores.append(score)  # 保存该条 completion 的得分

    return scores  # 返回所有 completion 的格式匹配得分列表


def match_format_approximately(completions, **kwargs):
    """近似格式匹配奖励函数"""
    reasoning_end = kwargs.get('reasoning_end')
    solution_start = kwargs.get('solution_start')
    solution_end = kwargs.get('solution_end')
    
    scores = []  # 存储每个 completion 的近似格式匹配得分

    for completion in completions:
        score = 0
        response = completion[0]["content"]  # 获取该条生成结果的文本内容

        # 本函数不是精确匹配整段模板，而是检查关键标签是否恰好出现一次
        # 评分标准如下（每个关键标签出现一次加 0.5 分，出现多次或漏掉则减 1 分）：

        # <start_working_out> 不需要判断，因为一般在 prompt 中已加，无需重复奖励

        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0  # 检查 <end_working_out>
        score += 0.5 if response.count(solution_start)  == 1 else -1.0  # 检查 <SOLUTION>
        score += 0.5 if response.count(solution_end)    == 1 else -1.0  # 检查 </SOLUTION>

        scores.append(score)  # 保存该条 completion 的评分结果

    return scores  # 返回所有样本的评分结果列表


def check_answer(prompts, completions, answer, **kwargs):
    """答案检查奖励函数"""
    match_format = kwargs.get('match_format')
    
    # 获取原始问题（一般为 prompts 中最后一个 user 消息的内容）
    question = prompts[0][-1]["content"]

    # 提取每个 completion 的生成结果（假设为 assistant 的第一条回复）
    responses = [completion[0]["content"] for completion in completions]

    # 从每个 response 中提取 <SOLUTION> 标签内的答案（使用正则匹配）
    extracted_responses = [
        guess.group(1)  # 如果匹配成功，取出括号中的 group(1)
        if (guess := match_format.search(r)) is not None else None  # 否则为 None
        for r in responses
    ]

    scores = []  # 存储每个样本的评分结果
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)  # 若未成功提取答案，直接扣分
            continue

        # 正确答案完全一致，奖励 5 分
        if guess == true_answer:
            score += 5.0

        # 若去除空格后匹配成功，奖励略少（3.5 分）
        elif guess.strip() == true_answer.strip():
            score += 3.5

        # 否则，尝试进行"近似数值"匹配
        else:
            try:
                ratio = float(guess) / float(true_answer)  # 转换为 float 并计算比值
                if   ratio >= 0.9 and ratio <= 1.1:
                    score += 2.0  # 误差在 ±10% 内
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 1.5  # 误差在 ±20% 内
                else:
                    score -= 2.5  # 偏差太大，扣分
            except:
                score -= 4.5  # 无法转为数值（如包含文本、单位等），严重扣分

        scores.append(score)  # 记录当前样本的得分

    return scores  # 返回每个 completion 的分数


def check_numbers(prompts, completions, answer, **kwargs):
    """数值检查奖励函数"""
    match_numbers = kwargs.get('match_numbers')
    PRINTED_TIMES = 0  # 当前已打印次数

    PRINT_EVERY_STEPS = 5  # 每间隔多少步打印一次

    # 获取问题文本（通常为 prompts 中最后一个 user 消息）
    question = prompts[0][-1]["content"]

    # 提取模型生成的文本内容（假设每个 completion 是一个消息列表，取第一条）
    responses = [completion[0]["content"] for completion in completions]

    # 使用正则表达式 match_numbers 提取数字
    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []  # 存储得分结果

    # 控制打印调试信息（每隔 N 次打印一次，用于查看 sample 匹配结果）
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            '*'*20 + f"Question:\n{question}",  # 打印问题
            f"\nAnswer:\n{answer[0]}",           # 打印参考答案
            f"\nResponse:\n{responses[0]}",      # 打印模型生成
            f"\nExtracted:\n{extracted_responses[0]}"  # 打印提取结果
        )
    PRINTED_TIMES += 1  # 打印次数增加

    # 核心评分逻辑
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)  # 没提取出数字，直接扣分
            continue

        try:
            # 去除空格，转换为 float；guess 先去掉千位分隔符（例如 123,456）
            true_answer = float(true_answer.strip())
            guess = float(guess.strip().replace(",", ""))

            # 如果完全数值一致，得分 3.5；否则扣分
            scores.append(3.5 if guess == true_answer else -1.5)

        except:
            scores.append(0)  # 解析失败不给分也不扣分
            continue

    return scores  # 返回所有样本的得分列表


def create_reward_functions(tokenizer, reasoning_end, solution_start, solution_end):
    """创建奖励函数列表"""
    match_format, match_numbers = setup_regex_patterns(tokenizer, reasoning_end, solution_start, solution_end)
    
    # 创建带有预设参数的奖励函数
    def match_format_exactly_wrapper(completions, **kwargs):
        kwargs['match_format'] = match_format
        return match_format_exactly(completions, **kwargs)
    
    def match_format_approximately_wrapper(completions, **kwargs):
        kwargs['reasoning_end'] = reasoning_end
        kwargs['solution_start'] = solution_start
        kwargs['solution_end'] = solution_end
        return match_format_approximately(completions, **kwargs)
    
    def check_answer_wrapper(prompts, completions, answer, **kwargs):
        kwargs['match_format'] = match_format
        return check_answer(prompts, completions, answer, **kwargs)
    
    def check_numbers_wrapper(prompts, completions, answer, **kwargs):
        kwargs['match_numbers'] = match_numbers
        return check_numbers(prompts, completions, answer, **kwargs)
    
    reward_funcs = [
        match_format_exactly_wrapper,       # 检查是否严格符合 <start_working_out>...<SOLUTION> 格式，匹配得分高
        match_format_approximately_wrapper, # 检查是否大致有格式标签，宽松评分
        check_answer_wrapper,               # 与参考答案对比，进行精确匹配、模糊匹配、近似数值匹配
        check_numbers_wrapper,              # 提取数值进行比较（用于数学题）
    ]
    
    return reward_funcs
