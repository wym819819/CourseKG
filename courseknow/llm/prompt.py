def knowledge_point_prompt(content: str) -> str:
    """ 知识点提取提示词

    Args:
        content (str): 待抽取的文本内容

    Returns:
        str: 组合后的提示词
    """
    return '{\"instruction\": \"你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。\", \"schema\": [\"知识点\"], \"input\": \"' + content + '\"}'
