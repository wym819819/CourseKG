def knowledgepoint_prompt(input: str, domin: str) -> str:
    """ 知识点提取提示词

    Args:
        input (str): 待抽取的文本内容
        dmoin (str): 知识点相关的领域

    Returns:
        str: 组合后的提示词
    """
    return """
    {\"instruction\": \"你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体, 实体需要与""" + domin + """领域相关, 不存在的实体类型返回空列表。请按照JSON字符串的格式回答, 格式应该为 [\"entity1\", \"entity2\"]。\",
    \"schema\": \"知识点\",
    \"input\": \"""" + input + "\"}"
