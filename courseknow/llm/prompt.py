def knowledgepoint_prompt(content: str, domain: str, example=True) -> str:
    """ 知识点提取提示词

    Args:
        content (str): 待抽取的文本内容
        domain (str): 知识点相关的领域
        example (bool, optional): 提示词中带有示例. Defaults to True.

    Returns:
        str: 组合后的提示词
    """
    examples = """
        \"example\": [{
            \"input\": \" 函数的实现看上去有些复杂，但它执行的处理和求单变量的数值微分基本没有区别。现在，我们用这个函数实际计算一下梯度。这个梯度意味着什么呢？为了更好地理解，我们把的梯度画在图上。不过，这里我们画的是元素值为负梯度B的向量。 \",
            \"output\": [
                    \"函数\",
                    \"梯度\"]}
        }],
    """ if example else ''
    return """
        {\"instruction\": \"你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，实体需要与""" + domain + """领域相关，若不存在相关实体则返回空列表。请按照数组形式回答，格式应该为 [\"entity1\", \"entity2\"]。\",
        \"schema\": {
            \"知识点\": \"知识点实体类型表示特定领域或学科中的知识单元。\"},
        """ + examples + "\"input\": \"" + content + "\"}"
