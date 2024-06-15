import json
from loguru import logger


class KnowledgepointPrompt:

    def __init__(self) -> None:
        """ 知识点提示词类, 包含获取提示词和将模型返回处理成知识点列表两个方法
        """

    def get_prompt(self, content: str) -> str:
        """ 获取知识点提取提示词

        Args:
            content (str): 待抽取的文本内容

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            str: 组合后的提示词
        """
        raise NotImplementedError

    @staticmethod
    def post_process(response: str) -> list[str]:
        """ 将模型返回处理成知识点列表

        Args:
            response (str): 模型输出

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            list[str]: 知识点列表
        """
        raise NotImplementedError


class DeepKEPrompt(KnowledgepointPrompt):

    def __init__(self, domain: str, example: bool = True) -> None:
        """ 获取知识点提取提示词，来源于 DeepKE

        Args:
            domain (str): 知识点相关的领域
            example (bool, optional): 提示词中带有示例. Defaults to True.
        """
        self.domain = domain
        self.example = example

    def get_prompt(self, content: str) -> str:
        """ 获取知识点提取提示词

        Args:
            content (str): 待抽取的文本内容

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
        """ if self.example else ''
        return """
            {\"instruction\": \"你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，实体需要与""" + self.domain + """领域相关，若不存在相关实体则返回空列表。请按数组形式回答，格式应该为 [\"entity1\", \"entity2\"]。\",
            \"schema\": {
                \"知识点\": \"知识点实体类型表示特定领域或学科中的知识单元。\"},
            """ + examples + "\"input\": \"" + content + "\"}"

    @staticmethod
    def post_process(response: str) -> list[str]:
        """ 将模型返回处理成知识点列表

        Args:
            response (str): 模型输出

        Returns:
            list[str]: 知识点列表
        """
        try:
            # 模型可能以md格式返回
            if response.startswith('```'):  # 多行代码
                response = '\n'.join(response.split('\n')[1:-1])
            elif response.startswith('`'):  # 单行代码
                response = response[1:-1]
            return json.loads(response)
        except Exception as e:
            logger.error(e)
            return []


class GPTNERPrompt(KnowledgepointPrompt):

    def get_prompt(self, content: str) -> str:
        """ 获取知识点提取提示词

        Args:
            content (str): 待抽取的文本内容

        Returns:
            str: 组合后的提示词
        """

        return """
            {\"instruction\": \"你是一个优秀的语言学家, 你的任务是从给定的句子中使用@@和##标记所有的知识点实体,接下来是一些例子。\",
            \"example\": [{
                \"input\": \" 函数的实现看上去有些复杂，但它执行的处理和求单变量的数值微分基本没有区别。现在，我们用这个函数实际计算一下梯度。这个梯度意味着什么呢？为了更好地理解，我们把的梯度画在图上。不过，这里我们画的是元素值为负梯度B的向量。 \",
                \"output\": \" @@函数##的实现看上去有些复杂，但它执行的处理和求单变量的数值微分基本没有区别。现在，我们用这个函数实际计算一下@@梯度##。这个@@梯度##意味着什么呢？为了更好地理解，我们把的@@梯度##画在图上。不过，这里我们画的是元素值为负@@梯度##B的向量。 \"}]
            }],
            \"input\": \"""" + content + "\"}"

    @staticmethod
    def post_process(response: str) -> list[str]:
        """ 将模型返回处理成知识点列表

        Args:
            response (str): 模型输出

        Returns:
            list[str]: 知识点列表
        """
        pass
