import json
from loguru import logger
from abc import ABC, abstractmethod
import re


class KnowledgepointPrompt(ABC):

    def __init__(self) -> None:
        """ 知识点提示词类, 包含获取提示词和将模型返回处理成知识点列表两个方法
        """
        pass

    @abstractmethod
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

    def __init__(self, example: bool = True) -> None:
        """ 获取知识点提取提示词，来源于 DeepKE

        Args:
            example (bool, optional): 提示词中带有示例. Defaults to True.
        """
        self.example = example

    def get_prompt(self, content: str) -> str:
        """ 获取知识点提取提示词

        Args:
            content (str): 待抽取的文本内容

        Returns:
            str: 组合后的提示词
        """
        prompt = {
            "instruction":
            "你是专门进行实体抽取的专家。请从input中抽取出符合schema定义的实体，若不存在相关实体则返回空列表。请按数组的形式回答，格式应该为 [\"entity1\", \"entity2\"]。",
            "schema": {
                "知识点": "知识点实体类型表示特定领域或学科中的知识单元。"
            }
        }
        if self.example:
            prompt['examples'] = [{
                "input":
                """Python是一个简单、易读、易记的编程语言，而且是开源的，可以免费地自由使用。Python可以用类似英语的语法编写程序，编译起来也不费力，因此我们可以很轻松地使用Python。特别是对首次接触编程的人士来说，Python是最合适不过的语言。事实上，很多高校和大专院校的计算机课程均采用Python作为入门语言。""",
                "output": ["Python"]
            }, {
                "input":
                """神经网络的学习的目的是找到使损失函数的值尽可能小的参数。这是寻找最优参数的问题，解决这个问题的过程称为最优化（optimization）。遗憾的是，神经网络的最优化问题非常难。这是因为参数空间非常复杂，无法轻易找到最优解（无法使用那种通过解数学式一下子就求得最小值的方法）。
                   而且，在深度神经网络中，参数的数量非常庞大，导致最优化问题更加复杂。在前几章中，为了找到最优参数，我们将参数的梯度（导数）作为了线索。使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数，这个过程称为随机梯度下降法（stochastic gradient descent），
                   简称SGD。SGD是一个简单的方法，不过比起胡乱地搜索参数空间，也算是“聪明”的方法。但是，根据不同的问题，也存在比SGD更加聪明的方法。本节我们将指出SGD的缺点，并介绍SGD以外的其他最优化方法。""",
                "output": ["最优化", "随机梯度下降法(SGD)"]
            }, {
                "input":
                """顺便提一下，在图3-2的网络中，偏置b并没有被画出来。如果要明确地表示出b，可以像图3-3那样做。图3-3中添加了权重为b的输入信号1。这个感知机将x1、x2、1三个信号作为神经元的输入，将其和各自的权重相乘后，传送至下一个神经元。在下一个神经元中，计算这些加权信号的总和。
                   如果这个总和超过0，则输出1，否则输出0。另外，由于偏置的输入信号一直是1，所以为了区别于其他神经元，我们在图中把这个神经元整个涂成灰色。""",
                "output": []
            }]
        prompt['input'] = content
        return json.dumps(prompt, indent=4, ensure_ascii=False)

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
            res: list[str] = json.loads(response)
            res = [r.replace('"', '') for r in res]
            return list(set(res))
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

        prompt = {
            "instruction":
            "你是一个优秀的语言学家, 你的任务是从给定的句子中使用@@和##标记所有的知识点实体, 接下来是一些例子。",
            "example": [{
                "input":
                """Python是一个简单、易读、易记的编程语言，而且是开源的，可以免费地自由使用。Python可以用类似英语的语法编写程序，编译起来也不费力，因此我们可以很轻松地使用Python。特别是对首次接触编程的人士来说，Python是最合适不过的语言。事实上，很多高校和大专院校的计算机课程均采用Python作为入门语言。""",
                "output":
                """@@Python##是一个简单、易读、易记的编程语言，而且是开源的，可以免费地自由使用。@@Python##可以用类似英语的语法编写程序，编译起来也不费力，因此我们可以很轻松地使用@@Python##。特别是对首次接触编程的人士来说，@@Python##是最合适不过的语言。事实上，很多高校和大专院校的计算机课程均采用@@Python##作为入门语言。"""
            }, {
                "input":
                """神经网络的学习的目的是找到使损失函数的值尽可能小的参数。这是寻找最优参数的问题，解决这个问题的过程称为最优化（optimization）。遗憾的是，神经网络的最优化问题非常难。这是因为参数空间非常复杂，无法轻易找到最优解（无法使用那种通过解数学式一下子就求得最小值的方法）。
                   而且，在深度神经网络中，参数的数量非常庞大，导致最优化问题更加复杂。在前几章中，为了找到最优参数，我们将参数的梯度（导数）作为了线索。使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数，这个过程称为随机梯度下降法（stochastic gradient descent），
                   简称SGD。SGD是一个简单的方法，不过比起胡乱地搜索参数空间，也算是“聪明”的方法。但是，根据不同的问题，也存在比SGD更加聪明的方法。本节我们将指出SGD的缺点，并介绍SGD以外的其他最优化方法。""",
                "output":
                """神经网络的学习的目的是找到使损失函数的值尽可能小的参数。这是寻找最优参数的问题，解决这个问题的过程称为@@最优化##（optimization）。遗憾的是，神经网络的最优化问题非常难。这是因为参数空间非常复杂，无法轻易找到最优解（无法使用那种通过解数学式一下子就求得最小值的方法）。
                   而且，在深度神经网络中，参数的数量非常庞大，导致最优化问题更加复杂。在前几章中，为了找到最优参数，我们将参数的梯度（导数）作为了线索。使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数，这个过程称为@@随机梯度下降法##（stochastic gradient descent），
                   简称SGD。SGD是一个简单的方法，不过比起胡乱地搜索参数空间，也算是“聪明”的方法。但是，根据不同的问题，也存在比SGD更加聪明的方法。本节我们将指出SGD的缺点，并介绍SGD以外的其他最优化方法。"""
            }, {
                "input":
                """顺便提一下，在图3-2的网络中，偏置b并没有被画出来。如果要明确地表示出b，可以像图3-3那样做。图3-3中添加了权重为b的输入信号1。这个感知机将x1、x2、1三个信号作为神经元的输入，将其和各自的权重相乘后，传送至下一个神经元。在下一个神经元中，计算这些加权信号的总和。
                   如果这个总和超过0，则输出1，否则输出0。另外，由于偏置的输入信号一直是1，所以为了区别于其他神经元，我们在图中把这个神经元整个涂成灰色。""",
                "output":
                """顺便提一下，在图3-2的网络中，偏置b并没有被画出来。如果要明确地表示出b，可以像图3-3那样做。图3-3中添加了权重为b的输入信号1。这个感知机将x1、x2、1三个信号作为神经元的输入，将其和各自的权重相乘后，传送至下一个神经元。在下一个神经元中，计算这些加权信号的总和。
                   如果这个总和超过0，则输出1，否则输出0。另外，由于偏置的输入信号一直是1，所以为了区别于其他神经元，我们在图中把这个神经元整个涂成灰色。"""
            }],
            "input":
            content
        }
        return json.dumps(prompt, indent=4, ensure_ascii=False)

    @staticmethod
    def post_process(response: str) -> list[str]:
        """ 将模型返回处理成知识点列表

        Args:
            response (str): 模型输出

        Returns:
            list[str]: 知识点列表
        """
        pattern = r'@@(.*?)##'
        res: list[str] = re.findall(pattern, response)
        res = [r.replace('"', '') for r in res]
        return list(set(res))
