import json
from loguru import logger
from abc import ABC, abstractmethod
import re
import chardet


def is_garbled(s: str) -> bool:
    """ 检测字符串是否乱码

    Args:
        s (str): 待检测的字符串

    Returns:
        bool: 是否包含乱码
    """
    detected = chardet.detect(s.encode())
    return detected['encoding'] is None or detected['confidence'] < 0.5


entities = {"知识点": "知识点实体类型表示特定领域或学科中的知识单元。"}
relations = {
    "包含": "某一个知识点也可能包含另一个知识点",
    "相关": "某几个知识点之间存在相互联系、相互影响和相互作用",
    "顺序": "学习某几个知识点具有明显的先后关系，也就是学习某一个知识点后才能学习另一个，存在前驱后继的关系"
}


class Prompt(ABC):

    def __init__(self) -> None:
        """ 提示词类, 包含获取提示词和将模型返回处理成列表两个方法
        """
        pass

    @abstractmethod
    def get_prompt(self, content: str) -> str:
        """ 获取提取提示词

        Args:
            content (str): 待抽取的文本内容

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            str: 组合后的提示词
        """
        raise NotImplementedError

    @staticmethod
    def post_process(response: str) -> list[list[str]]:
        """ 将模型返回处理成三元组

        Args:
            response (str): 模型输出

        Returns:
            list[list[str]]: 知识点三元组列表 [head, relation, tail]
        """
        try:
            pattern = r'\[(.|\n)*\]'
            res = re.search(pattern,
                            response).group().replace("\\", "").replace(
                                '“', '"').replace('”', '"')  # 替换到可能出现的非法字符
            res: list[dict] = json.loads(res)
            res = [[triple['head'], triple['relation'], triple['tail']]
                   for triple in res
                   if triple['relation'] in list(relations.keys())
                   ]  # 只能是特定的关系类型
            return res
        except Exception as e:
            logger.error(e)
            return []


class CoTPrompt(Prompt):

    def __init__(self, example: bool = True) -> None:
        """ 获取提取提示词, 使用多种提示词优化, 包括CoT、基于动态检索的ICL

        Args:
            example (bool, optional): 提示词中带有示例. Defaults to True.
        """
        self.example = example

    def get_prompt(self, content: str) -> str:
        """ 获取提取提示词

        Args:
            content (str): 待抽取的文本内容

        Returns:
            str: 组合后的提示词
        """
        prompt = {
            "instruction":
            "你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组, 不存在的关系返回空列表。请按照以下格式进行回答: [{\"head\":\" \", \"relation\":\" \", \"tail\":\" \"}]。",
            "schema": {
                "entities": entities,
                "relations": relations
            }
        }
        if self.example:
            prompt['examples'] = [{
                "input":
                """如果有人问你现在有多幸福，你会如何回答呢？一般的人可能会给出诸如“还可以吧”或者“不是那么幸福”等笼统的回答。如果有人回答“我现在的幸福指数是10.23”的话，可能会把人吓一跳吧。因为他用一个数值指标来评判自己的幸福程度。这里的幸福指数只是打个比方，实际上神经网络的学习也在做同样的事情。
                   神经网络的学习通过某个指标表示现在的状态。然后，以这个指标为基准，寻找最优权重参数。和刚刚那位以幸福指数为指引寻找“最优人生”的人一样，神经网络以某个指标为线索寻找最优权重参数。神经网络的学习中所用的指标称为损失函数（loss function）。这个损失函数可以使用任意函数，但一般用均方误差和交叉熵误差等。
                   损失函数是表示神经网络性能的“恶劣程度”的指标，即当前的神经网络对监督数据在多大程度上不拟合，在多大程度上不一致。以“性能的恶劣程度”为指标可能会使人感到不太自然，但是如果给损失函数乘上一个负值，就可以解释为“在多大程度上不坏”，即“性能有多好”。并且，“使性能的恶劣程度达到最小”和“使性能的优良程度达到最大”是等价的，不管是用“恶劣程度”还是“优良程度”，做的事情本质上都是一样的。""",
                "output": [{
                    "head": "神经网络的学习",
                    "relation": "包含",
                    "tail": "损失函数"
                }]
            }, {
                "input":
                """神经网络的学习的目的是找到使损失函数的值尽可能小的参数。这是寻找最优参数的问题，解决这个问题的过程称为最优化（optimization）。遗憾的是，神经网络的最优化问题非常难。这是因为参数空间非常复杂，无法轻易找到最优解（无法使用那种通过解数学式一下子就求得最小值的方法）。
                   而且，在深度神经网络中，参数的数量非常庞大，导致最优化问题更加复杂。在前几章中，为了找到最优参数，我们将参数的梯度（导数）作为了线索。使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数，这个过程称为随机梯度下降法（stochastic gradient descent），
                   简称SGD。SGD是一个简单的方法，不过比起胡乱地搜索参数空间，也算是“聪明”的方法。但是，根据不同的问题，也存在比SGD更加聪明的方法。本节我们将指出SGD的缺点，并介绍SGD以外的其他最优化方法。""",
                "output": [{
                    "head": "最优化",
                    "relation": "包含",
                    "tail": "随机梯度下降法"
                }]
            }, {
                "input":
                """顺便提一下，在图3-2的网络中，偏置b并没有被画出来。如果要明确地表示出b，可以像图3-3那样做。图3-3中添加了权重为b的输入信号1。这个感知机将x1、x2、1三个信号作为神经元的输入，将其和各自的权重相乘后，传送至下一个神经元。在下一个神经元中，计算这些加权信号的总和。
                   如果这个总和超过0，则输出1，否则输出0。另外，由于偏置的输入信号一直是1，所以为了区别于其他神经元，我们在图中把这个神经元整个涂成灰色。""",
                "output": []
            }]
        prompt['input'] = content
        return json.dumps(prompt, indent=4, ensure_ascii=False)
