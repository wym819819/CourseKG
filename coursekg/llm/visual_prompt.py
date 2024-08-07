from dataclasses import dataclass
from PIL import Image


@dataclass
class Example:
    image_paths: str | list[str]
    question: str
    answer: str


class MiniCPMPrompt:

    def __init__(self, image_path: str | list[str], question: str) -> None:
        """ 多模态模型提示词, 支持多轮对话, 多图学习和上下文学习

        Args:
            image_path (str, list[str]): 图片路径
            question (str): 问题
        """
        self.examples: list[Example] = []
        self.image_path = image_path
        self.question = question

    def add_examples(self, example: Example) -> None:
        """ 添加示例 (示例也可以是上文对话)

        Args:
            example (Example): 示例
        """
        self.examples.append(example)

    def get_prompt(self) -> list:
        """ 获取提示词

        Returns:
            list: 组装后的提示词
        """

        def get_content(image_paths, question) -> list:
            if len(image_paths) == 0:
                content = []
            elif isinstance(image_paths, str):
                content = [Image.open(image_paths).convert('RGB')]
            else:
                content = [
                    Image.open(path).convert('RGB')
                    for path in image_paths
                ]
            content.append(question)
            return content

        msgs = [{{
                'role': 'user',
                'content': get_content(example.image_paths, example.question)
            }, {
                'role': 'assistant',
                'content': [example.answer]
            }} for example in self.examples]
         
        msgs.append({{
            'role':
            'user',
            'content': get_content(self.image_path, self.question)
        }})
        return msgs
