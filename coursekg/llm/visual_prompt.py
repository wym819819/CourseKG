from dataclasses import dataclass
from PIL import Image
import os
import json
from typing import Literal
import random


@dataclass
class Interaction:
    image_paths: str | list[str]
    question: str
    answer: str


class MiniCPMPrompt:

    def __init__(self, type_: Literal['ocr', 'ie']) -> None:
        """ MiniCPM提示词, 支持多轮对话, 多图对话和上下文学习

        Args:
            type_ (Literal['ocr', 'ie']): 提示词类型: OCR/信息提取
        """
        self.interactions: list[Interaction] = []
        self.type_ = type_
        if self.type_ == 'ocr':
            self.prompt = """将图片中识别到的文字转换文本输出。你必须做到：
1. 你的回答中严禁包含 “以下是根据图片内容生成的文本：”或者 “ 将图片中识别到的文字转换文本格式输出如下：” 等这样的提示语。
2. 不需要对内容进行解释和总结。
3. 代码包含在``` ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式。
4. 如果图片中包含图表，对图表形成摘要即可，无需添加例如“图片中的文本内容如下：”等这样的提示语。
再次强调，不要输出和识别到的内容无关的文字。"""
            self.sys_prompt = '你是一个OCR模型。'
        elif self.type_ == 'ie':
            self.prompt = '请帮我提取图片中的主要内容'
            self.sys_prompt = '你是一个能够总结图片内容的模型。'
        else:
            self.prompt = ''
            self.sys_prompt = None

    def use_examples(
        self,
        example_dataset_path: str = 'dataset/image_example'
    ) -> 'MiniCPMPrompt':
        """ 添加示例 (示例也可以是上文对话)

        Args:
            example_dataset_path (str, optional): 使用多模态模型上下文学习源数据地址文件夹. Defaults to 'dataset/image_example'.
        """
        self.interactions = []
        with open(os.path.join(example_dataset_path, 'example.json')) as f:
            examples = json.load(f)
            for line in examples:
                if line['type'] == 'ocr':
                    self.interactions.append(Interaction(line['image'],  self.prompt, line['output']))
        if len(self.interactions) > 5:
            self.interactions = random.sample(self.interactions, 5)
        return self

    def add_history(self, history: Interaction) -> 'MiniCPMPrompt':
        """ 输入历史记录以进行多轮问答

        Args:
            history (Interaction): 一条问答记录
        """
        self.interactions.append(history)
        return self

    def get_prompt(self, image_path: str | list[str]) -> list:
        """ 获取提示词

        Args:
            image_path (str, list[str]): 图片路径

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
                    Image.open(path).convert('RGB') for path in image_paths
                ]
            content.append(question)
            return content

        msgs = []
        for example in self.interactions:
            msgs.append({
                'role': 'user',
                'content': get_content(example.image_paths, example.question)
            })
            msgs.append({
                'role': 'assistant',
                'content': [example.answer]
            })

        msgs.append({
            'role': 'user',
            'content': get_content(image_path,  self.prompt)
        })
        return msgs

    def get_sys_prompt(self) -> str | None:
        return self.sys_prompt
