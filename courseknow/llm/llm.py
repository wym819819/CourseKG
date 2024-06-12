import os
from enum import Enum
import requests


class LLMType(Enum):
    Qwen_Turbo = 'qwen-turbo'


class LLM:

    def __init__(self, type: LLMType, api_key: str = None) -> None:
        """ 调用 LLM 的 API 服务

        Args:
            type (LLMType): 模型类型
            api_key (str, optional): API_KEY, 不输入则尝试从环境变量中获取. Defaults to None.
        """
        self.type = type
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        self.api_key = api_key

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            str: 模型输出
        """
        if self.type.name.split('_')[0] == 'Qwen':
            url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            body = {
                'model': self.type.value,
                "input": {
                    "messages": [{
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }, {
                        "role": "user",
                        "content": message
                    }]
                },
                "parameters": {
                    "result_format": "message"
                }
            }
            response = requests.post(url, headers=headers, json=body)
            return response.json(
            )['output']['choices'][0]['message']['content']
